"""Configurable critic loop — Task 3.2 / issue #16.

The loop drives the single-pass :class:`SectionGenerationWorkflow`
through up to *N* refinement iterations (1 ≤ N ≤ 5). Each iteration:

1. Asks the :class:`CriticAgent` to review the prior draft against the
   call/profile requirements list and emit a critique.
2. Builds an *augmented* :class:`GenerationRequest` whose
   ``user_intent`` is extended with the critique so the workflow's
   prompt-builder injects the critic's feedback into the next draft.
3. Runs the workflow to produce a refined draft.
4. Diffs the prior vs. refined drafts to build a ``changes_summary``,
   then assembles an :class:`IterationRecord` and appends it to the
   refined draft's ``iterations`` list (carrying forward any prior
   iterations so the full history travels with the draft).

The loop is *per-iteration* rather than a server-side multi-pass call:
the public entry point :meth:`CriticLoopWorkflow.iterate` performs
exactly one critic+regenerate pass per call. This is what satisfies
AC #2 ("user can stop the loop after any completed iteration") — the
client decides whether to call again or to accept the current draft.

Stopping is therefore a *client-side* concept; the loop has no
cancellation token, no thread state, and no persistent session. The
:class:`GenerationService` re-runs the loop for each request, and the
React workspace's "Accept draft" button simply stops calling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from eurpe.generation.critic import CriticAgent, build_requirements_checked
from eurpe.generation.errors import GenerationError
from eurpe.generation.models import (
    GenerationDraft,
    GenerationRequest,
    IterationRecord,
)
from eurpe.generation.workflow import SectionGenerationWorkflow

if TYPE_CHECKING:
    from eurpe.generation.profiles import DraftingProfile

logger = logging.getLogger(__name__)


#: Hard ceiling on the user-facing iteration count. Matches AC #1
#: ("User can set critic iterations between 1 and 5") and the
#: ``IterationRecord.iteration_index`` field's ``le=5`` constraint so
#: a request that bypasses Pydantic-level validation still hits this
#: guard at the workflow boundary.
MAX_ITERATIONS_CEILING: int = 5


#: Default ``max_iterations`` when the caller does not supply one.
#: Mirrors the issue body: "defaulting to 3 revision cycles".
DEFAULT_MAX_ITERATIONS: int = 3


@dataclass(frozen=True)
class IterationResult:
    """One pass through the critic loop, returned by :meth:`CriticLoopWorkflow.iterate`.

    The ``draft`` carries the *new* refined text and the full
    cumulative iteration history (prior iterations + the new one) so a
    caller can render or persist the most recent draft without holding
    the prior one. The ``stopped`` flag is set when this iteration is
    the last permitted (``iteration_index == max_iterations``) — the
    UI can flip the "Refine" button into a disabled state to
    communicate that no further iteration is possible without
    re-configuring the cap.
    """

    draft: GenerationDraft
    iteration_index: int
    max_iterations: int
    stopped: bool


def _clamp_max_iterations(value: int) -> int:
    """Clamp ``value`` into ``[1, MAX_ITERATIONS_CEILING]`` with logging.

    Raises :class:`ValueError` on a non-positive request because that
    is a programming error, not a recoverable input. Out-of-range
    positive values are clamped + logged so a CLI / API caller that
    asked for 10 iterations gets 5 with a warning rather than a hard
    rejection (matching the issue's "between 1 and 5" wording, which
    bounds rather than rejects out-of-range values).
    """

    if value < 1:
        raise ValueError(f"max_iterations must be >= 1, got {value}")
    if value > MAX_ITERATIONS_CEILING:
        logger.warning(
            "max_iterations=%d exceeds the ceiling of %d; clamping.",
            value,
            MAX_ITERATIONS_CEILING,
        )
        return MAX_ITERATIONS_CEILING
    return value


def _build_changes_summary(prior: GenerationDraft, refined: GenerationDraft) -> str:
    """Build the per-iteration ``changes_summary`` from two drafts.

    Combines a structural diff (char-count delta, citation-count
    delta) with a single-line natural description. Mirrors how
    ``git diff --stat`` summarises a change: enough signal for the
    operator to see at a glance whether the iteration grew or shrank
    the draft, without including the raw text (which the operator can
    inspect in the draft itself).
    """

    char_delta = len(refined.text) - len(prior.text)
    citation_delta = len(refined.citations) - len(prior.citations)

    if char_delta == 0 and citation_delta == 0:
        narrative = (
            "No structural change — text length and citation count are identical "
            "to the prior draft. Review the critique for substantive feedback."
        )
    elif char_delta > 0:
        narrative = "Draft expanded — added detail in response to the critique."
    else:
        narrative = "Draft tightened — removed or condensed content per the critique."

    sign_char = "+" if char_delta >= 0 else ""
    sign_cite = "+" if citation_delta >= 0 else ""
    return f"{narrative} text {sign_char}{char_delta} chars, citations {sign_cite}{citation_delta}."


def _augment_request_with_critique(
    base_request: GenerationRequest,
    critique_text: str,
) -> GenerationRequest:
    """Return a copy of ``base_request`` with the critique woven into ``user_intent``.

    The user's original intent stays at the top of the field so the
    LLM keeps treating it as the primary directive; the critic's
    feedback follows under a labelled "Critic feedback" sub-block.
    Using ``GenerationRequest.model_copy`` preserves every other
    field unchanged (programme filter, topic context, top-k, etc.) so
    each refinement runs against the same retrieval/profile envelope
    as the original.
    """

    augmented_intent = (
        f"{base_request.user_intent}\n"
        "\n"
        "## Critic feedback from the prior iteration\n"
        f"{critique_text}\n"
        "\n"
        "Address the critic's points in this revision while preserving "
        "the original intent above."
    )
    return base_request.model_copy(update={"user_intent": augmented_intent})


class CriticLoopWorkflow:
    """Per-iteration critic+regenerate workflow.

    Construct once per service / request lifetime with the same
    :class:`SectionGenerationWorkflow` the single-pass path uses (so
    every retriever / LLM / analytics knob carries through) plus the
    :class:`CriticAgent` that holds the critic LLM client. Call
    :meth:`iterate` once per iteration; the caller (CLI / API / UI)
    orchestrates the loop and decides when to stop.

    Why iteration is one method (not a generator)
    ---------------------------------------------
    A generator would couple the loop to a single in-process caller,
    which makes the API / HTTP path awkward (one HTTP request per
    iteration would need to resume a generator, which forces server
    state). Exposing one synchronous ``iterate()`` that takes the prior
    draft as input keeps the workflow stateless: the *draft* carries
    all the state the loop needs, and the *client* drives the loop.
    """

    def __init__(
        self,
        *,
        workflow: SectionGenerationWorkflow,
        critic: CriticAgent,
    ) -> None:
        self._workflow = workflow
        self._critic = critic

    @property
    def workflow(self) -> SectionGenerationWorkflow:
        """Read-only access — useful for tests / logs."""

        return self._workflow

    @property
    def critic(self) -> CriticAgent:
        """Read-only access — useful for tests / logs."""

        return self._critic

    def iterate(
        self,
        *,
        prior_draft: GenerationDraft,
        request: GenerationRequest,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        profile: DraftingProfile | None = None,
    ) -> IterationResult:
        """Run exactly one critic+regenerate pass on top of ``prior_draft``.

        ``prior_draft`` MUST be the draft produced by the previous
        iteration (or the initial single-pass draft for the first
        critic call). ``request`` is the *original* request that
        produced the initial draft; the loop augments its
        ``user_intent`` with the critique for the regeneration, leaving
        every other field unchanged.

        The returned :class:`IterationResult` carries:

        * ``draft`` — the refined draft with the full cumulative
          iteration history.
        * ``iteration_index`` — the 1-indexed iteration number this
          call produced (the implicit first pass is 1, so the first
          critic call returns 2, the second returns 3, etc.).
        * ``max_iterations`` — echoed back so the caller can drive UI
          state without re-passing the cap.
        * ``stopped`` — True when ``iteration_index ==
          max_iterations`` and the loop must end. The caller MAY ignore
          this and stop earlier; the flag is advisory.

        Raises :class:`GenerationError` when ``max_iterations`` is 1
        (no critic passes are permitted) or when the next iteration
        would exceed the cap. :class:`~eurpe.generation.errors.LLMUnavailableError`
        propagates from the critic LLM or the section LLM unchanged.
        """

        cap = _clamp_max_iterations(max_iterations)
        if cap < 2:
            raise GenerationError(
                "Critic loop cannot run with max_iterations<2; the loop only "
                "makes sense when at least one refinement pass is permitted. "
                f"Got max_iterations={max_iterations}."
            )

        # The new iteration's 1-indexed position. The implicit single-
        # pass draft is iteration 1, so the *first* critic pass is 2.
        next_index = prior_draft.total_iterations() + 1
        if next_index > cap:
            raise GenerationError(
                f"Critic loop already at max_iterations={cap}; cannot run "
                f"another pass (prior draft already has "
                f"{prior_draft.total_iterations()} iterations)."
            )

        # Critique first. The deterministic requirements list is built
        # by the agent and returned alongside so we can carry the exact
        # same list onto the IterationRecord — no re-derivation.
        critique_text, requirements = self._critic.critique(
            prior_draft=prior_draft,
            profile=profile,
            topic_context=request.topic_context,
            iteration_index=next_index,
            max_iterations=cap,
        )

        # Regenerate with the augmented intent. ``iteration_count`` is
        # forwarded so the DraftCompletedEvent the workflow emits
        # carries the right 1-indexed iteration number (matching
        # ``next_index``). Privacy contract: only the integer is
        # passed through to analytics — never the critique text.
        augmented = _augment_request_with_critique(request, critique_text)
        refined_draft = self._workflow.run(
            augmented,
            profile=profile,
            iteration_count=next_index,
        )

        # Stitch the iteration history. The refined draft starts with
        # iterations=[]; we prepend the prior_draft's history and
        # append the new record so the cumulative list travels with
        # the most recent draft.
        new_record = IterationRecord(
            iteration_index=next_index,
            changes_summary=_build_changes_summary(prior_draft, refined_draft),
            requirements_checked=requirements,
            critique_text=critique_text,
        )
        cumulative_iterations = list(prior_draft.iterations) + [new_record]
        refined_with_history = refined_draft.model_copy(
            update={"iterations": cumulative_iterations}
        )

        # Sanity: requirements_checked is also derivable from the
        # request; assert they match so a future bug where the critic
        # rebuilds the list from a different input would trip a test
        # (the tests pin this equality).
        expected_requirements = build_requirements_checked(
            section_type=request.section_type,
            profile=profile,
            topic_context=request.topic_context,
        )
        if expected_requirements != requirements:  # pragma: no cover - defensive
            logger.warning(
                "Critic returned a different requirements list than the "
                "expected deterministic set; using critic's list. expected=%r "
                "actual=%r",
                expected_requirements,
                requirements,
            )

        return IterationResult(
            draft=refined_with_history,
            iteration_index=next_index,
            max_iterations=cap,
            stopped=(next_index >= cap),
        )
