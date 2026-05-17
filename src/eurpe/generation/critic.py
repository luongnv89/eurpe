"""Source-grounded critic agent — Task 3.2 / issue #16.

The critic takes a *prior* draft (with its citations and the original
request that produced it) and emits a critique: a free-text assessment
plus the structured list of call/profile requirements that were checked
on this pass. The downstream :class:`CriticLoopWorkflow` then uses the
critique to drive a refinement pass through
:class:`SectionGenerationWorkflow.run`, producing a new draft.

Why the critic is a separate seam (not folded into the workflow)
-----------------------------------------------------------------
The workflow's job is retrieve → prompt → generate → validate →
assemble. Folding a critic into that contract would either (a) make
the workflow output two different shapes (draft vs. critique) or (b)
push the critic prompt into the same builder that already handles the
section prompt. Keeping :class:`CriticAgent` separate means:

* The single-pass entry point (:meth:`SectionGenerationWorkflow.run`)
  keeps its existing contract — every Sprint 1 caller and every test
  in :mod:`tests.test_generation_workflow` keeps passing untouched.
* The critic's prompt evolves independently of the section-drafting
  prompt; the two have different audiences (the section prompt is
  consumed by an LLM that produces prose; the critic prompt asks the
  LLM to *judge* prose) and different stability requirements.
* The critic gets its own LLM client reference, which lets a future
  caller wire a smaller / cheaper model for critique than for
  generation (a common pattern).

Deterministic ``requirements_checked``
--------------------------------------
The list of requirements the critic checks on each iteration is
constructed *deterministically* from the request inputs — never
parsed back from the LLM's output. This is the safest design under
the offline contract: the deterministic LLM stub does not produce
JSON, and real LLMs are unreliable JSON emitters. Building the list
server-side from
``profile.expected_outputs[section]`` ∪
``topic_context.section_guidance.keys()`` ∪
``{"default-section-guidance"}`` gives us:

* A guaranteed-non-empty list (the sentinel ``default-section-guidance``
  is always present).
* A list whose contents the operator can verify by looking at the
  profile YAML and the TopicContext they supplied.
* An AC #3 ("which call/profile requirements were checked") signal
  that is provable without any LLM round-trip.

The critic LLM is then *instructed* to evaluate against those
requirements in plain prose — its job is to judge, not to enumerate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eurpe.generation.llm import LLMClient
from eurpe.generation.models import GenerationDraft
from eurpe.schema import SectionType

if TYPE_CHECKING:
    from eurpe.generation.profiles import DraftingProfile
    from eurpe.intake.models import TopicContext

logger = logging.getLogger(__name__)


#: Hard cap on the critique text the critic returns. Matches the cap on
#: :class:`IterationRecord.critique_text` so a longer critique is truncated
#: at the seam rather than rejected by Pydantic at construction time.
_CRITIQUE_MAX_CHARS = 4000


#: Sentinel requirement always present in the ``requirements_checked``
#: list. Guarantees a non-empty list even when no profile and no topic
#: context were supplied, which keeps the AC #3 invariant simple.
_DEFAULT_REQUIREMENT_SENTINEL = "default-section-guidance"


def build_requirements_checked(
    *,
    section_type: SectionType,
    profile: DraftingProfile | None,
    topic_context: TopicContext | None,
) -> list[str]:
    """Return the list of call/profile requirements to check this iteration.

    Order is stable (sentinel first, then profile entries in their
    declared order, then topic section_guidance entries in dict
    insertion order) so a snapshot test can pin the list verbatim.
    Duplicates are dropped while preserving first-seen position so a
    profile/topic that names the same requirement twice yields one
    entry — keeps the critique cleanly itemised.

    The :class:`SectionType` is required even when no profile and no
    topic context are supplied so the sentinel name can be expanded
    into a section-specific form later without breaking callers.
    """

    seen: set[str] = set()
    items: list[str] = []

    def _push(label: str) -> None:
        label = label.strip()
        if not label or label in seen:
            return
        seen.add(label)
        items.append(label)

    # Sentinel first so the list is never empty even when the caller
    # supplies neither a profile nor a topic context. Keep the literal
    # string (not f-string with section) so the test harness can pin
    # it independently of the section under test.
    _push(_DEFAULT_REQUIREMENT_SENTINEL)

    if profile is not None:
        for output in profile.get_expected_outputs(section_type):
            _push(output)

    if topic_context is not None:
        # The TopicContext model exposes section_guidance as a dict
        # keyed by SectionType; the *keys* (cast to their string value)
        # are the requirement names the critic should check. The
        # *values* are the guidance prose, which lands in the critic
        # prompt (not in this list).
        for known_section in topic_context.section_guidance:
            _push(f"topic:{known_section.value}")

    return items


def _format_requirements_block(requirements: list[str]) -> str:
    """Render the requirements list as a Markdown bullet list for the prompt."""

    return "\n".join(f"* {req}" for req in requirements)


def _format_citations_block(draft: GenerationDraft) -> str:
    """Render the prior draft's citations as a compact ``[N] STATUS — call`` list."""

    if not draft.citations:
        return "(no citations attached to the prior draft)"
    lines = []
    for c in draft.citations:
        lines.append(
            f"[{c.citation_id}] {c.source_status.value.upper()} — "
            f"{c.programme.value} call {c.call_id}"
        )
    return "\n".join(lines)


def build_critic_prompt(
    *,
    draft: GenerationDraft,
    requirements: list[str],
    iteration_index: int,
    max_iterations: int,
) -> str:
    """Build the prompt the LLM receives when asked to critique a draft.

    The prompt explicitly names every requirement the critic should
    consider (sourced from :func:`build_requirements_checked`) so the
    LLM's prose has somewhere concrete to anchor. The prompt also
    states the iteration number and ceiling so the LLM can scale its
    suggestions (an early iteration may warrant structural rewrites; a
    later iteration is closer to polish).

    The format is intentionally not JSON: the deterministic stub
    cannot reliably emit JSON and real LLMs are unreliable JSON
    emitters. The critic's job is prose evaluation; the structured
    requirements list is built server-side, not parsed from the LLM.
    """

    citations_block = _format_citations_block(draft)
    requirements_block = _format_requirements_block(requirements)
    section_title = draft.section_type.value.replace("_", " ").title()

    return (
        "# Critique a proposal section draft\n"
        "\n"
        f"You are reviewing iteration {iteration_index - 1} of {max_iterations - 1} "
        f"of a draft for an EU research proposal **{section_title}** section. "
        "Your job is to identify what should improve in the next revision, "
        "grounded in the supplied call/profile requirements and the "
        "evidence already cited.\n"
        "\n"
        "## Prior draft\n"
        f"{draft.text}\n"
        "\n"
        "## Citations the prior draft used\n"
        f"{citations_block}\n"
        "\n"
        "## Call / profile requirements to check\n"
        "Evaluate the prior draft against each of the requirements below. "
        "For each, say whether it is addressed, partly addressed, or missing. "
        "Suggest specific improvements that the next revision should make.\n"
        "\n"
        f"{requirements_block}\n"
        "\n"
        "## Output\n"
        "Write a concise critique in markdown. Do not rewrite the draft; "
        "produce only the assessment and improvement suggestions. Keep it "
        "under 1500 words.\n"
    )


class CriticAgent:
    """Wrap an :class:`LLMClient` with the critic-prompt + requirements logic.

    Stateless aside from the injected LLM client. Safe to share across
    iterations and across requests (LLM clients are themselves
    thread-safe by contract).

    A single :meth:`critique` call returns a ``(critique_text,
    requirements_checked)`` tuple — both halves of an
    :class:`~eurpe.generation.models.IterationRecord` that the caller
    (:class:`CriticLoopWorkflow`) needs to assemble alongside the
    refined draft.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        """Read-only access to the configured critic LLM — useful for tests / logs."""

        return self._llm

    def critique(
        self,
        *,
        prior_draft: GenerationDraft,
        profile: DraftingProfile | None,
        topic_context: TopicContext | None,
        iteration_index: int,
        max_iterations: int,
    ) -> tuple[str, list[str]]:
        """Return ``(critique_text, requirements_checked)`` for the prior draft.

        ``iteration_index`` is the 1-indexed *next* iteration the loop
        is about to produce (so on the critic call before iteration 2
        the value is 2). ``max_iterations`` is the user-configured
        ceiling (1..5). Both are surfaced in the prompt so the LLM can
        scale its suggestions.

        Raises :class:`~eurpe.generation.errors.LLMUnavailableError`
        when the LLM endpoint is unreachable — propagated unchanged
        from the underlying client so callers get one consistent
        recovery surface.
        """

        requirements = build_requirements_checked(
            section_type=prior_draft.section_type,
            profile=profile,
            topic_context=topic_context,
        )
        prompt = build_critic_prompt(
            draft=prior_draft,
            requirements=requirements,
            iteration_index=iteration_index,
            max_iterations=max_iterations,
        )
        raw = self._llm.generate(prompt)

        # Truncate at the seam rather than letting Pydantic reject a
        # long critique on construction. The cap matches
        # :data:`_CRITIQUE_MAX_CHARS` and ``IterationRecord.critique_text``.
        if len(raw) > _CRITIQUE_MAX_CHARS:
            logger.debug(
                "Critique text exceeded %d chars (was %d); truncating.",
                _CRITIQUE_MAX_CHARS,
                len(raw),
            )
            raw = raw[: _CRITIQUE_MAX_CHARS - 1] + "…"

        # Guard against an empty LLM response. The deterministic stub
        # always produces text; a real LLM that returns "" leaves the
        # critique unfindable — surface a safe default rather than
        # raising, so the loop keeps moving and the operator can read
        # the empty critique in the iteration record.
        if not raw.strip():
            raw = (
                "(no critique produced — the LLM returned an empty response; "
                "review the prior draft manually)"
            )

        return raw, requirements
