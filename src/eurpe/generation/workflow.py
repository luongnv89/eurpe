"""Section-generation workflow — the linear retrieve → prompt → generate → assemble pipeline.

What this module is (and is not)
--------------------------------
This is the **POC pipeline** for Sprint 1 (Issue #6). The PRD names
LangGraph as the orchestration layer that will eventually host a
multi-agent workflow (Supervisor, Retriever, Generator, Critic — see
``prd.md`` § "Architecture"). LangGraph is intentionally NOT
introduced here because the critic loop and human-in-the-loop pauses
that justify its weight only land in Sprint 3 (Issue #16). Adding it
now would mean either a one-node trivial graph (no value) or
speculative scaffolding for nodes that haven't been designed yet
(both worse than not adding it).

Instead the workflow is a single class with one public method
(:meth:`SectionGenerationWorkflow.run`) that calls four small,
named, testable steps in sequence. The seams are deliberately clean
so the LangGraph wrapper in Issue #16 can call the same steps as
nodes — the only thing that has to change is the orchestration, not
the step implementations.

Acceptance criteria covered
---------------------------
* **AC1.** :meth:`SectionGenerationWorkflow.run` accepts any
  :class:`~eurpe.schema.SectionType` (Methodology and Impact Pathway
  are exercised explicitly in the tests). The section type drives
  both the retrieval filter and the prompt's section-guidance block.
* **AC2.** Every returned :class:`~eurpe.generation.GenerationDraft`
  carries a list of :class:`~eurpe.generation.CitationRef` records,
  each with a non-None ``source_status``. The workflow builds the
  list from the retrieval results and validates that the LLM's
  output references only valid citation indices — a hallucinated
  ``[N]`` raises :class:`GenerationError`.
* **AC3.** The whole pipeline runs offline:
  :class:`~eurpe.retrieval.DeterministicHashEmbedder` +
  :class:`~eurpe.generation.DeterministicLLMClient` need no network,
  and the workflow exercises no other external endpoints. The
  ``test_offline_end_to_end_under_no_network_fixture`` test pins
  this end-to-end.

Drafting profiles (Task 2.1)
-----------------------------
The workflow optionally accepts a
:class:`~eurpe.generation.profiles.DraftingProfile` to apply
programme-specific guidance. When a profile is provided, the workflow
records its name in the :class:`GenerationDraft` for audit and
traceability.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import TYPE_CHECKING

from eurpe.analytics import (
    AnalyticsLogger,
    DraftCompletedEvent,
    DraftStartedEvent,
    EventType,
)
from eurpe.generation.errors import GenerationError
from eurpe.generation.llm import LLMClient
from eurpe.generation.models import CitationRef, GenerationDraft, GenerationRequest
from eurpe.generation.prompt import SectionPromptBuilder
from eurpe.retrieval import RetrievalResult, SourceStatusAwareRetriever

if TYPE_CHECKING:
    from eurpe.generation.profiles import DraftingProfile

logger = logging.getLogger(__name__)


#: Regex matching ``[N]`` citation markers in the generated draft text.
#: Matches one- or two-digit numbers, which covers the request's
#: ``top_k_examples`` cap of 20. A simple regex is enough for the POC;
#: it does NOT try to skip markers inside fenced code blocks (the LLM
#: rarely emits those for prose sections, and the validation only fires
#: on out-of-range indices anyway).
_CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")


def _scan_citation_markers(text: str) -> list[int]:
    """Return every ``[N]`` integer that appears in ``text``, in order.

    Public-by-convention helper: pulled out so the validation logic in
    :meth:`SectionGenerationWorkflow._validate_citations` is easy to
    test in isolation. Duplicates are preserved so a test can assert
    on "the model cited [1] three times" if relevant.
    """

    return [int(m.group(1)) for m in _CITATION_MARKER.finditer(text)]


class SectionGenerationWorkflow:
    """Orchestrates retrieve → prompt → generate → validate → assemble.

    Construct once with a configured retriever and LLM client; call
    :meth:`run` per request. The instance is stateless apart from the
    immutable references it holds, so it is safe to share across
    threads (subject to the underlying retriever / LLM being
    thread-safe — Chroma is, the LLM clients here are).

    Why the workflow takes a constructed retriever rather than
    ``index + policy``
    -------------------------------------------------------------
    Per the design rationale in :mod:`eurpe.generation.models`,
    source-status policy is a constructor-time setting on the
    retriever (only ``lessons_learned`` is per-call). Asking callers
    to hand in a configured retriever lets them build it once with
    ``RetrievalPolicy(include_esr=False, ...)`` and re-use it across
    many drafting calls. The CLI does this in ``cli.py``.
    """

    def __init__(
        self,
        *,
        retriever: SourceStatusAwareRetriever,
        llm: LLMClient,
        prompt_builder: SectionPromptBuilder | None = None,
        analytics: AnalyticsLogger | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        # The default prompt builder is stateless and reusable. Allowing
        # the caller to inject one keeps the door open for a future
        # programme- or tenant-specific builder without forcing every
        # call site to construct one.
        self._prompt_builder = prompt_builder or SectionPromptBuilder()
        # ``analytics`` is optional so tests can build workflows without
        # touching the analytics log. When set, the workflow emits
        # :class:`DraftStartedEvent` + :class:`DraftCompletedEvent`
        # around each ``run()``. Emission is wrapped in try/except so a
        # downstream analytics failure cannot break drafting (the
        # workflow's primary contract is to produce a draft).
        self._analytics = analytics

    @property
    def llm(self) -> LLMClient:
        """Read-only access to the configured LLM — useful for tests / logs."""

        return self._llm

    @property
    def retriever(self) -> SourceStatusAwareRetriever:
        """Read-only access to the configured retriever — useful for tests / logs."""

        return self._retriever

    def run(
        self,
        request: GenerationRequest,
        *,
        profile: DraftingProfile | None = None,
        iteration_count: int = 1,
    ) -> GenerationDraft:
        """Drive the full pipeline for one request and return a :class:`GenerationDraft`.

        Steps (each implemented as a private method so the LangGraph
        wrapper in Issue #16 can adopt them as nodes):

        1. :meth:`_retrieve` — call the retriever with the request's
           filters and the user intent as the query.
        2. :meth:`_build_prompt` — turn the retrieved evidence into a
           prompt string and a structured citation list (1:1 with
           ``[N]`` markers).
        3. :meth:`_generate` — call the LLM with the prompt.
        4. :meth:`_validate_citations` — scan the generated text for
           ``[N]`` markers and confirm every index falls within the
           citation list. Raises :class:`GenerationError` on
           hallucinated markers.
        5. Assemble a :class:`GenerationDraft` and return it.

        If ``profile`` is provided, programme-specific section guidance
        and expected outputs are used. The profile name is recorded in
        the draft for audit and traceability.

        ``iteration_count`` is recorded on the
        :class:`DraftCompletedEvent` emitted at the end of the run.
        The single-pass entry point (Sprint 1 / issue #6) leaves it at
        the default ``1``; the Sprint 3 critic loop (issue #16) passes
        the 1-indexed iteration number through so the analytics log
        carries per-iteration counts. Content-safe: only the integer
        crosses the analytics boundary — never the critique text.

        Errors:

        * :class:`GenerationError` (base) — anything that goes wrong
          inside the workflow that is not a recoverable LLM
          connection problem.
        * :class:`~eurpe.generation.LLMUnavailableError` — propagated
          from the LLM client when the daemon is unreachable.
        """

        start_time_ns = time.monotonic_ns()
        self._emit_draft_started(request, profile=profile)

        results = self._retrieve(request)
        prompt, citations = self._build_prompt(request, results, profile=profile)
        text = self._generate(prompt)
        self._validate_citations(text=text, citations=citations)

        generation_time_ms = (time.monotonic_ns() - start_time_ns) // 1_000_000
        self._emit_draft_completed(
            request,
            citations=citations,
            generation_time_ms=generation_time_ms,
            profile=profile,
            iteration_count=iteration_count,
        )

        return GenerationDraft(
            section_type=request.section_type,
            text=text,
            citations=citations,
            prompt_used=prompt,
            model=self._llm.model,
            request=request,
            drafting_profile=profile.name if profile is not None else None,
            topic_context=request.topic_context,
        )

    # ------------------------------------------------------------------
    # Steps — kept private but small so the public API is the contract
    # ------------------------------------------------------------------

    def _retrieve(self, request: GenerationRequest) -> list[RetrievalResult]:
        """Build the retrieval query and call the retriever.

        The query is the user intent verbatim (no synthesis) because:
        (a) the user's wording is the most honest signal of intent,
        and (b) any rewriting here would be hidden state that a
        future critic loop would have to undo. ``call_context`` is
        intentionally NOT folded into the query — it lands in the
        prompt instead, where the LLM can treat it as context rather
        than as a search term.
        """

        return self._retriever.retrieve(
            request.user_intent,
            top_k=request.top_k_examples,
            programme=request.target_programme,
            section_type=request.section_type,
            lessons_learned=request.lessons_learned,
        )

    def _build_prompt(
        self,
        request: GenerationRequest,
        results: list[RetrievalResult],
        *,
        profile: DraftingProfile | None = None,
    ) -> tuple[str, list[CitationRef]]:
        """Delegate to the configured :class:`SectionPromptBuilder`.

        If ``profile`` is provided, programme-specific guidance is used.
        """

        return self._prompt_builder.build(request, results, profile=profile)

    def _generate(self, prompt: str) -> str:
        """Call the configured LLM with the built prompt.

        The model and decoding parameters are the LLM's defaults;
        the workflow does not currently expose them per-request. A
        future request schema can grow ``temperature`` / ``max_tokens``
        fields if user demand justifies it.
        """

        return self._llm.generate(prompt)

    @staticmethod
    def _validate_citations(*, text: str, citations: list[CitationRef]) -> None:
        """Confirm every ``[N]`` in ``text`` indexes a real citation.

        * Hallucinated marker (index out of range or non-positive) →
          :class:`GenerationError` with the offending marker named in
          the message ("an operator needs to know which marker to
          debug").
        * Zero markers in ``text`` → soft warning via :mod:`logging`,
          NOT an error. The deterministic stub on small inputs and
          some real LLMs on terse prompts may legitimately omit
          markers; the citations are still attached to the draft as
          a list and can be rendered as a "see references" footer by
          the downstream renderer.
        """

        emitted = _scan_citation_markers(text)
        if not emitted:
            logger.warning(
                "Generated draft contains no [N] citation markers; "
                "%d citation(s) are still attached to the draft as a "
                "list and can be rendered as a references section.",
                len(citations),
            )
            return

        valid_ids = {c.citation_id for c in citations}
        invalid = [n for n in emitted if n < 1 or n not in valid_ids]
        if invalid:
            # Surface the *first* offending marker prominently — the rest
            # are listed for completeness but the message reads cleanly
            # for the common case of a single hallucination.
            first = invalid[0]
            raise GenerationError(
                f"Generated draft references hallucinated citation [{first}] "
                f"(only [1]..[{len(citations)}] are valid). "
                f"All hallucinated markers: {sorted(set(invalid))}."
            )

    # ------------------------------------------------------------------
    # Analytics emission — optional, never breaks drafting on failure
    # ------------------------------------------------------------------

    def _emit_draft_started(
        self,
        request: GenerationRequest,
        *,
        profile: DraftingProfile | None,
    ) -> None:
        """Emit a :class:`DraftStartedEvent` if an analytics logger is configured.

        Content-safe by construction: the event takes the request's
        operational knobs (section type, top-k, profile name) but
        NEVER the user-intent text or the topic-context body. The
        ``topic_context_present`` flag is the only signal carried.

        Wrapped in ``try/except`` so a transient analytics failure
        (full disk, permission error) cannot break drafting — the
        workflow's primary contract is to return a draft.
        """

        if self._analytics is None:
            return
        try:
            event = DraftStartedEvent(
                event_type=EventType.DRAFT_STARTED,
                section_type=request.section_type.value,
                target_programme=(
                    request.target_programme.value if request.target_programme is not None else None
                ),
                top_k_examples=request.top_k_examples,
                lessons_learned=request.lessons_learned,
                model=self._llm.model,
                drafting_profile=profile.name if profile is not None else None,
                topic_context_present=request.topic_context is not None,
            )
            self._analytics.log(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to emit DraftStartedEvent (analytics logging is best-effort): %s",
                exc,
            )

    def _emit_draft_completed(
        self,
        request: GenerationRequest,
        *,
        citations: list[CitationRef],
        generation_time_ms: int,
        profile: DraftingProfile | None,
        iteration_count: int = 1,
    ) -> None:
        """Emit a :class:`DraftCompletedEvent` if an analytics logger is configured.

        ``source_status_mix`` is built from the citation list as a
        plain string-keyed dict — content-free, only the counts of
        each ``SourceStatus`` label. The draft text and citation
        snippets are NEVER passed to the event.

        ``iteration_count`` is the 1-indexed iteration number that
        produced this draft. Defaults to 1 for the Sprint 1 single-
        pass workflow; the Sprint 3 critic loop passes 2, 3, ... so
        the analytics log carries one event per iteration with the
        correct count. Privacy contract: only the integer is logged
        — never the critique text.

        Wrapped in ``try/except`` so an analytics failure here cannot
        break the workflow returning its draft to the caller.
        """

        if self._analytics is None:
            return
        try:
            mix = Counter(c.source_status.value for c in citations)
            event = DraftCompletedEvent(
                event_type=EventType.DRAFT_COMPLETED,
                section_type=request.section_type.value,
                generation_time_ms=int(generation_time_ms),
                citation_count=len(citations),
                source_status_mix=dict(mix),
                model=self._llm.model,
                drafting_profile=profile.name if profile is not None else None,
                iteration_count=iteration_count,
            )
            self._analytics.log(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to emit DraftCompletedEvent (analytics logging is best-effort): %s",
                exc,
            )
