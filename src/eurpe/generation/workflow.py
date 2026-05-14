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
"""

from __future__ import annotations

import logging
import re

from eurpe.generation.errors import GenerationError
from eurpe.generation.llm import LLMClient
from eurpe.generation.models import CitationRef, GenerationDraft, GenerationRequest
from eurpe.generation.prompt import SectionPromptBuilder
from eurpe.retrieval import RetrievalResult, SourceStatusAwareRetriever

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
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        # The default prompt builder is stateless and reusable. Allowing
        # the caller to inject one keeps the door open for a future
        # programme- or tenant-specific builder without forcing every
        # call site to construct one.
        self._prompt_builder = prompt_builder or SectionPromptBuilder()

    @property
    def llm(self) -> LLMClient:
        """Read-only access to the configured LLM — useful for tests / logs."""

        return self._llm

    @property
    def retriever(self) -> SourceStatusAwareRetriever:
        """Read-only access to the configured retriever — useful for tests / logs."""

        return self._retriever

    def run(self, request: GenerationRequest) -> GenerationDraft:
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

        Errors:

        * :class:`GenerationError` (base) — anything that goes wrong
          inside the workflow that is not a recoverable LLM
          connection problem.
        * :class:`~eurpe.generation.LLMUnavailableError` — propagated
          from the LLM client when the daemon is unreachable.
        """

        results = self._retrieve(request)
        prompt, citations = self._build_prompt(request, results)
        text = self._generate(prompt)
        self._validate_citations(text=text, citations=citations)
        return GenerationDraft(
            section_type=request.section_type,
            text=text,
            citations=citations,
            prompt_used=prompt,
            model=self._llm.model,
            request=request,
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
    ) -> tuple[str, list[CitationRef]]:
        """Delegate to the configured :class:`SectionPromptBuilder`."""

        return self._prompt_builder.build(request, results)

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
