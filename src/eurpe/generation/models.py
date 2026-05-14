"""Generation-layer Pydantic models.

Three records flow through the section-generation workflow:

* :class:`CitationRef` — a single inline citation in the generated draft.
  Carries enough provenance for the renderer to print a source-status-
  labelled reference (e.g., ``[1] FUNDED — HORIZON-CL5-2024-D3-02,
  p. 12, §Methodology``). ``frozen=True`` so a citation list can be
  hashed / set-deduplicated later if needed.
* :class:`GenerationRequest` — the user-supplied inputs to one
  drafting call. Only carries knobs that the *workflow* needs: the
  source-status policy (threshold, ESR inclusion) is configured on the
  retriever instance the workflow is given, NOT here. The only
  per-call retrieval knob is ``lessons_learned`` because the retriever
  exposes a per-call override for it (the others are constructor-time).
* :class:`GenerationDraft` — the workflow's output. Echoes the request
  for traceability, carries the prompt that was sent to the LLM (for
  audit), and lists every citation referenced in the draft text.

Why ``include_esr`` is not on :class:`GenerationRequest`
--------------------------------------------------------
Earlier drafts of the schema put ``include_esr`` on the request to
mirror the retriever's option. The retriever exposes it on the
*policy* (constructor) rather than per-call, so the only honest
implementation would have been to rebuild the retriever for every
request — which forces the workflow to take a "policy factory" rather
than a ``SourceStatusAwareRetriever`` instance, doubling the surface
area for callers and tests. The current shape mirrors how
``eurpe index query`` ships: the CLI builds the policy with
``--no-esr`` and hands a configured retriever to the workflow.
``lessons_learned`` is fine to keep on the request because the
retriever's ``retrieve()`` method does take a per-call override.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from eurpe.schema import Programme, SectionType, SourceStatus


class CitationRef(BaseModel):
    """A single inline citation in the generated draft.

    The :attr:`citation_id` is the 1-indexed integer that appears in
    the draft text as ``[N]``. The remaining fields carry enough
    information for a downstream renderer (Issue #7) to print a
    fully-labelled reference line. ``snippet`` is a short excerpt of
    the source chunk so the user can sanity-check the citation
    in-place without opening the full PDF.

    ``frozen=True`` so a list of citations can be hashed or
    set-deduplicated. ``extra="forbid"`` keeps typos in field names
    loud, matching the convention used across :mod:`eurpe.schema`,
    :mod:`eurpe.ingestion`, and :mod:`eurpe.retrieval`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: int = Field(
        ge=1,
        description="1-indexed integer that appears in the draft as ``[N]``.",
    )
    source_status: SourceStatus = Field(
        description=(
            "Source-status label of the cited chunk; surfaced for the renderer "
            "so the printed reference can show e.g. ``FUNDED`` or ``REJECTED``."
        ),
    )
    programme: Programme = Field(
        description="EU programme that issued the call the source proposal targeted.",
    )
    call_id: str = Field(
        min_length=1,
        description="Identifier of the call (e.g., ``HORIZON-CL5-2024-D3-02``).",
    )
    proposal_title: str | None = Field(
        default=None,
        description="Title of the source proposal, when known.",
    )
    section_heading: str | None = Field(
        default=None,
        description="Section heading the snippet was taken from, when known.",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed page number, when the source format exposes pagination.",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Stable id of the source chunk; lets an audit trace back.",
    )
    snippet: str = Field(
        max_length=500,
        description=(
            "Short excerpt of the source chunk text. Capped at 500 chars to "
            "keep the citation list compact — the renderer can fetch the "
            "full chunk from the index via ``chunk_id`` when needed."
        ),
    )


class GenerationRequest(BaseModel):
    """Inputs to one section-generation call.

    Only the knobs that the *workflow* uses live here. Source-status
    policy (threshold, ESR inclusion, rejected-fraction cap, funded-
    first ordering) is set on the :class:`~eurpe.retrieval.RetrievalPolicy`
    of the retriever instance handed to the workflow — see this
    module's docstring for the rationale.

    Defaults are chosen so the most common call ("draft a methodology
    section using up to 5 examples") needs only ``section_type`` and
    ``user_intent``.
    """

    model_config = ConfigDict(extra="forbid")

    section_type: SectionType = Field(
        description=(
            "Which section to draft. Used to (a) select section-specific "
            "guidance text injected into the prompt and (b) hard-filter the "
            "retrieved evidence to chunks whose own ``section_type`` matches."
        ),
    )
    user_intent: str = Field(
        min_length=1,
        description=(
            "What the user wants this section to communicate. The single "
            "most important freeform field — drives the retrieval query "
            "and is quoted verbatim in the prompt."
        ),
    )
    call_context: str = Field(
        default="",
        description=(
            "Optional pasted call/topic text. Provides additional context "
            "for the LLM beyond the user's intent. May be empty."
        ),
    )
    target_programme: Programme | None = Field(
        default=None,
        description=(
            "Optional programme filter for retrieval (e.g., HORIZON_EUROPE). "
            "When set, only chunks from proposals targeting this programme "
            "are retrieved. ``None`` (default) retrieves across all "
            "programmes."
        ),
    )
    top_k_examples: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of examples to retrieve. Capped at 20 because the "
            "prompt grows linearly with this value and most local LLMs "
            "have modest context windows."
        ),
    )
    lessons_learned: bool = Field(
        default=False,
        description=(
            "Forwarded to the retriever's per-call lessons-learned "
            "override. When True, rejected examples are surfaced more "
            "aggressively as cautionary evidence (the retriever relaxes "
            "their threshold and skips the rejected-fraction cap). The "
            "renderer should frame those citations as 'what to avoid'."
        ),
    )


class GenerationDraft(BaseModel):
    """Output of one section-generation call.

    The whole record is what callers serialise to disk for downstream
    review. ``prompt_used`` is included so an auditor can replay the
    exact prompt that produced ``text``; this is the single most useful
    field when a draft surprises the user.

    The two helper methods give callers a one-line answer to common
    questions ("how many sources?", "is anything unlabelled?") without
    forcing them to re-implement the logic.
    """

    model_config = ConfigDict(extra="forbid")

    section_type: SectionType = Field(description="Section that was drafted.")
    text: str = Field(
        min_length=1,
        description=(
            "The generated draft text, in markdown. May contain inline "
            "``[N]`` markers that index into :attr:`citations`."
        ),
    )
    citations: list[CitationRef] = Field(
        description=(
            "1-to-1 with the ``[N]`` markers the LLM was asked to use. "
            "Citations may have unused entries (the model didn't reference "
            "every example) but never extra entries — every ``[N]`` in "
            "``text`` MUST correspond to a citation in this list."
        ),
    )
    prompt_used: str = Field(
        min_length=1,
        description="Full prompt sent to the LLM. Useful for debugging / audit.",
    )
    model: str = Field(
        min_length=1,
        description=(
            "Identifier of the LLM that produced ``text`` (e.g., "
            "``llama3.1:8b`` or ``deterministic-stub-v1``)."
        ),
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the draft was assembled.",
    )
    request: GenerationRequest = Field(
        description=(
            "Echo of the request that produced this draft, for "
            "traceability. Lets a stored draft be re-run end-to-end."
        ),
    )

    def has_unlabeled_citations(self) -> bool:
        """True iff any citation lacks a ``source_status``.

        Defensive: Pydantic already requires the field, so this should
        always return False in practice. Kept so callers (especially the
        renderer in Issue #7) have a single named check rather than
        re-implementing the loop.
        """

        return any(c.source_status is None for c in self.citations)

    def citation_count(self) -> int:
        """Number of citations attached to this draft."""

        return len(self.citations)
