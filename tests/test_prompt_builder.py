"""Tests for ``eurpe.generation.prompt``.

The prompt is a contract between the workflow and the LLM (and in
particular between the workflow and the deterministic stub, which
parses ``[N]`` markers out of the prompt with a regex). The tests
below pin:

* The section-guidance text appears in the prompt for every
  :class:`SectionType` (so a new section type without guidance is a
  loud failure).
* The user intent and call context are present verbatim.
* Funded and rejected examples are visibly labelled (``**FUNDED**``
  / ``**REJECTED**``) so the LLM cannot confuse the two.
* The ``[N]`` markers in the evidence block are 1-indexed and
  rendered in retrieval order.
* The "Do not invent" safety phrase is present.
* Empty retrieval results still yield a well-formed prompt.
"""

from __future__ import annotations

import re

from eurpe.generation.models import GenerationRequest
from eurpe.generation.prompt import (
    SECTION_GUIDANCE,
    SectionPromptBuilder,
)
from eurpe.retrieval import Chunk, RetrievalResult
from eurpe.retrieval.retriever import POLICY_REASON_FUNDED, POLICY_REASON_REJECTED
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    status: SourceStatus,
    programme: Programme = Programme.HORIZON_EUROPE,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    section_type: SectionType = SectionType.METHODOLOGY,
    document_id: str = "doc",
    chunk_index: int = 0,
    text: str = "stub chunk text used in prompt-builder tests",
    section_heading: str | None = "1.2 Methodology",
    page: int | None = 8,
) -> Chunk:
    proposal = ProposalMetadata(
        programme=programme,
        call_id=call_id,
        year=2024,
        outcome=status,
        proposal_title="Sample Proposal",
        source_path=f"data/{document_id}.pdf",
    )
    anchor = CitationAnchor(
        document_id=document_id,
        section_heading=section_heading,
        page=page,
    )
    metadata = ChunkMetadata(
        proposal=proposal,
        section_type=section_type,
        parent_section_heading=section_heading,
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=status,
    )
    return Chunk(text=text, metadata=metadata)


def _make_result(chunk: Chunk, score: float, rank: int, reason: str) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score, rank=rank, policy_reason=reason)


def _request(section_type: SectionType = SectionType.METHODOLOGY) -> GenerationRequest:
    return GenerationRequest(
        section_type=section_type,
        user_intent="Describe our deep learning approach",
        call_context="Funding call about cyber-physical resilience.",
        target_programme=Programme.HORIZON_EUROPE,
    )


# ---------------------------------------------------------------------------
# Section guidance
# ---------------------------------------------------------------------------


def test_prompt_includes_section_guidance_for_every_section_type() -> None:
    """SECTION_GUIDANCE has an entry for every SectionType (no missing key).

    A new SectionType added without a guidance entry should fail this
    test loudly. The build path uses ``.get(..., SECTION_GUIDANCE[OTHER])``
    so it would silently fall back; this test pins the explicit
    coverage requirement separately.
    """

    for st in SectionType:
        assert st in SECTION_GUIDANCE, (
            f"SectionType.{st.name} missing from SECTION_GUIDANCE — "
            "add an entry in eurpe.generation.prompt."
        )


def test_prompt_includes_section_guidance_text() -> None:
    builder = SectionPromptBuilder()
    prompt, _ = builder.build(_request(SectionType.METHODOLOGY), [])
    # The methodology guidance text must show up in the prompt.
    assert SECTION_GUIDANCE[SectionType.METHODOLOGY] in prompt


def test_prompt_includes_user_intent() -> None:
    builder = SectionPromptBuilder()
    prompt, _ = builder.build(_request(), [])
    assert "Describe our deep learning approach" in prompt


def test_prompt_includes_call_context_when_provided() -> None:
    builder = SectionPromptBuilder()
    prompt, _ = builder.build(_request(), [])
    assert "cyber-physical resilience" in prompt


def test_prompt_marks_no_context_when_call_context_empty() -> None:
    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        call_context="",
    )
    prompt, _ = builder.build(req, [])
    assert "(none provided)" in prompt


def test_prompt_includes_no_inventions_instruction() -> None:
    """The 'Do not invent' phrase is a key safety behaviour and must appear."""

    builder = SectionPromptBuilder()
    prompt, _ = builder.build(_request(), [])
    assert "Do not invent" in prompt


def test_prompt_includes_section_type_marker_for_stub_parsing() -> None:
    """The ``**Section type:** <Title>`` line is what the deterministic stub parses."""

    builder = SectionPromptBuilder()
    prompt, _ = builder.build(_request(SectionType.IMPACT_PATHWAY), [])
    assert "**Section type:** Impact Pathway" in prompt


# ---------------------------------------------------------------------------
# Source-status labelling
# ---------------------------------------------------------------------------


def test_prompt_marks_funded_and_rejected_distinctly() -> None:
    """Both **FUNDED** and **REJECTED** labels must appear when both are present.

    AC2 keystone for the prompt: the LLM must be able to tell the two
    apart. Any change to the label format that breaks this assertion
    is a contract change and should be rejected unless the workflow
    test for citation parsing is updated in lockstep.
    """

    builder = SectionPromptBuilder()
    funded = _make_chunk(status=SourceStatus.FUNDED, document_id="f1")
    rejected = _make_chunk(
        status=SourceStatus.REJECTED,
        programme=Programme.HORIZON_2020,
        call_id="H2020-X",
        document_id="r1",
    )
    results = [
        _make_result(funded, 0.9, 1, POLICY_REASON_FUNDED),
        _make_result(rejected, 0.6, 2, POLICY_REASON_REJECTED),
    ]
    prompt, _ = builder.build(_request(), results)
    assert "**FUNDED**" in prompt
    assert "**REJECTED**" in prompt


def test_prompt_handles_empty_results() -> None:
    """No retrieved evidence → still well-formed prompt with explicit marker."""

    builder = SectionPromptBuilder()
    prompt, citations = builder.build(_request(), [])
    assert "(no examples retrieved)" in prompt
    assert citations == []


def test_prompt_includes_esr_and_unknown_labels_when_present() -> None:
    """The other two source statuses are also visibly labelled."""

    builder = SectionPromptBuilder()
    esr = _make_chunk(
        status=SourceStatus.ESR_NOTE,
        call_id="ESR-001",
        document_id="e1",
    )
    unknown = _make_chunk(
        status=SourceStatus.UNKNOWN,
        programme=Programme.OTHER,
        call_id="UNKNOWN-X",
        document_id="u1",
    )
    results = [
        _make_result(esr, 0.7, 1, "esr_advisory"),
        _make_result(unknown, 0.6, 2, "unknown_low_confidence"),
    ]
    prompt, _ = builder.build(_request(), results)
    assert "**ESR NOTE**" in prompt
    assert "**UNKNOWN**" in prompt


# ---------------------------------------------------------------------------
# Citation construction
# ---------------------------------------------------------------------------


def test_citations_are_one_indexed_in_retrieval_order() -> None:
    builder = SectionPromptBuilder()
    chunks = [
        _make_chunk(status=SourceStatus.FUNDED, document_id="a", chunk_index=0),
        _make_chunk(status=SourceStatus.FUNDED, document_id="b", chunk_index=1),
        _make_chunk(status=SourceStatus.REJECTED, document_id="c", chunk_index=2),
    ]
    results = [
        _make_result(chunks[0], 0.9, 1, POLICY_REASON_FUNDED),
        _make_result(chunks[1], 0.8, 2, POLICY_REASON_FUNDED),
        _make_result(chunks[2], 0.7, 3, POLICY_REASON_REJECTED),
    ]
    _, citations = builder.build(_request(), results)
    assert [c.citation_id for c in citations] == [1, 2, 3]
    # Retrieval order is preserved in the citation list.
    assert citations[0].chunk_id == chunks[0].chunk_id
    assert citations[2].chunk_id == chunks[2].chunk_id


def test_citation_snippet_truncated_to_max_length() -> None:
    """Long chunk text is truncated to keep the prompt and citation list compact."""

    builder = SectionPromptBuilder()
    long_text = "lorem ipsum " * 200  # ~2400 chars
    chunk = _make_chunk(status=SourceStatus.FUNDED, text=long_text)
    results = [_make_result(chunk, 0.9, 1, POLICY_REASON_FUNDED)]
    _, citations = builder.build(_request(), results)
    # Snippet capped at the documented 300 chars and ends with an ellipsis.
    assert len(citations[0].snippet) <= 300
    assert citations[0].snippet.endswith("…")


def test_citation_carries_source_status_label() -> None:
    """Every citation built from a retrieval result carries its source_status."""

    builder = SectionPromptBuilder()
    chunks = [
        _make_chunk(status=SourceStatus.FUNDED, document_id="a"),
        _make_chunk(
            status=SourceStatus.REJECTED,
            programme=Programme.HORIZON_2020,
            call_id="H2020-X",
            document_id="b",
        ),
    ]
    results = [
        _make_result(chunks[0], 0.9, 1, POLICY_REASON_FUNDED),
        _make_result(chunks[1], 0.6, 2, POLICY_REASON_REJECTED),
    ]
    _, citations = builder.build(_request(), results)
    statuses = {c.source_status for c in citations}
    assert SourceStatus.FUNDED in statuses
    assert SourceStatus.REJECTED in statuses


def test_evidence_block_uses_pinned_marker_format() -> None:
    """Each evidence header line matches the regex the deterministic stub parses."""

    builder = SectionPromptBuilder()
    funded = _make_chunk(status=SourceStatus.FUNDED, document_id="f1")
    rejected = _make_chunk(
        status=SourceStatus.REJECTED,
        programme=Programme.HORIZON_2020,
        call_id="H2020-X",
        document_id="r1",
    )
    results = [
        _make_result(funded, 0.9, 1, POLICY_REASON_FUNDED),
        _make_result(rejected, 0.6, 2, POLICY_REASON_REJECTED),
    ]
    prompt, _ = builder.build(_request(), results)
    # Same pattern that DeterministicLLMClient uses to extract markers.
    pattern = re.compile(r"^\[(\d{1,2})\]\s+\*\*", re.MULTILINE)
    found = sorted({int(m.group(1)) for m in pattern.finditer(prompt)})
    assert found == [1, 2]


# ---------------------------------------------------------------------------
# Issue #9 — structured topic_context rendering in the prompt.
# ---------------------------------------------------------------------------


def _topic_context(
    *,
    section_guidance_for: SectionType | None = None,
) -> "TopicContext":
    """Build a fully-populated :class:`TopicContext` for prompt assertions.

    ``section_guidance_for`` lets a test request a guidance entry on a
    specific section without re-pasting the full struct.
    """

    from eurpe.intake import TopicContext, TopicSource

    section_guidance: dict[SectionType, str] = {}
    if section_guidance_for is not None:
        section_guidance[section_guidance_for] = (
            "Validate the approach on at least two real-world pilots."
        )

    return TopicContext(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL3-2024-CS-01",
        topic_id="952672",
        topic_title="Resilient digital infrastructure for critical sectors",
        expected_outcomes=[
            "Reduced mean-time-to-recover by 30%.",
            "New open standards for protocols.",
        ],
        scope="Proposals should address resilience of digital infrastructure.",
        destination="Cluster 3 — Civil Security for Society",
        section_guidance=section_guidance,
        raw_text="(unused in tests)",
        source=TopicSource.PASTED_TEXT,
    )


def test_prompt_includes_structured_topic_context_when_provided() -> None:
    """All six label lines + outcomes + scope appear when topic_context is set.

    Pins the rendering contract on which the React UI / FastAPI route
    will eventually rely.
    """

    from eurpe.intake import TopicContext  # noqa: F401 — type used in helper

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        topic_context=_topic_context(),
    )
    prompt, _ = builder.build(req, [])

    # Label lines — exact-match each.
    assert "**Programme:** Horizon Europe" in prompt
    assert "**Call ID:** HORIZON-CL3-2024-CS-01" in prompt
    assert "**Topic ID:** 952672" in prompt
    assert "**Topic title:** Resilient digital infrastructure for critical sectors" in prompt
    assert "**Destination:** Cluster 3 — Civil Security for Society" in prompt
    # Outcomes bullets appear verbatim.
    assert "* Reduced mean-time-to-recover by 30%." in prompt
    assert "* New open standards for protocols." in prompt
    # Scope appears under its label.
    assert "**Scope:** Proposals should address resilience of digital infrastructure." in prompt


def test_prompt_includes_expected_outcomes_instruction_when_topic_supplied() -> None:
    """AC #3 keystone: the exact instruction sentence appears in the prompt.

    The wording is pinned because the React UI demo / acceptance test
    asserts on it verbatim. Changing the wording is a contract change.
    """

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        topic_context=_topic_context(),
    )
    prompt, _ = builder.build(req, [])

    assert (
        "Reference the supplied Expected Outcomes from the call / topic context "
        "where appropriate, framing the draft so it explicitly addresses the "
        "topic's intended outcomes."
    ) in prompt


def test_prompt_omits_expected_outcomes_instruction_when_no_topic_outcomes() -> None:
    """No topic_context → no Expected-Outcomes reference instruction."""

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        call_context="Some pasted call text",
    )
    prompt, _ = builder.build(req, [])

    assert "Reference the supplied Expected Outcomes" not in prompt


def test_prompt_omits_expected_outcomes_instruction_when_topic_outcomes_empty() -> None:
    """A topic_context with no outcomes also suppresses the instruction."""

    from eurpe.intake import TopicContext, TopicSource

    ctx = TopicContext(
        topic_id="952672",
        source=TopicSource.PASTED_TEXT,
    )

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        topic_context=ctx,
    )
    prompt, _ = builder.build(req, [])

    assert "Reference the supplied Expected Outcomes" not in prompt


def test_prompt_renders_topic_section_guidance_under_section_guidance_block() -> None:
    """Topic-supplied section guidance appears under its labelled prefix.

    The ``**Topic requirements for this section:**`` prefix is part of
    the prompt contract — pinned here so a future refactor of
    :meth:`build` cannot silently drop it.
    """

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        topic_context=_topic_context(section_guidance_for=SectionType.METHODOLOGY),
    )
    prompt, _ = builder.build(req, [])

    assert (
        "**Topic requirements for this section:** Validate the approach on at "
        "least two real-world pilots."
    ) in prompt
    # The guidance must sit BEFORE the "## Call / topic context" block —
    # i.e., inside the section-guidance block.
    guidance_idx = prompt.index("**Topic requirements for this section:**")
    ctc_idx = prompt.index("## Call / topic context")
    assert guidance_idx < ctc_idx


def test_prompt_does_not_render_topic_section_guidance_for_other_sections() -> None:
    """Guidance keyed on Methodology does NOT leak into an Impact prompt."""

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.IMPACT,
        user_intent="x",
        topic_context=_topic_context(section_guidance_for=SectionType.METHODOLOGY),
    )
    prompt, _ = builder.build(req, [])

    assert "**Topic requirements for this section:**" not in prompt


def test_prompt_renders_call_context_alongside_topic_context_under_freetext_notes() -> None:
    """``call_context`` + ``topic_context`` → both rendered, free-text labelled.

    Pins the contract that the legacy ``--context`` flag still works
    even when the new structured intake path is in use.
    """

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        call_context="Pasted free-text supplement.",
        topic_context=_topic_context(),
    )
    prompt, _ = builder.build(req, [])

    # Structured context is present.
    assert "**Topic ID:** 952672" in prompt
    # Free-text notes appear under the labelled sub-block.
    assert "**Free-text notes:**" in prompt
    assert "Pasted free-text supplement." in prompt


def test_prompt_falls_back_to_call_context_when_no_topic_context() -> None:
    """Without ``topic_context`` the legacy free-text rendering is preserved.

    Same behaviour as the pre-issue-#9 build: ``call_context`` is
    rendered verbatim under ``## Call / topic context`` and there are
    no labelled sub-blocks.
    """

    builder = SectionPromptBuilder()
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="x",
        call_context="Funding call about cyber-physical resilience.",
    )
    prompt, _ = builder.build(req, [])

    # Legacy text appears verbatim.
    assert "Funding call about cyber-physical resilience." in prompt
    # No structured labels.
    assert "**Topic ID:**" not in prompt
    assert "**Free-text notes:**" not in prompt
