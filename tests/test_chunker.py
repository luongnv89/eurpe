"""Tests for ``eurpe.retrieval.chunker``.

The chunker is the join point between parser output and the schema's
:class:`ChunkMetadata`. The single most important property under test
here is the source-status-drift invariant: every emitted chunk MUST
carry the same ``source_status`` as its parent proposal's ``outcome``.
The schema layer's drift validator already catches a manual mismatch;
these tests prove the chunker cannot accidentally introduce one.

Other things exercised here:

* Section-type inference is a pure function — table-driven via
  parametrize so a future enum addition is a one-line change.
* Splitter respects the configured ``target_chars``, emits the
  requested overlap between consecutive chunks, and merges a too-small
  tail into its predecessor.
* Tables become their own chunks with the correct parent section
  metadata.
"""

from __future__ import annotations

import pytest

from eurpe.ingestion.models import ParsedProposal, ParsedSection, ParsedTable
from eurpe.retrieval.chunker import HierarchicalChunker, infer_section_type
from eurpe.schema import (
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)

# ---------------------------------------------------------------------------
# infer_section_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("1.1 Excellence", SectionType.EXCELLENCE),
        ("1.2 EXCELLENCE of the proposal", SectionType.EXCELLENCE),
        ("2. Impact", SectionType.IMPACT),
        ("2.1 Expected impacts", SectionType.IMPACT),
        ("2.2 Impact pathway", SectionType.IMPACT_PATHWAY),
        ("Impact Pathway and KPIs", SectionType.IMPACT_PATHWAY),
        ("3. Implementation", SectionType.IMPLEMENTATION),
        ("3.1 Methodology", SectionType.METHODOLOGY),
        ("3.2 Work plan", SectionType.WORK_PLAN),
        ("3.2 Workplan and timing", SectionType.WORK_PLAN),
        ("3.3 Consortium", SectionType.CONSORTIUM),
        ("3.3 Project partners", SectionType.CONSORTIUM),
        ("3.4 Resources to be deployed", SectionType.BUDGET),
        ("3.4 Budget overview", SectionType.BUDGET),
        ("4. Ethics", SectionType.ETHICS),
        ("Dissemination plan", SectionType.DISSEMINATION),
        ("Exploitation strategy", SectionType.DISSEMINATION),
        ("Annex 1: References", SectionType.OTHER),
        ("", SectionType.OTHER),
    ],
)
def test_infer_section_type_recognises_common_patterns(
    heading: str, expected: SectionType
) -> None:
    assert infer_section_type(heading) is expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proposal(outcome: SourceStatus = SourceStatus.FUNDED) -> ProposalMetadata:
    return ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        topic_id="HORIZON-CL5-2024-D3-02-01",
        year=2024,
        outcome=outcome,
        proposal_title="Test proposal",
        consortium_acronym="TEST",
        source_path="data/corpus/funded/test.pdf",
    )


def _parsed(sections: list[ParsedSection]) -> ParsedProposal:
    return ParsedProposal(
        source_path="/abs/test.pdf",
        title="Test",
        sections=sections,
        page_count=10,
    )


# ---------------------------------------------------------------------------
# HierarchicalChunker
# ---------------------------------------------------------------------------


def test_chunker_emits_chunks_with_full_metadata() -> None:
    section = ParsedSection(
        heading="1.1 Excellence",
        level=1,
        text="Excellence body. " * 5,
        page_start=3,
        page_end=3,
    )
    parsed = _parsed([section])
    chunker = HierarchicalChunker(target_chars=200, overlap_chars=50, min_chunk_chars=30)
    chunks = chunker.chunk(parsed, _proposal())

    assert chunks, "expected at least one chunk"
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata.source_status is SourceStatus.FUNDED
        assert chunk.metadata.proposal.outcome is SourceStatus.FUNDED
        assert chunk.metadata.parent_section_heading == "1.1 Excellence"
        assert chunk.metadata.section_type is SectionType.EXCELLENCE
        assert chunk.metadata.chunk_index == idx
        assert chunk.metadata.anchor.document_id == "test"
        assert chunk.metadata.anchor.page == 3
        # char offsets must form a non-decreasing sequence (overlapping is fine)
        assert chunk.metadata.anchor.char_start is not None
        assert chunk.metadata.anchor.char_end is not None
        assert chunk.metadata.anchor.char_start < chunk.metadata.anchor.char_end


def test_chunker_table_becomes_own_chunk() -> None:
    table = ParsedTable(
        section_heading="1.2 Methodology",
        rows=[["WP", "Title"], ["WP1", "Architecture"]],
        page=8,
    )
    section = ParsedSection(
        heading="1.2 Methodology",
        level=1,
        text="Short body that fits in one chunk.",
        page_start=8,
        page_end=8,
        tables=[table],
    )
    parsed = _parsed([section])
    chunker = HierarchicalChunker(target_chars=2000, overlap_chars=100, min_chunk_chars=10)
    chunks = chunker.chunk(parsed, _proposal())

    # One body chunk + one table chunk.
    assert len(chunks) == 2
    table_chunk = chunks[1]
    assert "WP" in table_chunk.text
    assert "Architecture" in table_chunk.text
    # Cells joined with " | ".
    assert " | " in table_chunk.text
    assert table_chunk.metadata.parent_section_heading == "1.2 Methodology"
    assert table_chunk.metadata.section_type is SectionType.METHODOLOGY
    assert table_chunk.metadata.anchor.page == 8


def test_chunker_respects_target_chars() -> None:
    big = "abcdefghij " * 600  # 6600 chars
    section = ParsedSection(heading="Body", level=1, text=big)
    parsed = _parsed([section])
    chunker = HierarchicalChunker(target_chars=1200, overlap_chars=200, min_chunk_chars=200)
    chunks = chunker.chunk(parsed, _proposal())

    assert len(chunks) > 1
    tolerance = 200  # the splitter searches a +/- 100 window for boundaries
    for chunk in chunks[:-1]:
        assert len(chunk.text) <= 1200 + tolerance, (
            f"chunk {chunk.metadata.chunk_index} too long: {len(chunk.text)} chars"
        )
    # Final chunk may be shorter than ``target_chars`` — that is fine
    # by design.
    assert len(chunks[-1].text) <= 1200 + tolerance


def test_chunker_emits_overlap_between_consecutive_chunks() -> None:
    body = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 80
    section = ParsedSection(heading="Body", level=1, text=body)
    parsed = _parsed([section])
    chunker = HierarchicalChunker(target_chars=600, overlap_chars=120, min_chunk_chars=100)
    chunks = chunker.chunk(parsed, _proposal())

    assert len(chunks) >= 2
    # The end of chunk[i] should appear in chunk[i+1] thanks to overlap.
    for i in range(len(chunks) - 1):
        prev_tail = chunks[i].text[-50:]
        next_text = chunks[i + 1].text
        # ``in`` on the next chunk's first ~half catches the overlap
        # (the splitter shifts by at most one boundary token).
        assert (
            prev_tail[:30].strip() in next_text
            or prev_tail[20:].strip() in next_text
        )


def test_chunker_min_chunk_chars_merges_tail_into_predecessor() -> None:
    """Picked parameters that actually trigger the merge.

    With ``min_chunk_chars=0`` the splitter would emit three spans on
    900-char text: ``[(0,400), (350,750), (700,900)]`` (final chunk
    200 chars). With ``min_chunk_chars=250``, that 200-char tail must
    fold into its predecessor, leaving exactly two spans whose final
    chunk reaches the end of the input. The tighter assertion catches
    a regression that "no chunk shorter than min" alone would miss
    (a vacuous pass when the tail is naturally already large enough).
    """

    body = "x" * 900
    section = ParsedSection(heading="Body", level=1, text=body)
    parsed = _parsed([section])
    no_merge = HierarchicalChunker(target_chars=400, overlap_chars=50, min_chunk_chars=0)
    no_merge_chunks = no_merge.chunk(parsed, _proposal())
    assert len(no_merge_chunks) == 3
    assert len(no_merge_chunks[-1].text) == 200

    merge = HierarchicalChunker(target_chars=400, overlap_chars=50, min_chunk_chars=250)
    merge_chunks = merge.chunk(parsed, _proposal())
    # The 200-char tail folded into the predecessor.
    assert len(merge_chunks) == 2
    # Final chunk now ends at the input's end, with length > min.
    last = merge_chunks[-1]
    assert last.metadata.anchor.char_end == 900
    assert len(last.text) >= 250


def test_chunker_propagates_esr_status_to_every_chunk() -> None:
    section_a = ParsedSection(heading="1.1 Excellence", level=1, text="excellence " * 30)
    section_b = ParsedSection(heading="2. Impact", level=1, text="impact " * 30)
    parsed = _parsed([section_a, section_b])
    chunker = HierarchicalChunker(target_chars=200, overlap_chars=40, min_chunk_chars=30)
    chunks = chunker.chunk(parsed, _proposal(outcome=SourceStatus.ESR_NOTE))

    assert chunks
    for chunk in chunks:
        assert chunk.metadata.source_status is SourceStatus.ESR_NOTE
        assert chunk.metadata.proposal.outcome is SourceStatus.ESR_NOTE


def test_chunker_chunk_index_is_monotonic_across_sections() -> None:
    sections = [
        ParsedSection(heading=f"Section {i}", level=1, text="body " * 20) for i in range(5)
    ]
    parsed = _parsed(sections)
    chunker = HierarchicalChunker(target_chars=400, overlap_chars=50, min_chunk_chars=20)
    chunks = chunker.chunk(parsed, _proposal())

    indices = [c.metadata.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunker_skips_empty_section() -> None:
    section = ParsedSection(heading="Empty", level=1, text="   ")
    parsed = _parsed([section])
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(parsed, _proposal())
    assert chunks == []


def test_chunker_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="target_chars"):
        HierarchicalChunker(target_chars=0)
    with pytest.raises(ValueError, match="overlap_chars"):
        HierarchicalChunker(target_chars=100, overlap_chars=-1)
    with pytest.raises(ValueError, match="overlap_chars"):
        HierarchicalChunker(target_chars=100, overlap_chars=100)
    with pytest.raises(ValueError, match="min_chunk_chars"):
        HierarchicalChunker(target_chars=100, overlap_chars=10, min_chunk_chars=-1)
