"""Tests for ``eurpe.retrieval.index``.

The acceptance criteria for issue #4 land here:

1. Local embedding/index creation works without outbound network access.
2. Index can be rebuilt from fixtures and queried in a deterministic test.
3. Chunks retain parent document, section heading, programme, call, and
   source-status metadata across the round trip through Chroma.

Every test uses :class:`DeterministicHashEmbedder` so the suite stays
fast and reproducible. The ``no_network`` fixture is layered on top of
the headline indexing test to *prove* the offline contract — if Chroma
ever regains a phone-home behaviour, that test goes red immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eurpe.retrieval import (
    ChromaIndex,
    Chunk,
    DeterministicHashEmbedder,
)
from eurpe.retrieval.index import _chroma_to_metadata, _metadata_to_chroma
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)
from tests._chunk_helpers import build_fixture_chunks, query_text_for

# ---------------------------------------------------------------------------
# Metadata round-trip helpers
# ---------------------------------------------------------------------------


def _full_chunk_metadata() -> ChunkMetadata:
    """A ``ChunkMetadata`` with every optional field populated.

    Used by the round-trip test so we exercise the whole flatten /
    inflate path, not just the required-fields subset.
    """

    proposal = ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        topic_id="HORIZON-CL5-2024-D3-02-01",
        year=2024,
        outcome=SourceStatus.FUNDED,
        proposal_title="Edge AI for Resilient Energy Grids",
        consortium_acronym="EAGER",
        source_path="data/corpus/funded/eager_part_b.pdf",
        language="en",
    )
    anchor = CitationAnchor(
        document_id="eager_part_b",
        section_heading="1.2 Methodology",
        page=8,
        char_start=14250,
        char_end=15800,
    )
    return ChunkMetadata(
        proposal=proposal,
        section_type=SectionType.METHODOLOGY,
        parent_section_heading="1.2 Methodology",
        chunk_index=12,
        anchor=anchor,
        source_status=SourceStatus.FUNDED,
    )


def _minimal_chunk_metadata() -> ChunkMetadata:
    """A ``ChunkMetadata`` with every optional field left ``None``.

    Mirrors the ``unknown_other.yaml`` fixture's shape — exercises the
    "Chroma rejects None, so we omit and re-derive" rule.
    """

    proposal = ProposalMetadata(
        programme=Programme.OTHER,
        call_id="NATIONAL-FR-2022-CYBER",
        year=2022,
        outcome=SourceStatus.UNKNOWN,
        source_path="data/corpus/unsorted/national_cyber_range_part_b.pdf",
    )
    anchor = CitationAnchor(document_id="national_cyber_range_part_b")
    return ChunkMetadata(
        proposal=proposal,
        section_type=SectionType.OTHER,
        parent_section_heading=None,
        chunk_index=7,
        anchor=anchor,
        source_status=SourceStatus.UNKNOWN,
    )


def test_metadata_round_trip_preserves_every_field() -> None:
    original = _full_chunk_metadata()
    flat = _metadata_to_chroma(original)
    reloaded = _chroma_to_metadata(dict(flat))
    assert reloaded.section_type == original.section_type
    assert reloaded.parent_section_heading == original.parent_section_heading
    assert reloaded.chunk_index == original.chunk_index
    assert reloaded.source_status == original.source_status
    assert reloaded.anchor == original.anchor
    # ``ingested_at`` is recomputed at construction in the original so
    # we cannot compare it directly; assert it survives as a datetime.
    assert reloaded.proposal.ingested_at is not None
    # Field-by-field for the proposal half (skipping the timestamp).
    for field in (
        "programme",
        "call_id",
        "topic_id",
        "year",
        "outcome",
        "proposal_title",
        "consortium_acronym",
        "source_path",
        "language",
    ):
        assert getattr(reloaded.proposal, field) == getattr(original.proposal, field), (
            f"field {field} drifted"
        )


def test_metadata_round_trip_preserves_none_values() -> None:
    original = _minimal_chunk_metadata()
    flat = _metadata_to_chroma(original)
    # The flattener MUST omit ``None`` keys (Chroma rejects them).
    assert "anchor.page" not in flat
    assert "anchor.char_start" not in flat
    assert "anchor.section_heading" not in flat
    assert "parent_section_heading" not in flat
    assert "proposal.topic_id" not in flat
    reloaded = _chroma_to_metadata(dict(flat))
    assert reloaded.anchor.page is None
    assert reloaded.anchor.char_start is None
    assert reloaded.anchor.section_heading is None
    assert reloaded.parent_section_heading is None
    assert reloaded.proposal.topic_id is None


def test_metadata_round_trip_through_real_chroma_collection(tmp_path: Path) -> None:
    """End-to-end: write a chunk to Chroma, read it back, fields preserved.

    Catches any drift between ``_metadata_to_chroma`` and Chroma's own
    type checks (e.g., a metadata key whose value is the wrong scalar
    type would be silently dropped).
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="round_trip")
    chunk = Chunk(
        text="round trip text for the methodology chunk",
        metadata=_full_chunk_metadata(),
    )
    index.upsert([chunk])
    results = index.query("round trip text for the methodology chunk", top_k=1)
    assert len(results) == 1
    returned, _score = results[0]
    assert returned.metadata.proposal.programme is Programme.HORIZON_EUROPE
    assert returned.metadata.section_type is SectionType.METHODOLOGY
    assert returned.metadata.source_status is SourceStatus.FUNDED
    assert returned.metadata.anchor.document_id == "eager_part_b"
    assert returned.metadata.anchor.page == 8


# ---------------------------------------------------------------------------
# Build / query tests
# ---------------------------------------------------------------------------


def _build_index_with_fixtures(tmp_path: Path) -> tuple[ChromaIndex, list[Chunk]]:
    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="fixtures")
    chunks = build_fixture_chunks()
    index.upsert(chunks)
    return index, chunks


def test_index_upsert_then_query_is_deterministic(tmp_path: Path) -> None:
    """Same fixtures + same query → identical top-1 across two calls.

    Acceptance criterion: "Index can be rebuilt from fixtures and
    queried in a deterministic test."
    """

    index, _chunks = _build_index_with_fixtures(tmp_path)
    query = query_text_for("funded_horizon_europe.yaml")
    first = index.query(query, top_k=4)
    second = index.query(query, top_k=4)
    assert [c.metadata.anchor.document_id for c, _ in first] == [
        c.metadata.anchor.document_id for c, _ in second
    ]
    # Top-1 should be the funded fixture because its marker token is in
    # the query string.
    assert first[0][0].metadata.source_status is SourceStatus.FUNDED


def test_index_upsert_is_idempotent_on_chunk_id(tmp_path: Path) -> None:
    index, chunks = _build_index_with_fixtures(tmp_path)
    initial = index.count()
    assert initial == len(chunks)
    # Upsert again — count must not change.
    index.upsert(chunks)
    assert index.count() == initial


def test_index_query_with_status_filter_only_returns_matching(tmp_path: Path) -> None:
    index, _chunks = _build_index_with_fixtures(tmp_path)
    results = index.query(
        "query that mentions excellence and impact", top_k=10, where={"source_status": "funded"}
    )
    assert results, "expected at least one funded chunk"
    for chunk, _score in results:
        assert chunk.metadata.source_status is SourceStatus.FUNDED


def test_index_query_top_k_is_respected(tmp_path: Path) -> None:
    index, chunks = _build_index_with_fixtures(tmp_path)
    results = index.query("query terms", top_k=2)
    assert len(results) == min(2, len(chunks))


def test_index_persists_across_reloads(tmp_path: Path) -> None:
    """Closing and reopening the index must preserve the data.

    Confirms ``PersistentClient`` is doing its job and that the
    ``ChromaIndex`` constructor does NOT wipe an existing collection
    when a second instance is created against the same path.
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    first = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="persist")
    chunks = build_fixture_chunks()
    first.upsert(chunks)
    expected_count = first.count()
    del first
    # Re-open against the same path.
    second = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="persist")
    assert second.count() == expected_count
    results = second.query(query_text_for("funded_horizon_europe.yaml"), top_k=1)
    assert results
    assert results[0][0].metadata.source_status is SourceStatus.FUNDED


def test_index_delete_collection_drops_count(tmp_path: Path) -> None:
    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="droppable")
    index.upsert(build_fixture_chunks())
    assert index.count() > 0
    index.delete_collection()
    # Reopening creates a fresh empty collection.
    index2 = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="droppable")
    assert index2.count() == 0


def test_index_query_top_k_zero_raises(tmp_path: Path) -> None:
    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="bad_topk")
    index.upsert(build_fixture_chunks())
    with pytest.raises(ValueError, match="top_k"):
        index.query("anything", top_k=0)


# ---------------------------------------------------------------------------
# Duplicate-detection helpers and incremental-indexing regression
# (issue #11 acceptance criteria a + c)
# ---------------------------------------------------------------------------


def _proposal_with_hash(
    *,
    content_hash: str,
    proposal_title: str | None = None,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    source_path: str = "data/corpus/sample.pdf",
) -> ProposalMetadata:
    """Compact :class:`ProposalMetadata` factory carrying ``content_hash``."""

    return ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id=call_id,
        year=2024,
        outcome=SourceStatus.FUNDED,
        proposal_title=proposal_title,
        source_path=source_path,
        content_hash=content_hash,
    )


def _named_chunk(proposal: ProposalMetadata, *, document_id: str, chunk_index: int) -> Chunk:
    """Synthesise a chunk with a stable document_id and unique text."""

    anchor = CitationAnchor(document_id=document_id)
    meta = ChunkMetadata(
        proposal=proposal,
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=proposal.outcome,
    )
    return Chunk(
        text=f"chunk {chunk_index} for {document_id}: unique vocabulary tokens",
        metadata=meta,
    )


def test_index_round_trip_preserves_content_hash() -> None:
    """``content_hash`` survives the flatten + inflate round-trip."""

    original = _full_chunk_metadata()
    populated = original.model_copy(
        update={"proposal": original.proposal.model_copy(update={"content_hash": "a" * 64})}
    )
    flat = _metadata_to_chroma(populated)
    assert flat["proposal.content_hash"] == "a" * 64
    reloaded = _chroma_to_metadata(dict(flat))
    assert reloaded.proposal.content_hash == "a" * 64


def test_index_round_trip_preserves_none_content_hash() -> None:
    """When ``content_hash`` is None, the Chroma key is omitted on write."""

    original = _minimal_chunk_metadata()
    assert original.proposal.content_hash is None
    flat = _metadata_to_chroma(original)
    assert "proposal.content_hash" not in flat
    reloaded = _chroma_to_metadata(dict(flat))
    assert reloaded.proposal.content_hash is None


def test_index_find_by_content_hash_returns_matching_document_ids(tmp_path: Path) -> None:
    """Upsert two proposals with different hashes; finds the right doc_ids."""

    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="hash_lookup")
    p_a = _proposal_with_hash(content_hash="a" * 64)
    p_b = _proposal_with_hash(content_hash="b" * 64)
    index.upsert(
        [
            _named_chunk(p_a, document_id="doc_a", chunk_index=0),
            _named_chunk(p_a, document_id="doc_a", chunk_index=1),
            _named_chunk(p_b, document_id="doc_b", chunk_index=0),
        ]
    )

    assert index.find_by_content_hash("a" * 64) == ["doc_a"]
    assert index.find_by_content_hash("b" * 64) == ["doc_b"]


def test_index_find_by_content_hash_returns_empty_for_no_match(tmp_path: Path) -> None:
    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="hash_miss")
    index.upsert(
        [
            _named_chunk(
                _proposal_with_hash(content_hash="a" * 64),
                document_id="doc_a",
                chunk_index=0,
            )
        ]
    )
    assert index.find_by_content_hash("0" * 64) == []


def test_index_find_by_title_and_call_skips_when_title_none(tmp_path: Path) -> None:
    """Passing ``None`` for title short-circuits to ``[]`` without a query."""

    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="title_none")
    proposal = _proposal_with_hash(content_hash="a" * 64, proposal_title=None)
    index.upsert([_named_chunk(proposal, document_id="doc_a", chunk_index=0)])

    assert index.find_by_title_and_call(None, "HORIZON-CL5-2024-D3-02") == []


def test_index_delete_by_document_id_removes_only_matching_chunks(tmp_path: Path) -> None:
    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="delete_targeted")
    p_a = _proposal_with_hash(content_hash="a" * 64)
    p_b = _proposal_with_hash(content_hash="b" * 64)
    index.upsert(
        [
            _named_chunk(p_a, document_id="doc_a", chunk_index=0),
            _named_chunk(p_a, document_id="doc_a", chunk_index=1),
            _named_chunk(p_b, document_id="doc_b", chunk_index=0),
        ]
    )
    deleted = index.delete_by_document_id("doc_a")
    assert deleted == 2
    assert index.find_by_document_id("doc_a") == 0
    # The other document's chunks remain intact.
    assert index.find_by_document_id("doc_b") == 1
    assert index.count() == 1


def test_index_adding_new_proposal_preserves_existing_chunks_regression(
    tmp_path: Path,
) -> None:
    """AC(a): a fresh proposal does not delete or replace existing chunks."""

    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="incremental")

    p_a = _proposal_with_hash(content_hash="a" * 64, proposal_title="Alpha")
    a_chunks = [
        _named_chunk(p_a, document_id="doc_a", chunk_index=0),
        _named_chunk(p_a, document_id="doc_a", chunk_index=1),
    ]
    index.upsert(a_chunks)
    a_count = index.count()
    assert a_count == 2

    p_b = _proposal_with_hash(content_hash="b" * 64, proposal_title="Beta")
    b_chunks = [
        _named_chunk(p_b, document_id="doc_b", chunk_index=0),
        _named_chunk(p_b, document_id="doc_b", chunk_index=1),
        _named_chunk(p_b, document_id="doc_b", chunk_index=2),
    ]
    index.upsert(b_chunks)

    # AC(a): A's chunks survived B's upsert.
    assert index.find_by_document_id("doc_a") == 2
    assert index.find_by_document_id("doc_b") == 3
    assert index.count() == a_count + 3
    # And a hash lookup for A still resolves.
    assert index.find_by_content_hash("a" * 64) == ["doc_a"]


def test_index_reindex_with_fewer_chunks_leaves_no_orphans(tmp_path: Path) -> None:
    """Reindex (delete-then-upsert) shrinks ``find_by_document_id`` to the new count.

    Pins AC(c): a corrected document with fewer chunks must not leave
    stale higher-indexed chunks behind. ``ChromaIndex.upsert`` alone
    would replace matching chunk_ids but the trailing chunks from the
    previous version would survive — :meth:`delete_by_document_id` is
    why that does not happen.
    """

    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="reindex")
    initial = _proposal_with_hash(content_hash="a" * 64, proposal_title="Original")
    initial_chunks = [_named_chunk(initial, document_id="doc_x", chunk_index=i) for i in range(5)]
    index.upsert(initial_chunks)
    assert index.find_by_document_id("doc_x") == 5

    # Re-index with a corrected (smaller) version.
    index.delete_by_document_id("doc_x")
    corrected = _proposal_with_hash(content_hash="b" * 64, proposal_title="Original")
    corrected_chunks = [
        _named_chunk(corrected, document_id="doc_x", chunk_index=i) for i in range(3)
    ]
    index.upsert(corrected_chunks)

    assert index.find_by_document_id("doc_x") == 3
    # Stored hash now points at the corrected bytes.
    assert index.find_by_content_hash("b" * 64) == ["doc_x"]
    assert index.find_by_content_hash("a" * 64) == []


def test_reindex_proposal_loses_old_chunks_on_upsert_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the documented non-atomic-reindex contract.

    ``reindex_proposal`` deletes the old chunks before chunking and
    upserting the new ones. If ``index.upsert`` raises (transient Chroma
    failure, embedder error), the old chunks are gone and nothing
    replaces them — the operator sees a 500 and the document_id is empty
    in the index. The PR's known-follow-up bullet documents this; this
    test makes the data-loss window visible to CI so a future atomic
    rewrite can flip the assertion instead of inheriting silent breakage.
    """

    from unittest.mock import patch

    from eurpe.ingestion.models import ParsedProposal, ParsedSection
    from eurpe.retrieval.chunker import HierarchicalChunker
    from eurpe.retrieval.errors import IndexingError
    from eurpe.retrieval.pipeline import reindex_proposal

    embedder = DeterministicHashEmbedder(dimension=32)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="reindex_fail")
    initial = _proposal_with_hash(content_hash="a" * 64, proposal_title="Original")
    initial_chunks = [_named_chunk(initial, document_id="doc_x", chunk_index=i) for i in range(3)]
    index.upsert(initial_chunks)
    assert index.find_by_document_id("doc_x") == 3

    parsed = ParsedProposal(
        source_path="/abs/doc_x.pdf",
        title="Original",
        sections=[
            ParsedSection(
                heading="1.1 Excellence",
                level=1,
                text="Replacement body content. " * 10,
                page_start=1,
                page_end=1,
            )
        ],
        page_count=1,
    )
    corrected = _proposal_with_hash(content_hash="b" * 64, proposal_title="Original")
    chunker = HierarchicalChunker(target_chars=200, overlap_chars=50, min_chunk_chars=30)

    # Inject an upsert failure between the delete and the re-upsert.
    original_upsert = index.upsert

    def _failing_upsert(chunks: list[Chunk]) -> None:
        raise IndexingError("simulated Chroma write failure mid-reindex")

    with patch.object(index, "upsert", side_effect=_failing_upsert):
        with pytest.raises(IndexingError):
            reindex_proposal(
                parsed,
                corrected,
                chunker=chunker,
                index=index,
                replaced_document_id="doc_x",
            )

    # Restore the real upsert for the post-failure assertions.
    index.upsert = original_upsert  # type: ignore[method-assign]

    # Documented contract: the old chunks are gone, the new ones never
    # landed, the document_id is empty. If a future atomic rewrite makes
    # this go to 3 instead of 0, flip the assertion and remove the
    # known-follow-up bullet from the PR body.
    assert index.find_by_document_id("doc_x") == 0
    assert index.find_by_content_hash("a" * 64) == []
    assert index.find_by_content_hash("b" * 64) == []


def test_index_full_offline_flow(tmp_path: Path, no_network: None) -> None:
    """The complete chunk → embed → upsert → query path runs without network.

    This is the proof of the offline-by-default contract for the
    indexing layer. If Chroma's startup ever regains telemetry or the
    deterministic embedder grows a network call, this test goes red.
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(index_path=tmp_path, embedder=embedder, collection_name="offline")
    index.upsert(build_fixture_chunks())
    results = index.query(query_text_for("funded_horizon_europe.yaml"), top_k=2)
    assert results
    # The marker-token query should land the funded fixture at rank 1.
    assert results[0][0].metadata.source_status is SourceStatus.FUNDED
