"""Unit tests for :func:`eurpe.retrieval.dedup.evaluate_duplicate`.

The helper is pure — it only reads from the index — so the tests use a
real :class:`ChromaIndex` populated through the public ``upsert`` API
rather than mocking out the four query helpers. Building a stub would
double the code with no fidelity win, and the real Chroma path is the
one production uses.

The fixtures cover all four decision branches plus the title-None
guard that keeps two title-less proposals in the same call from
silently colliding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eurpe.retrieval import (
    ChromaIndex,
    Chunk,
    DeterministicHashEmbedder,
    DuplicateAction,
    evaluate_duplicate,
)
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SourceStatus,
)


def _proposal(
    *,
    content_hash: str | None,
    proposal_title: str | None,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    source_path: str = "data/corpus/sample.pdf",
) -> ProposalMetadata:
    """Minimal valid :class:`ProposalMetadata` carrying an optional hash."""

    return ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id=call_id,
        year=2024,
        outcome=SourceStatus.FUNDED,
        proposal_title=proposal_title,
        source_path=source_path,
        content_hash=content_hash,
    )


def _chunk(proposal: ProposalMetadata, *, document_id: str, chunk_index: int = 0) -> Chunk:
    """Synthesise a chunk for ``proposal`` with the requested document_id.

    Text contains the document_id so two different docs produce two
    different chunk_ids (chunk_id embeds a hash of the text).
    """

    anchor = CitationAnchor(document_id=document_id)
    meta = ChunkMetadata(
        proposal=proposal,
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=proposal.outcome,
    )
    return Chunk(
        text=f"sample text for {document_id} chunk {chunk_index}",
        metadata=meta,
    )


@pytest.fixture
def index(tmp_path: Path) -> ChromaIndex:
    """Empty :class:`ChromaIndex` for the test to populate as it sees fit."""

    return ChromaIndex(
        index_path=tmp_path,
        embedder=DeterministicHashEmbedder(dimension=32),
        collection_name="dedup",
    )


def test_evaluate_duplicate_returns_none_for_fresh_proposal(index: ChromaIndex) -> None:
    decision = evaluate_duplicate(
        index=index,
        content_hash="a" * 64,
        proposal_title="Brand New",
        call_id="HORIZON-CL5-2024-D3-02",
        new_document_id="brand_new_doc",
    )
    assert decision.action is DuplicateAction.NONE
    assert decision.conflicting_document_id is None


def test_evaluate_duplicate_block_hard_on_hash_collision(index: ChromaIndex) -> None:
    """Byte-identical content already in the index → ``BLOCK_HARD``."""

    existing_hash = "a" * 64
    proposal = _proposal(content_hash=existing_hash, proposal_title="Existing")
    index.upsert([_chunk(proposal, document_id="existing_doc")])

    decision = evaluate_duplicate(
        index=index,
        content_hash=existing_hash,
        proposal_title="Different Title Entirely",
        call_id="DIFFERENT-CALL",
        new_document_id="new_doc",
    )
    assert decision.action is DuplicateAction.BLOCK_HARD
    assert decision.conflicting_document_id == "existing_doc"


def test_evaluate_duplicate_block_hard_when_hash_and_document_id_both_match(
    index: ChromaIndex,
) -> None:
    """Hash check wins over doc_id check — same bytes re-uploaded under same name is BLOCK_HARD.

    Pins the order-of-checks contract in :func:`evaluate_duplicate`: the
    hash branch fires first, so a byte-identical re-upload (even with the
    same archive-stem document_id) is rejected as a hard duplicate rather
    than silently re-indexed. A future refactor that flips the order to
    "doc_id first" would otherwise convert this case to REINDEX without
    any other test failing.
    """

    existing_hash = "a" * 64
    proposal = _proposal(content_hash=existing_hash, proposal_title="Same Doc")
    index.upsert([_chunk(proposal, document_id="shared_doc_id")])

    decision = evaluate_duplicate(
        index=index,
        content_hash=existing_hash,  # same bytes
        proposal_title="Same Doc",
        call_id="HORIZON-CL5-2024-D3-02",
        new_document_id="shared_doc_id",  # same doc_id too
    )
    assert decision.action is DuplicateAction.BLOCK_HARD
    assert decision.conflicting_document_id == "shared_doc_id"


def test_evaluate_duplicate_reindex_on_document_id_collision(
    index: ChromaIndex,
) -> None:
    """Same archive-stem doc_id, different bytes → ``REINDEX``."""

    proposal = _proposal(content_hash="a" * 64, proposal_title="Original Title")
    index.upsert([_chunk(proposal, document_id="shared_doc_id")])

    decision = evaluate_duplicate(
        index=index,
        content_hash="b" * 64,  # different bytes
        proposal_title="Original Title",
        call_id="HORIZON-CL5-2024-D3-02",
        new_document_id="shared_doc_id",  # same doc_id
    )
    assert decision.action is DuplicateAction.REINDEX
    assert decision.conflicting_document_id == "shared_doc_id"


def test_evaluate_duplicate_block_soft_on_title_and_call_collision(
    index: ChromaIndex,
) -> None:
    """Same (title, call_id), different hash and document_id → ``BLOCK_SOFT``."""

    proposal = _proposal(
        content_hash="a" * 64,
        proposal_title="Shared Title",
        call_id="HORIZON-CL5-2024-D3-02",
    )
    index.upsert([_chunk(proposal, document_id="old_stem")])

    decision = evaluate_duplicate(
        index=index,
        content_hash="b" * 64,
        proposal_title="Shared Title",
        call_id="HORIZON-CL5-2024-D3-02",
        new_document_id="new_stem",
    )
    assert decision.action is DuplicateAction.BLOCK_SOFT
    assert decision.conflicting_document_id == "old_stem"


def test_evaluate_duplicate_block_soft_skipped_when_existing_title_is_none(
    index: ChromaIndex,
) -> None:
    """Two title-less records in the same call do not collide softly."""

    proposal = _proposal(
        content_hash="a" * 64,
        proposal_title=None,  # existing record has no title
        call_id="HORIZON-CL5-2024-D3-02",
    )
    index.upsert([_chunk(proposal, document_id="title_less_existing")])

    decision = evaluate_duplicate(
        index=index,
        content_hash="b" * 64,
        proposal_title=None,  # incoming also title-less
        call_id="HORIZON-CL5-2024-D3-02",
        new_document_id="title_less_incoming",
    )
    assert decision.action is DuplicateAction.NONE


def test_evaluate_duplicate_block_soft_skipped_when_incoming_title_is_none(
    index: ChromaIndex,
) -> None:
    """Incoming title None disables the title+call branch entirely."""

    proposal = _proposal(
        content_hash="a" * 64,
        proposal_title="Existing With Title",
        call_id="HORIZON-CL5-2024-D3-02",
    )
    index.upsert([_chunk(proposal, document_id="titled_existing")])

    decision = evaluate_duplicate(
        index=index,
        content_hash="b" * 64,
        proposal_title=None,
        call_id="HORIZON-CL5-2024-D3-02",
        new_document_id="title_less_new",
    )
    assert decision.action is DuplicateAction.NONE
