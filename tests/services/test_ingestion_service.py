"""Tests for :class:`eurpe.ingestion.service.IngestionService`.

Avoid running Docling (slow + heavy) by passing a pre-built
:class:`ParsedProposal` to the service. The service runs the
chunker + duplicate-eval + index-upsert path, which is the part this
issue's AC actually cares about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eurpe.ingestion.models import ParsedProposal, ParsedSection
from eurpe.ingestion.service import (
    DuplicateRefusedError,
    IngestionRequest,
    IngestionService,
)
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    HierarchicalChunker,
)
from eurpe.schema import Programme, ProposalMetadata, SourceStatus


@pytest.fixture
def ingestion_service(tmp_path) -> IngestionService:
    """Build a service backed by a fresh on-disk index.

    Parser is unused in these tests (callers pre-supply ``parsed``)
    but the service constructor still requires one; a real
    :class:`DoclingProposalParser` is fine — Docling is only loaded
    when ``.parse()`` is called.
    """

    from eurpe.ingestion.docling_parser import DoclingProposalParser

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="ingestion_service_tests",
    )
    return IngestionService(
        parser=DoclingProposalParser(offline=True),
        chunker=HierarchicalChunker(),
        index=index,
    )


def _make_parsed(source_path: str = "data/sample.pdf") -> ParsedProposal:
    return ParsedProposal(
        title="Sample Proposal",
        sections=[
            ParsedSection(
                heading="1.2 Methodology",
                level=2,
                text=(
                    "We propose a deep learning pipeline. "
                    "The methodology covers data ingestion and model training."
                ),
                page_start=8,
                page_end=8,
            ),
        ],
        source_path=source_path,
        page_count=12,
        parser="docling-test-stub",
        parsed_at=datetime.now(UTC),
    )


def _make_proposal(content_hash: str = "a" * 64) -> ProposalMetadata:
    return ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        topic_id="HORIZON-CL5-2024-D3-02-01",
        year=2024,
        outcome=SourceStatus.FUNDED,
        proposal_title="Sample Proposal",
        consortium_acronym="SAMPLE",
        source_path="data/sample.pdf",
        content_hash=content_hash,
    )


def test_ingestion_service_indexes_proposal_happy_path(ingestion_service) -> None:
    """Fresh proposal: service chunks, upserts, and reports decision NONE."""

    result = ingestion_service.ingest_proposal(
        IngestionRequest(
            proposal=_make_proposal(),
            parsed=_make_parsed(),
            document_id="sample",
        )
    )
    assert result.chunks_added >= 1
    assert result.duplicate_decision.value == "none"
    assert result.replaced_document_id is None


def test_ingestion_service_blocks_hard_duplicate(ingestion_service) -> None:
    """Re-ingesting identical bytes raises :class:`DuplicateRefusedError`.

    First call seeds the index; the second call hits the BLOCK_HARD
    branch because the content hash matches an indexed proposal.
    """

    proposal = _make_proposal()
    parsed = _make_parsed()
    ingestion_service.ingest_proposal(
        IngestionRequest(proposal=proposal, parsed=parsed, document_id="sample")
    )

    with pytest.raises(DuplicateRefusedError) as excinfo:
        ingestion_service.ingest_proposal(
            IngestionRequest(proposal=proposal, parsed=parsed, document_id="sample")
        )
    assert excinfo.value.decision.action.value == "block_hard"


def test_ingestion_service_requires_parsed_or_pdf_path() -> None:
    """Caller must supply exactly one of ``parsed`` / ``pdf_path``.

    The validator on :class:`IngestionRequest` fires at construction
    time so a FastAPI handler returns a 422 instead of a
    :class:`ValueError` from deep inside the service.
    """

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="parsed.*pdf_path"):
        IngestionRequest(proposal=_make_proposal(), document_id="sample")


def test_ingestion_service_requires_content_hash_when_pdf_path_missing(
    ingestion_service,
) -> None:
    """Pre-parsed callers must stamp the content hash themselves."""

    proposal = ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        year=2024,
        outcome=SourceStatus.FUNDED,
        proposal_title="Sample Proposal",
        source_path="data/sample.pdf",
        # No content_hash stamped — service should refuse rather than
        # compute one from non-existent bytes.
    )
    with pytest.raises(ValueError, match="content_hash"):
        ingestion_service.ingest_proposal(
            IngestionRequest(
                proposal=proposal,
                parsed=_make_parsed(),
                document_id="sample",
            )
        )


def test_ingestion_service_rejects_both_parsed_and_pdf_path(tmp_path: Path) -> None:
    """Supplying both ``parsed`` and ``pdf_path`` is a programmer error.

    Same construction-time guarantee as the
    ``test_ingestion_service_requires_parsed_or_pdf_path`` test.
    """

    from pydantic import ValidationError

    fake_pdf = tmp_path / "doesnt-matter.pdf"
    fake_pdf.write_bytes(b"%PDF-1.0\n")
    with pytest.raises(ValidationError, match="exactly one"):
        IngestionRequest(
            proposal=_make_proposal(),
            parsed=_make_parsed(),
            pdf_path=fake_pdf,
            document_id="sample",
        )
