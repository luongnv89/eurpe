"""Tests for ``eurpe.schema``.

Covers the three Pydantic models (``ProposalMetadata``, ``ChunkMetadata``,
``CitationAnchor``) and the four YAML fixtures under
``tests/fixtures/metadata/``. Each validation rule that the schema enforces
gets at least one negative test so a future refactor cannot quietly weaken a
guarantee.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "metadata"
FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.yaml"))


def _valid_proposal_kwargs(**overrides: object) -> dict[str, object]:
    """Return kwargs that build a valid ``ProposalMetadata`` by default.

    Tests override individual fields to exercise specific validation paths
    without rewriting the whole record.
    """

    base: dict[str, object] = {
        "programme": Programme.HORIZON_EUROPE,
        "call_id": "HORIZON-CL5-2024-D3-02",
        "topic_id": "HORIZON-CL5-2024-D3-02-01",
        "year": 2024,
        "outcome": SourceStatus.FUNDED,
        "source_path": "data/corpus/funded/example.pdf",
    }
    base.update(overrides)
    return base


def _valid_anchor() -> CitationAnchor:
    return CitationAnchor(
        document_id="doc-1",
        section_heading="1. Excellence",
        page=3,
        char_start=100,
        char_end=500,
    )


def test_fixture_directory_has_one_file_per_source_status() -> None:
    """The PRD requires fixture coverage for every ``SourceStatus`` value."""

    assert len(FIXTURE_FILES) == 4, f"Expected 4 fixtures, found {[f.name for f in FIXTURE_FILES]}"
    statuses_in_files = set()
    for path in FIXTURE_FILES:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        statuses_in_files.add(raw["source_status"])
    assert statuses_in_files == {s.value for s in SourceStatus}


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_FILES,
    ids=lambda p: p.name,
)
def test_chunk_metadata_round_trips_all_fixtures(fixture_path: Path) -> None:
    """Load → validate → re-serialize → re-validate; equality must hold."""

    raw = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    chunk = ChunkMetadata.model_validate(raw)

    # Invariant: chunk.source_status equals proposal.outcome (the validator
    # would have raised already, but assert it here so a future bug shows up
    # in the round-trip test rather than only in the negative test below).
    assert chunk.source_status is chunk.proposal.outcome

    dumped = chunk.model_dump(mode="json")
    serialized_yaml = yaml.safe_dump(dumped, sort_keys=False)
    reloaded = ChunkMetadata.model_validate(yaml.safe_load(serialized_yaml))
    assert reloaded == chunk


def test_proposal_metadata_accepts_minimal_valid_record() -> None:
    """Sanity: the helper builds a valid record."""

    proposal = ProposalMetadata(**_valid_proposal_kwargs())
    assert proposal.programme is Programme.HORIZON_EUROPE
    assert proposal.outcome is SourceStatus.FUNDED
    assert proposal.language == "en"
    assert proposal.ingested_at is not None
    # Default factory must produce a tz-aware UTC datetime (no naive datetimes).
    assert proposal.ingested_at.tzinfo is not None
    assert proposal.ingested_at.utcoffset() == datetime.now(UTC).utcoffset()


def test_proposal_metadata_rejects_missing_outcome() -> None:
    kwargs = _valid_proposal_kwargs()
    kwargs.pop("outcome")
    with pytest.raises(ValidationError) as excinfo:
        ProposalMetadata(**kwargs)
    assert "outcome" in str(excinfo.value)


def test_proposal_metadata_rejects_missing_programme() -> None:
    kwargs = _valid_proposal_kwargs()
    kwargs.pop("programme")
    with pytest.raises(ValidationError) as excinfo:
        ProposalMetadata(**kwargs)
    assert "programme" in str(excinfo.value)


def test_proposal_metadata_rejects_empty_call_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ProposalMetadata(**_valid_proposal_kwargs(call_id="   "))
    assert "call_id" in str(excinfo.value)


def test_proposal_metadata_rejects_missing_source_path() -> None:
    kwargs = _valid_proposal_kwargs()
    kwargs.pop("source_path")
    with pytest.raises(ValidationError) as excinfo:
        ProposalMetadata(**kwargs)
    assert "source_path" in str(excinfo.value)


@pytest.mark.parametrize("bad_year", [1900, 2200, 0])
def test_proposal_metadata_rejects_out_of_range_year(bad_year: int) -> None:
    with pytest.raises(ValidationError):
        ProposalMetadata(**_valid_proposal_kwargs(year=bad_year))


def test_chunk_metadata_rejects_status_drift() -> None:
    proposal = ProposalMetadata(**_valid_proposal_kwargs(outcome=SourceStatus.FUNDED))
    with pytest.raises(ValidationError) as excinfo:
        ChunkMetadata(
            proposal=proposal,
            section_type=SectionType.METHODOLOGY,
            chunk_index=0,
            anchor=_valid_anchor(),
            source_status=SourceStatus.REJECTED,
        )
    message = str(excinfo.value)
    assert "drift" in message.lower()
    assert "funded" in message
    assert "rejected" in message


def test_citation_anchor_is_hashable() -> None:
    anchor_a = _valid_anchor()
    anchor_b = _valid_anchor()
    assert anchor_a == anchor_b
    assert hash(anchor_a) == hash(anchor_b)
    # Hashable means usable in a set / dict.
    deduped = {anchor_a, anchor_b}
    assert len(deduped) == 1


def test_citation_anchor_rejects_inverted_offsets() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CitationAnchor(
            document_id="doc-1",
            char_start=100,
            char_end=50,
        )
    assert "char_end" in str(excinfo.value)


def test_citation_anchor_allows_omitted_offsets() -> None:
    """Missing offsets are valid (e.g., DOCX sources without pagination)."""

    anchor = CitationAnchor(document_id="doc-1")
    assert anchor.page is None
    assert anchor.char_start is None
    assert anchor.char_end is None


def test_extra_field_forbidden_on_proposal() -> None:
    """Catches typos like ``yr=2024`` instead of ``year=2024``."""

    kwargs = _valid_proposal_kwargs()
    kwargs["yr"] = 2024  # typo
    with pytest.raises(ValidationError) as excinfo:
        ProposalMetadata(**kwargs)
    assert "yr" in str(excinfo.value)


def test_extra_field_forbidden_on_chunk() -> None:
    proposal = ProposalMetadata(**_valid_proposal_kwargs())
    with pytest.raises(ValidationError) as excinfo:
        ChunkMetadata(
            proposal=proposal,
            chunk_index=0,
            anchor=_valid_anchor(),
            source_status=SourceStatus.FUNDED,
            unexpected_field="oops",
        )
    assert "unexpected_field" in str(excinfo.value)


def test_enums_have_required_members() -> None:
    """Every ``SourceStatus`` value the PRD calls out must exist."""

    assert {member.value for member in SourceStatus} == {
        "funded",
        "rejected",
        "esr_note",
        "unknown",
    }
    assert SourceStatus.FUNDED.value == "funded"
    assert SourceStatus.REJECTED.value == "rejected"
    assert SourceStatus.ESR_NOTE.value == "esr_note"
    assert SourceStatus.UNKNOWN.value == "unknown"


def test_chunk_metadata_uses_default_section_type_other() -> None:
    proposal = ProposalMetadata(**_valid_proposal_kwargs())
    chunk = ChunkMetadata(
        proposal=proposal,
        chunk_index=0,
        anchor=_valid_anchor(),
        source_status=SourceStatus.FUNDED,
    )
    assert chunk.section_type is SectionType.OTHER


def test_proposal_metadata_rejects_whitespace_source_path() -> None:
    """``source_path`` must reject whitespace-only strings.

    ``min_length=1`` only checks raw character count, so a non-empty
    whitespace string would slip past it without the dedicated validator.
    Mirrors the ``call_id`` whitespace contract.
    """

    with pytest.raises(ValidationError) as excinfo:
        ProposalMetadata(**_valid_proposal_kwargs(source_path="   "))
    assert "source_path" in str(excinfo.value)


def test_chunk_metadata_status_drift_blocked_after_construction() -> None:
    """Mutating ``chunk.source_status`` post-construction must re-validate.

    Without ``validate_assignment=True`` the model-validator only runs at
    construction time, leaving a hole where a caller can flip the label
    after the fact. The model_config flag closes that hole.
    """

    proposal = ProposalMetadata(**_valid_proposal_kwargs(outcome=SourceStatus.FUNDED))
    chunk = ChunkMetadata(
        proposal=proposal,
        section_type=SectionType.METHODOLOGY,
        chunk_index=0,
        anchor=_valid_anchor(),
        source_status=SourceStatus.FUNDED,
    )
    with pytest.raises(ValidationError, match="source_status"):
        chunk.source_status = SourceStatus.REJECTED
