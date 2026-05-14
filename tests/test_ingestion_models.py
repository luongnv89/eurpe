"""Tests for ``eurpe.ingestion.models``.

These tests do not import Docling, so they run on every install — including
ones where the heavy parser dependencies are unavailable. The model layer
is the contract every downstream consumer relies on, so the validators
are exercised explicitly (level bounds, ``extra="forbid"``, default
factories) to catch silent regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eurpe.ingestion import ParsedProposal, ParsedSection, ParsedTable


def _valid_section(**overrides: object) -> ParsedSection:
    """Build a valid section, allowing per-test overrides."""

    base: dict[str, object] = {
        "heading": "1. Introduction",
        "level": 1,
        "text": "Body text.",
    }
    base.update(overrides)
    return ParsedSection(**base)  # type: ignore[arg-type]


def test_parsed_section_rejects_level_zero() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _valid_section(level=0)
    assert "level" in str(excinfo.value)


def test_parsed_section_rejects_level_seven() -> None:
    """``level`` upper bound mirrors HTML H1..H6 — anything deeper is suspicious."""

    with pytest.raises(ValidationError) as excinfo:
        _valid_section(level=7)
    assert "level" in str(excinfo.value)


def test_parsed_section_accepts_levels_one_through_six() -> None:
    for level in range(1, 7):
        section = _valid_section(level=level)
        assert section.level == level


def test_parsed_section_defaults_tables_to_empty_list() -> None:
    section = _valid_section()
    assert section.tables == []
    # Mutating the default must not bleed into other instances — confirms
    # we're using ``default_factory`` and not a shared mutable default.
    section.tables.append(ParsedTable(rows=[["x"]]))
    other = _valid_section()
    assert other.tables == []


def test_parsed_section_rejects_empty_heading() -> None:
    with pytest.raises(ValidationError):
        _valid_section(heading="")


def test_parsed_section_rejects_extra_field() -> None:
    """``extra="forbid"`` catches typos like ``page_starts``."""

    with pytest.raises(ValidationError) as excinfo:
        ParsedSection(
            heading="x",
            level=1,
            text="t",
            page_starts=1,  # type: ignore[call-arg]
        )
    assert "page_starts" in str(excinfo.value)


def test_parsed_table_accepts_ragged_rows() -> None:
    """No required column count: merged cells produce ragged grids."""

    table = ParsedTable(
        rows=[["a", "b", "c"], ["d"], ["e", "f"]],
        page=1,
    )
    assert [len(r) for r in table.rows] == [3, 1, 2]


def test_parsed_table_defaults_rows_to_empty_list() -> None:
    table = ParsedTable()
    assert table.rows == []
    assert table.page is None
    assert table.section_heading is None


def test_parsed_table_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        ParsedTable(rows=[["a"]], unknown_field=42)  # type: ignore[call-arg]


def test_parsed_proposal_total_text_length_sums_sections() -> None:
    proposal = ParsedProposal(
        source_path="/abs/test.pdf",
        title="Test",
        sections=[
            _valid_section(text="abcde"),  # 5
            _valid_section(text=""),  # 0
            _valid_section(text="123456"),  # 6
        ],
    )
    assert proposal.total_text_length() == 11


def test_parsed_proposal_total_text_length_zero_when_no_sections() -> None:
    proposal = ParsedProposal(source_path="/abs/test.pdf")
    assert proposal.total_text_length() == 0


def test_parsed_proposal_default_parser_is_docling() -> None:
    proposal = ParsedProposal(source_path="/abs/test.pdf")
    assert proposal.parser == "docling"


def test_parsed_proposal_default_parsed_at_is_tz_aware_utc() -> None:
    proposal = ParsedProposal(source_path="/abs/test.pdf")
    assert proposal.parsed_at.tzinfo is not None
    # Same UTC offset as ``datetime.now(UTC)``.
    assert proposal.parsed_at.utcoffset() == datetime.now(UTC).utcoffset()


def test_parsed_proposal_rejects_empty_source_path() -> None:
    with pytest.raises(ValidationError):
        ParsedProposal(source_path="")


def test_parsed_proposal_rejects_extra_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ParsedProposal(source_path="/abs/test.pdf", flavor="vanilla")  # type: ignore[call-arg]
    assert "flavor" in str(excinfo.value)


def test_parsed_proposal_round_trips_via_model_dump() -> None:
    """JSON dump → reload preserves all fields including nested tables."""

    original = ParsedProposal(
        source_path="/abs/test.pdf",
        title="Round Trip",
        sections=[
            _valid_section(
                heading="2. Impact",
                level=2,
                text="impact body",
                page_start=4,
                page_end=5,
                tables=[ParsedTable(rows=[["a", "b"], ["c", "d"]], page=4)],
            )
        ],
        page_count=10,
    )
    data = original.model_dump(mode="json")
    reloaded = ParsedProposal.model_validate(data)
    assert reloaded == original
