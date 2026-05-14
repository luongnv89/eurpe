"""Tests for ``eurpe.ingestion.errors``.

The error hierarchy is the contract callers rely on to distinguish
"bad file type" from "parser blew up" without inspecting strings, so
every guarantee gets at least one explicit assertion.
"""

from __future__ import annotations

import pytest

from eurpe.ingestion import IngestionError, ParserError, UnsupportedFormatError


def test_unsupported_format_is_ingestion_error() -> None:
    """Catching ``IngestionError`` must catch unsupported-format failures too."""

    err = UnsupportedFormatError("not a pdf")
    assert isinstance(err, IngestionError)


def test_parser_error_is_ingestion_error() -> None:
    err = ParserError("/tmp/x.pdf", "boom")
    assert isinstance(err, IngestionError)


def test_parser_error_message_includes_source_path() -> None:
    """The string form must surface the failing path so logs are actionable."""

    err = ParserError("/abs/path/to/proposal.pdf", "docling crashed")
    assert "/abs/path/to/proposal.pdf" in str(err)
    assert "docling crashed" in str(err)


def test_parser_error_exposes_source_path_attribute() -> None:
    err = ParserError("/abs/path/to/proposal.pdf", "boom")
    assert err.source_path == "/abs/path/to/proposal.pdf"


def test_parser_error_preserves_cause_attribute() -> None:
    inner = ValueError("the underlying problem")
    err = ParserError("/tmp/x.pdf", "boom", cause=inner)
    assert err.cause is inner


def test_parser_error_chains_via_raise_from() -> None:
    """``raise X from Y`` wires ``__cause__`` — confirm it survives."""

    inner = ValueError("the underlying problem")
    try:
        raise ParserError("/tmp/x.pdf", "boom", cause=inner) from inner
    except ParserError as exc:
        assert exc.__cause__ is inner
        assert exc.cause is inner
    else:  # pragma: no cover - defensive
        pytest.fail("ParserError should have been raised")


def test_parser_error_cause_defaults_to_none() -> None:
    err = ParserError("/tmp/x.pdf", "no inner")
    assert err.cause is None


def test_unsupported_format_carries_message() -> None:
    err = UnsupportedFormatError("Unsupported file extension '.docx'")
    assert "docx" in str(err)
