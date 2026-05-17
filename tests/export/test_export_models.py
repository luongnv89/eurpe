"""Pydantic validation tests for the export wire models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eurpe.export.models import ExportFormat, ExportRequest, ExportResult
from eurpe.generation.models import GenerationDraft, GenerationRequest
from eurpe.schema import Programme, SectionType


def _draft() -> GenerationDraft:
    return GenerationDraft(
        section_type=SectionType.METHODOLOGY,
        text="Body.",
        citations=[],
        prompt_used="prompt",
        model="model",
        request=GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="intent",
            target_programme=Programme.HORIZON_EUROPE,
        ),
    )


def test_export_request_defaults_to_markdown_and_audit_on() -> None:
    request = ExportRequest(draft=_draft())
    assert request.format is ExportFormat.MARKDOWN
    assert request.run_audit is True


def test_export_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExportRequest(draft=_draft(), unknown="oops")  # type: ignore[call-arg]


def test_export_result_byte_count_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        ExportResult(
            content="x",
            format=ExportFormat.MARKDOWN,
            byte_count=-1,
            citation_count=0,
        )


def test_export_format_enum_round_trips_strings() -> None:
    assert ExportFormat("markdown") is ExportFormat.MARKDOWN
    assert ExportFormat("docx") is ExportFormat.DOCX


def test_export_result_content_bytes_defaults_to_none() -> None:
    """Issue #17: text-only formats leave ``content_bytes`` as ``None``.

    The default keeps the Markdown branch's API backwards-compatible —
    callers that construct an :class:`ExportResult` directly (tests,
    alternate renderers) do not need to pass a binary payload they do
    not have.
    """

    result = ExportResult(
        content="rendered markdown",
        format=ExportFormat.MARKDOWN,
        byte_count=len("rendered markdown"),
        citation_count=0,
    )
    assert result.content_bytes is None


def test_export_result_accepts_binary_content_bytes() -> None:
    """Issue #17 AC #2: binary formats round-trip through ``content_bytes``.

    A frozen model with an Optional ``bytes`` field is a typical
    Pydantic shape but worth pinning explicitly — a future refactor
    that accidentally narrows the type to ``str`` would silently
    corrupt the DOCX wire form by triggering an implicit decode.
    """

    payload = b"\x50\x4b\x03\x04docx-stub"
    result = ExportResult(
        content="shadow markdown",
        content_bytes=payload,
        format=ExportFormat.DOCX,
        byte_count=len(payload),
        citation_count=0,
    )
    assert result.content_bytes == payload
    # Byte count is the on-the-wire size of the binary payload, not
    # the shadow string. The service is responsible for setting this
    # correctly; the model only enforces non-negativity.
    assert result.byte_count == len(payload)
