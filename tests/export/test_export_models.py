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
