"""Pydantic input/output models for the export service.

The export service is the UI-independent seam that turns a
:class:`eurpe.generation.GenerationDraft` into a byte string a caller
can write to disk or stream over HTTP. The models here are the wire
contract — Task 3.1 (React UI) and any future REST endpoint both
exchange these shapes so behaviour stays consistent across surfaces.

Why a separate :class:`ExportRequest` rather than a plain
``(draft, format)`` tuple
--------------------------------------------------------------------
A typed request model lets the caller carry forward-compatible
optional fields (``run_audit``, future ``template_name``, etc.) without
breaking signatures, and it gives the FastAPI / Typer layers a single
``model_validate`` step that surfaces validation errors uniformly.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from eurpe.generation.models import GenerationDraft


class ExportFormat(StrEnum):
    """Closed enum of output formats the export service supports.

    DOCX is reserved for Task 3.3 and currently raises
    :class:`~eurpe.export.errors.UnsupportedExportFormatError` so the
    wire schema (and React UI) can already speak the value while the
    renderer is wired up.
    """

    MARKDOWN = "markdown"
    DOCX = "docx"


class ExportRequest(BaseModel):
    """Input to :meth:`ExportService.export_section`.

    ``run_audit`` defaults to True because every PRD §22 caller wants
    the release-blocking source-status check. Tests and benchmarks can
    set it to False when they know the draft is synthetic and the audit
    would only add noise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft: GenerationDraft = Field(
        description="The :class:`GenerationDraft` to render.",
    )
    format: ExportFormat = Field(
        default=ExportFormat.MARKDOWN,
        description="Output format. Markdown is the MVP target; DOCX is reserved.",
    )
    run_audit: bool = Field(
        default=True,
        description=(
            "Run :class:`eurpe.generation.CitationAudit` on the rendered "
            "output before returning. The PRD §22 release-blocking guarantee "
            "is preserved for every caller when this is True."
        ),
    )


class ExportResult(BaseModel):
    """Output of :meth:`ExportService.export_section`.

    ``content`` is the rendered string for text formats. ``byte_count``
    is computed from the UTF-8 encoding so analytics events (and HTTP
    Content-Length headers, when this is exposed over the wire) don't
    need to re-encode.

    Source-status labels are preserved end-to-end: every citation
    rendered into ``content`` carries its
    :class:`~eurpe.schema.SourceStatus` via the existing
    :class:`~eurpe.generation.render.MarkdownCitationRenderer` (which
    enforces visible badges + caveats). ``citation_count`` is the
    structural count exposed here so callers can assert AC #2 of issue
    #14 without parsing the rendered text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(description="Rendered output as a string (UTF-8).")
    format: ExportFormat = Field(description="Echoes the format used for the render.")
    byte_count: int = Field(
        ge=0,
        description="UTF-8 byte length of :attr:`content`.",
    )
    citation_count: int = Field(
        ge=0,
        description="Number of citations carried into the rendered output.",
    )
    audit_passed: bool | None = Field(
        default=None,
        description=(
            "True if :class:`CitationAudit` ran and passed, None if the "
            "audit was skipped via ``run_audit=False``. The service never "
            "returns an :class:`ExportResult` with ``audit_passed=False`` "
            "— it raises :class:`ExportAuditError` instead, so a failed "
            "draft never reaches a caller as a normal return value. The "
            "``False`` value is a reserved slot for direct "
            ":class:`ExportResult` construction (tests / alternate "
            "renderers) and is not part of the service's public contract."
        ),
    )
