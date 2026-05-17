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

    Both formats produce a render that preserves citations and the
    ``SourceStatus`` labels end-to-end (PRD §22). Adding a new format
    is a two-step change: add the enum member here, wire a renderer
    into :class:`~eurpe.export.ExportService.export_section`.
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

    ``content`` is the textual rendering of the draft. For Markdown
    exports it is the rendered Markdown source. For DOCX exports it
    is the *shadow Markdown* mirror the DOCX renderer emits alongside
    the binary payload — it is the same byte string the Markdown
    branch produces, so the citation audit can apply the same
    PRD §22 checks regardless of the chosen format.

    ``content_bytes`` holds the binary payload for formats whose wire
    form is not a UTF-8 string (DOCX today, PDF in a hypothetical
    future). It is ``None`` for text-only formats. A caller writing
    to disk picks the field that matches the format:

    .. code-block:: python

        if result.content_bytes is not None:
            path.write_bytes(result.content_bytes)
        else:
            path.write_text(result.content, encoding="utf-8")

    ``byte_count`` is the on-the-wire byte length: ``len(content_bytes)``
    for binary formats, ``len(content.encode("utf-8"))`` for text
    formats. Analytics events (and a future HTTP Content-Length
    header) read this value directly — they should never need to
    inspect the format enum to know which field to size.

    Source-status labels are preserved end-to-end: every citation
    rendered into ``content`` (and the matching DOCX paragraph tree)
    carries its :class:`~eurpe.schema.SourceStatus` via the existing
    :class:`~eurpe.generation.render.MarkdownCitationRenderer` (and,
    for DOCX, :class:`~eurpe.export.docx.DocxCitationRenderer`).
    ``citation_count`` is the structural count exposed here so callers
    can assert AC #2 of issue #14 without parsing the rendered text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(
        description=(
            "Textual rendering of the draft. For MARKDOWN this is the "
            "rendered Markdown source. For DOCX this is the shadow "
            "Markdown the DocxCitationRenderer emits alongside the "
            "binary payload — same content the audit checks."
        ),
    )
    content_bytes: bytes | None = Field(
        default=None,
        description=(
            "Binary payload for formats whose wire form is not a UTF-8 "
            "string (DOCX). ``None`` for text-only formats. A caller "
            "writing to disk picks bytes when present and falls back to "
            "``content.encode('utf-8')`` otherwise."
        ),
    )
    format: ExportFormat = Field(description="Echoes the format used for the render.")
    byte_count: int = Field(
        ge=0,
        description=(
            "On-the-wire byte length. ``len(content_bytes)`` for binary "
            "formats, ``len(content.encode('utf-8'))`` for text formats. "
            "Set by the service so analytics events do not need to "
            "switch on :attr:`format`."
        ),
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
