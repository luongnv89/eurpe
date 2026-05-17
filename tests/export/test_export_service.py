"""Tests for :class:`eurpe.export.ExportService`.

Covers the AC #3 floor for issue #14: one happy path + one error path
per service. Builds a synthetic :class:`GenerationDraft` with one
funded citation so the audit's badge check passes without relying on
the full generation workflow.

Issue #17 (Task 3.3) extends this with happy-path DOCX coverage:

* The service hands back ``content_bytes`` for DOCX and ``None`` for
  Markdown (AC #2 — DOCX export works at all).
* The audit runs on DOCX too via the shadow Markdown the renderer
  emits, so an audit-failure draft is refused for DOCX (AC #3 — labels
  preserved regardless of format).
* ``byte_count`` reports the binary size for DOCX so analytics events
  ship a meaningful Content-Length without re-encoding.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document

from eurpe.export import (
    ExportAuditError,
    ExportFormat,
    ExportRequest,
    ExportService,
)
from eurpe.generation.models import CitationRef, GenerationDraft, GenerationRequest
from eurpe.schema import Programme, SectionType, SourceStatus


def _make_draft(*, with_citation: bool = True) -> GenerationDraft:
    """Build a minimal valid draft for service-level testing.

    One funded citation, one ``[1]`` marker in the text. The audit
    passes because the citation has a complete source-status label
    and the renderer always emits the FUNDED badge.
    """

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="describe the methodology",
        target_programme=Programme.HORIZON_EUROPE,
    )
    citations: list[CitationRef] = []
    text = "Body."
    if with_citation:
        citations = [
            CitationRef(
                citation_id=1,
                source_status=SourceStatus.FUNDED,
                programme=Programme.HORIZON_EUROPE,
                call_id="HORIZON-CL5-2024-D3-02",
                proposal_title="Sample",
                section_heading="1.2 Methodology",
                page=8,
                chunk_id="chunk-1",
                snippet="example snippet",
            )
        ]
        text = "Methodology overview with reference [1]."
    return GenerationDraft(
        section_type=SectionType.METHODOLOGY,
        text=text,
        citations=citations,
        prompt_used="prompt-for-test",
        model="deterministic-stub-v1",
        request=request,
    )


def test_export_service_markdown_happy_path() -> None:
    """Service renders Markdown and reports byte/citation counts + audit pass."""

    service = ExportService()
    result = service.export_section(ExportRequest(draft=_make_draft()))

    assert result.format is ExportFormat.MARKDOWN
    assert result.citation_count == 1
    assert result.byte_count > 0
    assert result.byte_count == len(result.content.encode("utf-8"))
    assert result.audit_passed is True
    # AC #2: source-status labels surface in the rendered output.
    assert "FUNDED" in result.content
    assert "## References" in result.content
    # Markdown branch never populates content_bytes — the wire form is
    # the UTF-8 string in ``content``.
    assert result.content_bytes is None


def test_export_service_docx_happy_path() -> None:
    """Service renders DOCX, populates bytes, and audits via the shadow string.

    Issue #17 AC #2: user can export a generated section to DOCX. The
    returned :class:`ExportResult` carries both the binary payload (for
    writing to disk) and the shadow Markdown the audit ran against (so
    callers can inspect what the renderer mirrored without re-parsing
    the docx).
    """

    service = ExportService()
    result = service.export_section(
        ExportRequest(draft=_make_draft(), format=ExportFormat.DOCX)
    )

    assert result.format is ExportFormat.DOCX
    assert result.citation_count == 1
    assert result.content_bytes is not None
    assert result.byte_count == len(result.content_bytes)
    # ``byte_count`` reports the binary size, NOT the shadow string
    # length — analytics gets the right Content-Length for free.
    assert result.byte_count != len(result.content.encode("utf-8"))
    assert result.audit_passed is True
    # AC #3: the shadow Markdown carries the source-status label so
    # downstream audit/log inspection sees the same vocabulary the
    # Markdown branch produces.
    assert "FUNDED" in result.content
    # AC #2 (round-trip): the binary payload is a valid DOCX a Word
    # or LibreOffice client can open without conversion.
    loaded = Document(BytesIO(result.content_bytes))
    paragraph_texts = [p.text for p in loaded.paragraphs]
    assert "Methodology" in paragraph_texts
    assert any("[1]" in p for p in paragraph_texts)


def test_export_service_docx_audit_failure_is_blocking() -> None:
    """A DOCX whose rendered output drops a status badge is refused.

    Wires the service with a stub DOCX renderer whose shadow Markdown
    is missing the FUNDED badge — the audit then reports ``bad_render``
    and the service raises :class:`ExportAuditError` instead of
    returning silently.
    """

    class _ShadowStripRenderer:
        def render(self, draft):  # noqa: ANN001 - test double, signature mirrors real renderer
            shadow = (
                f"# {draft.section_type.value}\n\n(rendered body omitted)\n\n## References\n\n"
            )
            return b"\x50\x4b\x03\x04stub-docx-bytes", shadow

    service = ExportService(docx_renderer=_ShadowStripRenderer())
    with pytest.raises(ExportAuditError) as excinfo:
        service.export_section(
            ExportRequest(draft=_make_draft(), format=ExportFormat.DOCX)
        )
    assert excinfo.value.finding_count >= 1


def test_export_service_audit_failure_is_blocking() -> None:
    """A draft whose rendered output drops a citation badge is refused.

    Forces the audit to fail by pointing the service at a renderer that
    emits an empty Markdown body — the rendered text won't contain the
    FUNDED badge for the draft's citation, so ``audit_rendered`` flags
    a ``bad_render`` finding.
    """

    class _StripRenderer:
        def render(self, draft):  # noqa: ANN001 - test double, signature mirrors real renderer
            return f"# {draft.section_type.value}\n\n(rendered body omitted)\n"

    service = ExportService(markdown_renderer=_StripRenderer())
    with pytest.raises(ExportAuditError) as excinfo:
        service.export_section(ExportRequest(draft=_make_draft()))
    assert excinfo.value.finding_count >= 1


def test_export_service_run_audit_false_returns_audit_none() -> None:
    """Opting out of the audit skips the badge check (synthetic-test escape hatch)."""

    service = ExportService()
    result = service.export_section(ExportRequest(draft=_make_draft(), run_audit=False))
    assert result.audit_passed is None


def test_export_service_handles_empty_citations() -> None:
    """A draft with zero citations renders without raising and reports count 0."""

    service = ExportService()
    result = service.export_section(ExportRequest(draft=_make_draft(with_citation=False)))
    assert result.citation_count == 0
    assert "_No citations._" in result.content
