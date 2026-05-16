"""Service facade for exporting generated drafts.

The export service is the UI-independent entry point Task 3.1 (React
UI) and any future REST endpoint use to turn a
:class:`eurpe.generation.GenerationDraft` into a byte string.

Why this lives in a service rather than directly in the CLI
-----------------------------------------------------------
Before this issue, exporting was inlined in
``src/eurpe/generation/cli.py``: the CLI ran the renderer, ran the
audit, atomically wrote the file, and logged the analytics event in
~30 lines of orchestration. The React UI and any future HTTP export
endpoint would each have had to reimplement that chain. Lifting it
into a single :class:`ExportService` keeps the source-status audit on
one well-tested code path (PRD §22 release-blocking) and lets the CLI
shrink to "build request → call service → write bytes".
"""

from __future__ import annotations

from eurpe.export.errors import ExportAuditError, UnsupportedExportFormatError
from eurpe.export.models import ExportFormat, ExportRequest, ExportResult
from eurpe.generation.audit import CitationAudit
from eurpe.generation.render import MarkdownCitationRenderer


class ExportService:
    """Render a :class:`GenerationDraft` to one of the supported export formats.

    Stateless aside from the renderer + audit it holds. Safe to share
    across threads and across requests; tests build a fresh instance
    per case because the cost is trivial.

    The service ALWAYS runs the citation audit by default
    (``ExportRequest.run_audit=True``) so any caller that goes through
    the service gets the PRD §22 source-status guarantee. Skipping the
    audit is opt-in via the request flag — the CLI / API never sets
    that flag, only synthetic tests do.
    """

    def __init__(
        self,
        *,
        markdown_renderer: MarkdownCitationRenderer | None = None,
        audit: CitationAudit | None = None,
    ) -> None:
        # Allow injection so future formats (DOCX) or test doubles can
        # be wired without reaching into the service internals. Default
        # construction matches what the CLI does today.
        self._markdown_renderer = markdown_renderer or MarkdownCitationRenderer()
        self._audit = audit or CitationAudit()

    def export_section(self, request: ExportRequest) -> ExportResult:
        """Render ``request.draft`` and return an :class:`ExportResult`.

        Errors:

        * :class:`UnsupportedExportFormatError` — DOCX is reserved for
          Task 3.3; today the service refuses anything but Markdown.
        * :class:`ExportAuditError` — the rendered output failed the
          citation audit (a citation lost its source-status label or a
          required badge is missing from the Markdown). Raising rather
          than returning protects PRD §22.
        """

        if request.format is ExportFormat.MARKDOWN:
            content = self._markdown_renderer.render(request.draft)
        elif request.format is ExportFormat.DOCX:
            # DOCX is intentionally a reserved enum slot — see
            # eurpe.export.models.ExportFormat for the rationale.
            raise UnsupportedExportFormatError(
                "DOCX export is reserved for Task 3.3 and not yet implemented; "
                "use ExportFormat.MARKDOWN for the MVP."
            )
        else:  # pragma: no cover - StrEnum guards against this
            raise UnsupportedExportFormatError(f"unsupported export format: {request.format!r}")

        audit_passed: bool | None = None
        if request.run_audit:
            audit_result = self._audit.audit_rendered(request.draft, content)
            audit_passed = audit_result.passed
            if not audit_result.passed:
                raise ExportAuditError(
                    f"export refused: citation audit failed with "
                    f"{len(audit_result.findings)} finding(s); "
                    "rendered draft would have shipped without complete "
                    "source-status labels.",
                    finding_count=len(audit_result.findings),
                )

        return ExportResult(
            content=content,
            format=request.format,
            byte_count=len(content.encode("utf-8")),
            citation_count=len(request.draft.citations),
            audit_passed=audit_passed,
        )
