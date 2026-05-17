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

Format dispatch
---------------
The service supports two formats today:

* :attr:`ExportFormat.MARKDOWN` — UTF-8 Markdown string. ``content``
  carries the rendered Markdown; ``content_bytes`` is ``None``.
* :attr:`ExportFormat.DOCX` — Office Open XML byte stream. ``content``
  carries the *shadow Markdown* the DOCX renderer emits so the audit
  can apply the same release-blocking checks; ``content_bytes`` holds
  the binary payload a caller writes to a ``.docx`` file.

Both branches feed the audit the same textual surface (the Markdown /
shadow Markdown) so PRD §22 stays on one code path regardless of the
chosen format.
"""

from __future__ import annotations

from eurpe.export.docx import DocxCitationRenderer
from eurpe.export.errors import ExportAuditError, UnsupportedExportFormatError
from eurpe.export.models import ExportFormat, ExportRequest, ExportResult
from eurpe.generation.audit import CitationAudit
from eurpe.generation.render import MarkdownCitationRenderer


class ExportService:
    """Render a :class:`GenerationDraft` to one of the supported export formats.

    Stateless aside from the renderers + audit it holds. Safe to share
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
        docx_renderer: DocxCitationRenderer | None = None,
        audit: CitationAudit | None = None,
    ) -> None:
        # Allow injection so test doubles can be wired without reaching
        # into the service internals. Defaults match what the CLI does
        # today. The DocxCitationRenderer reuses the same
        # MarkdownCitationRenderer for its shadow string so a stub
        # markdown renderer passed here would not affect the DOCX
        # branch — wire a custom DocxCitationRenderer explicitly when
        # you need to stub the shadow path.
        self._markdown_renderer = markdown_renderer or MarkdownCitationRenderer()
        self._docx_renderer = docx_renderer or DocxCitationRenderer()
        self._audit = audit or CitationAudit()

    def export_section(self, request: ExportRequest) -> ExportResult:
        """Render ``request.draft`` and return an :class:`ExportResult`.

        Errors:

        * :class:`UnsupportedExportFormatError` — the requested format
          has no renderer wired up (e.g., a hypothetical PDF slot).
        * :class:`ExportAuditError` — the rendered output failed the
          citation audit (a citation lost its source-status label or a
          required badge is missing from the textual surface). Raising
          rather than returning protects PRD §22.
        """

        content_bytes: bytes | None = None
        if request.format is ExportFormat.MARKDOWN:
            content = self._markdown_renderer.render(request.draft)
        elif request.format is ExportFormat.DOCX:
            # The DOCX renderer returns ``(bytes, shadow_md)`` so the
            # audit can run against the shadow string instead of
            # opening the binary payload back up. Shadow content is
            # byte-equal to the Markdown renderer's output for the
            # same draft — see eurpe.export.docx module docstring.
            content_bytes, content = self._docx_renderer.render(request.draft)
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

        # ``byte_count`` is the on-the-wire size: bytes for binary
        # formats, UTF-8 byte length of ``content`` otherwise. A
        # caller writing to disk picks the matching field; analytics
        # never need to switch on the format enum.
        byte_count = (
            len(content_bytes) if content_bytes is not None else len(content.encode("utf-8"))
        )

        return ExportResult(
            content=content,
            content_bytes=content_bytes,
            format=request.format,
            byte_count=byte_count,
            citation_count=len(request.draft.citations),
            audit_passed=audit_passed,
        )
