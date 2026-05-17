"""Export package — UI-independent rendering of generated drafts.

The export service is the seam Task 3.1 (React UI) and any future
REST/CLI export endpoint use to turn a
:class:`eurpe.generation.GenerationDraft` into a byte string with
source-status labels preserved (PRD §22 + §59).

Public surface kept narrow on purpose so callers import from
``eurpe.export`` and never reach into internal modules:

* :class:`ExportService` — the only public entry point.
* :class:`ExportRequest`, :class:`ExportResult`, :class:`ExportFormat` —
  the wire models.
* :class:`DocxCitationRenderer` — the DOCX renderer for callers that
  want to drive it directly (tests, alternative servers).
* :class:`ExportError`, :class:`UnsupportedExportFormatError`,
  :class:`ExportAuditError` — narrow exception types.

The actual Markdown rendering lives in
:mod:`eurpe.generation.render` (the :class:`MarkdownCitationRenderer`
class); the DOCX rendering lives in :mod:`eurpe.export.docx`. Export
depends on generation; generation never depends on export. This
one-way dependency avoids the cycle the previous service-boundary
design risked.
"""

from __future__ import annotations

from eurpe.export.docx import DocxCitationRenderer
from eurpe.export.errors import (
    ExportAuditError,
    ExportError,
    UnsupportedExportFormatError,
)
from eurpe.export.models import ExportFormat, ExportRequest, ExportResult
from eurpe.export.service import ExportService

__all__ = [
    "DocxCitationRenderer",
    "ExportAuditError",
    "ExportError",
    "ExportFormat",
    "ExportRequest",
    "ExportResult",
    "ExportService",
    "UnsupportedExportFormatError",
]
