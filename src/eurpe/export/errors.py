"""Exceptions raised by the export service.

A separate module so callers can ``from eurpe.export.errors import ...``
without importing the renderer or service implementations.
"""

from __future__ import annotations


class ExportError(Exception):
    """Base class for every export failure.

    Catch this when you want to surface any export problem (unsupported
    format, audit refusal, etc.) without depending on a specific subtype.
    """


class UnsupportedExportFormatError(ExportError):
    """Raised when the requested :class:`ExportFormat` has no renderer wired up.

    DOCX is on the roadmap (Task 3.3) but ships behind this error in the
    MVP. The enum slot exists so call sites and the wire schema don't
    change when DOCX lands.
    """


class ExportAuditError(ExportError):
    """Raised when the citation audit refuses a draft.

    PRD §22 makes source-status labels release-blocking. The export
    service runs :class:`eurpe.generation.CitationAudit` on the rendered
    output and raises this rather than handing the caller a draft that
    silently dropped a status label.
    """

    def __init__(self, message: str, *, finding_count: int) -> None:
        super().__init__(message)
        self.finding_count = finding_count
