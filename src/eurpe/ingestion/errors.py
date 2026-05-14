"""Exceptions raised by the EURPE ingestion pipeline.

Three layers, base → specific:

* :class:`IngestionError` — anything that goes wrong inside the ingestion
  package. Callers (CLI, future orchestration) catch this to convert pipeline
  failures into user-facing errors without leaking parser internals.
* :class:`UnsupportedFormatError` — the file extension or MIME type is not
  handled by any registered parser. Distinct from ``ParserError`` so the CLI
  can suggest a fix ("install a parser plugin", "convert to PDF first") rather
  than a vague "parsing failed".
* :class:`ParserError` — a parser was selected and ran, but produced unusable
  output. Always carries the source path and the original exception (if any)
  so the CLI can print a clear summary while the caller still has full access
  to the underlying traceback for debugging.

The acceptance criterion that drives this hierarchy is "parser failures
produce clear errors without corrupting existing indexed data": every parser
in this package MUST raise one of these exceptions on failure rather than
returning a half-built object, and the caller MUST NOT write any partial
state to disk before a successful return.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for any ingestion-pipeline failure.

    Catching ``IngestionError`` is the recommended way for callers (CLI,
    orchestration code) to handle ingestion problems uniformly. Use a more
    specific subclass when raising so callers that want to react differently
    (e.g., suggest converting a .doc to .pdf for an ``UnsupportedFormatError``)
    can do so.
    """


class UnsupportedFormatError(IngestionError):
    """The file extension or MIME type is not handled by any parser.

    Raised by :meth:`DoclingProposalParser.parse` and similar entry points
    *before* any heavy work is attempted, so callers can fail fast on a wrong
    file type without paying the cost of loading a model.
    """


class ParserError(IngestionError):
    """A parser ran but produced unusable output.

    Always carries the absolute source path and (when available) the
    underlying exception. The string form embeds the path for log/CLI
    readability; the structured ``source_path`` and ``cause`` attributes
    are kept available for programmatic handling and re-raising with full
    chaining.
    """

    def __init__(
        self,
        source_path: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(f"Failed to parse {source_path}: {message}")
        # Stored as plain attributes (not @property) so callers can pattern-match
        # without reaching into the parent ``args`` tuple. The base ``Exception``
        # already keeps the formatted message in ``self.args[0]``.
        self.source_path = source_path
        self.cause = cause
