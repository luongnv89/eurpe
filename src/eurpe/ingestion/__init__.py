"""Ingestion package for EURPE.

Hosts Docling-based PDF parsing for EU research proposals. The public
surface is intentionally narrow:

* :class:`DoclingProposalParser` — turns a PDF into a typed
  :class:`ParsedProposal`.
* Three parser-output models: :class:`ParsedProposal`, :class:`ParsedSection`,
  :class:`ParsedTable`.
* Three exception types: :class:`IngestionError` (base),
  :class:`ParserError` (parser ran but failed), and
  :class:`UnsupportedFormatError` (file type cannot be parsed).

Hierarchical chunking and the join with :class:`eurpe.schema.ChunkMetadata`
land in issue #4. This module deliberately stops at the structural-output
boundary so that change can land without touching the parser layer.

Importing this package is intentionally cheap: Docling itself is imported
lazily inside :meth:`DoclingProposalParser.parse` (see that module's
docstring for the rationale).
"""

from __future__ import annotations

from eurpe.ingestion.docling_parser import DoclingProposalParser
from eurpe.ingestion.errors import (
    IngestionError,
    ParserError,
    UnsupportedFormatError,
)
from eurpe.ingestion.models import ParsedProposal, ParsedSection, ParsedTable

# NOTE: IngestionService lives in eurpe.ingestion.service and is NOT
# re-exported here. The service imports eurpe.retrieval, which imports
# eurpe.ingestion.models — re-exporting at this level would create a
# circular import. Callers should use
# ``from eurpe.ingestion.service import IngestionService`` or the
# top-level ``from eurpe import IngestionService`` which sequences the
# imports correctly.

__all__ = [
    "DoclingProposalParser",
    "IngestionError",
    "ParsedProposal",
    "ParsedSection",
    "ParsedTable",
    "ParserError",
    "UnsupportedFormatError",
]
