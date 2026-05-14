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

__all__ = [
    "DoclingProposalParser",
    "IngestionError",
    "ParsedProposal",
    "ParsedSection",
    "ParsedTable",
    "ParserError",
    "UnsupportedFormatError",
]
