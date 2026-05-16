"""EURPE — fully-local AI assistant for drafting EU research proposals.

The four MVP service boundaries (Task 2.7 / issue #14) are re-exported
here so the UI and other consumers import one stable seam:

* :class:`IngestionService` — parse + duplicate-check + chunk + upsert.
* :class:`RetrievalService` — source-status-aware retrieval queries.
* :class:`GenerationService` — section-draft generation workflow.
* :class:`ExportService` — render a draft to Markdown (DOCX reserved
  for Task 3.3).

Each service is also importable from its home package
(``eurpe.ingestion``, ``eurpe.retrieval``, ``eurpe.generation``,
``eurpe.export``); the top-level re-export is purely an ergonomics
choice so React / FastAPI / CLI code can read ``from eurpe import
GenerationService``.
"""

from __future__ import annotations

from eurpe.export import (
    ExportError,
    ExportFormat,
    ExportRequest,
    ExportResult,
    ExportService,
)
from eurpe.generation import (
    GenerationService,
    SectionGenerationRequest,
)
from eurpe.ingestion.service import (
    DuplicateRefusedError,
    IngestionRequest,
    IngestionResult,
    IngestionService,
)
from eurpe.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalService,
)

__version__ = "0.1.0"

__all__ = [
    "DuplicateRefusedError",
    "ExportError",
    "ExportFormat",
    "ExportRequest",
    "ExportResult",
    "ExportService",
    "GenerationService",
    "IngestionRequest",
    "IngestionResult",
    "IngestionService",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievalService",
    "SectionGenerationRequest",
]
