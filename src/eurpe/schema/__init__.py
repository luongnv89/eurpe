"""Public schema package for EURPE proposal and chunk metadata.

Re-exports the enums and Pydantic models so callers can write::

    from eurpe.schema import ChunkMetadata, ProposalMetadata, SourceStatus

without having to know the internal module layout.
"""

from __future__ import annotations

from eurpe.schema.enums import Programme, SectionType, SourceStatus
from eurpe.schema.metadata import ChunkMetadata, CitationAnchor, ProposalMetadata

__all__ = [
    "ChunkMetadata",
    "CitationAnchor",
    "Programme",
    "ProposalMetadata",
    "SectionType",
    "SourceStatus",
]
