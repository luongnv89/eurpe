"""Intermediate Pydantic models for the ingestion pipeline.

These models capture the *raw structural output* produced by a parser
(currently :class:`~eurpe.ingestion.docling_parser.DoclingProposalParser`)
**before** chunking, embedding, or production of the retrieval-facing
:class:`eurpe.schema.ChunkMetadata` records. Chunking lands in Issue #4 and
will consume :class:`ParsedProposal` objects.

Why a separate model layer instead of building :class:`ChunkMetadata`
directly?

* Parser output is structural (sections, tables, page ranges) and does not
  yet know about ``ProposalMetadata`` (programme, call_id, outcome) — that
  comes from the YAML sidecar provided by the caller and is joined in
  later.
* A flat ``ParsedSection`` list (not a nested tree) is enough for the
  prototype's section-level acceptance criterion, and lets us defer
  hierarchical chunking decisions to Issue #4 without committing to a
  specific tree representation here.
* Keeping the parser output strictly-validated (``extra="forbid"``) catches
  Docling-API drift at parse time rather than at retrieval time.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Mirrors the helper in :mod:`eurpe.schema.metadata` so both layers use the
    same factory (avoids the deprecated ``datetime.utcnow`` and keeps the
    ``parsed_at`` timestamp directly comparable to ``ProposalMetadata``'s
    ``ingested_at``).
    """

    return datetime.now(UTC)


class ParsedTable(BaseModel):
    """A table extracted from a proposal section.

    Stored as a 2D list of cell text. Rows MAY be ragged (different lengths)
    because EU proposals frequently use merged cells which Docling represents
    by repeating or truncating cells per row. Downstream consumers MUST cope
    with ragged grids rather than assuming a fixed column count.
    """

    model_config = ConfigDict(extra="forbid")

    section_heading: str | None = Field(
        default=None,
        description="Heading of the section the table appeared in, if known.",
    )
    rows: list[list[str]] = Field(
        default_factory=list,
        description="Plain-text cell values, row-major. May be ragged.",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed page number where the table starts, if known.",
    )


class ParsedSection(BaseModel):
    """A single section extracted from a proposal.

    ``level`` records the original heading depth (1 = top-level, 6 =
    deepest) so a future tree-builder can reconstruct the hierarchy. The
    prototype keeps the list flat and lets chunking (Issue #4) decide
    whether to nest.

    Note: char offsets (start/end byte positions inside the original
    document) are intentionally not modelled here. They will be added
    when chunking lands (Issue #4) so :class:`CitationAnchor` records can
    point at exact text spans rather than only sections + pages.
    """

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(
        min_length=1,
        description="Verbatim section heading, e.g., '1.1 Excellence'.",
    )
    level: int = Field(
        ge=1,
        le=6,
        description="Heading depth: 1 = H1, 6 = H6.",
    )
    text: str = Field(
        default="",
        description="Plain-text body of the section (excludes nested headings).",
    )
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed first page covered by this section.",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed last page covered by this section.",
    )
    tables: list[ParsedTable] = Field(
        default_factory=list,
        description="Tables that appeared inside this section's text range.",
    )


class ParsedProposal(BaseModel):
    """A successfully-parsed proposal document.

    Holds the raw structural output of the Docling parser before chunking
    and embedding. The ``parser`` tag is recorded for observability so a
    fallback parser (e.g., a plain-PDF text extractor) can be distinguished
    from Docling output later in pipeline metrics.
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        min_length=1,
        description="Absolute path to the PDF that was parsed.",
    )
    title: str | None = Field(
        default=None,
        description="Document title — taken from Docling metadata or the first H1.",
    )
    sections: list[ParsedSection] = Field(
        default_factory=list,
        description="Flat, ordered list of sections; depth recorded in each section.",
    )
    page_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of pages in the source PDF, if reported by the parser.",
    )
    parser: str = Field(
        default="docling",
        description="Identifier of the parser that produced this output.",
    )
    parsed_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when parsing completed.",
    )

    def total_text_length(self) -> int:
        """Return the sum of ``len(section.text)`` across all sections.

        Cheap diagnostic used by the CLI summary and useful in tests to
        confirm the parser actually extracted content (a value of 0 almost
        always means the parser walked the document but failed to recover
        body text — a useful early-warning signal).
        """

        return sum(len(s.text) for s in self.sections)
