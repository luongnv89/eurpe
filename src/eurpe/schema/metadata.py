"""Pydantic v2 models for EURPE proposal, chunk, and citation metadata.

Three composable models live here:

* :class:`CitationAnchor` — *where* in a source document a chunk came from
  (page, section heading, character offsets). Frozen so anchors are hashable
  and can be used as set / dict keys.
* :class:`ProposalMetadata` — identifies the *source document* (programme,
  call, year, outcome, on-disk path).
* :class:`ChunkMetadata` — per-chunk metadata used by the retriever and the
  citation renderer; composes a :class:`ProposalMetadata` plus section info
  and a :class:`CitationAnchor`.

Validation rules guard the two invariants that matter most for EURPE:

1. Every proposal carries an explicit :class:`SourceStatus` outcome and an
   explicit :class:`Programme`. Records missing either field are rejected at
   construction time.
2. A chunk's ``source_status`` MUST equal its parent ``proposal.outcome``,
   so the retrieval-facing label can never silently drift from the source
   proposal's ground truth.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eurpe.schema.enums import Programme, SectionType, SourceStatus


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Wrapped in a helper so :class:`ProposalMetadata` can use it as a
    ``default_factory`` without depending on the deprecated
    ``datetime.utcnow`` (deprecated in Python 3.12+).
    """

    return datetime.now(UTC)


class CitationAnchor(BaseModel):
    """Where in a source document a chunk's content originated.

    Frozen and hashable so the citation renderer can deduplicate anchors via
    ``set()``. Page numbers are 1-indexed when present; ``None`` means the
    source format does not expose pagination (e.g., a DOCX without explicit
    page breaks). Character offsets refer to the extracted plain text and are
    optional for the same reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1, description="Stable id of the source document.")
    section_heading: str | None = Field(
        default=None, description="Verbatim section header text, if any."
    )
    page: int | None = Field(default=None, ge=1, description="1-indexed page number, if known.")
    char_start: int | None = Field(
        default=None, ge=0, description="Inclusive start offset in the source plain text."
    )
    char_end: int | None = Field(
        default=None, ge=0, description="Exclusive end offset in the source plain text."
    )

    @model_validator(mode="after")
    def _offsets_well_ordered(self) -> CitationAnchor:
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )
        return self


class ProposalMetadata(BaseModel):
    """Identifies the source proposal a chunk came from.

    ``outcome`` is the source-status label of the *proposal* itself; ESR
    reviewer notes carry :attr:`SourceStatus.ESR_NOTE`. ``source_path`` stays
    a string (not :class:`pathlib.Path`) so the model serializes cleanly to
    YAML / JSON without custom encoders.
    """

    # ``validate_assignment=True`` re-runs validators on every attribute set,
    # so post-construction mutation of ``outcome`` (or any other field) cannot
    # silently violate invariants. Combined with the matching flag on
    # :class:`ChunkMetadata`, this catches the realistic drift path where a
    # caller mutates the proposal object directly.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    programme: Programme = Field(description="EU funding programme.")
    call_id: str = Field(
        min_length=1, description="Identifier of the call, e.g., HORIZON-CL5-2024-D3-02."
    )
    topic_id: str | None = Field(
        default=None,
        description="Topic id within the call, or None for whole-call documents.",
    )
    year: int = Field(description="Call year.")
    outcome: SourceStatus = Field(description="Source-status of the proposal as a whole.")
    proposal_title: str | None = Field(default=None)
    consortium_acronym: str | None = Field(default=None)
    source_path: str = Field(min_length=1, description="On-disk path to the source document.")
    language: str = Field(default="en", description="ISO 639-1 language code.")
    ingested_at: datetime | None = Field(
        default_factory=_utc_now,
        description="When this metadata record was created (UTC).",
    )

    @field_validator("call_id")
    @classmethod
    def _call_id_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("call_id must be non-empty after stripping whitespace")
        return stripped

    @field_validator("source_path", mode="after")
    @classmethod
    def _source_path_non_empty(cls, value: str) -> str:
        # ``min_length=1`` only checks the raw character count, so a string of
        # whitespace would slip past it. Strip and re-check so a path that is
        # only spaces / tabs is rejected the same way an empty string is.
        stripped = value.strip()
        if not stripped:
            raise ValueError("source_path must be non-empty after stripping whitespace")
        return stripped

    @field_validator("year")
    @classmethod
    def _year_in_range(cls, value: int) -> int:
        if value < 2014 or value > 2099:
            raise ValueError(f"year must be in [2014, 2099], got {value}")
        return value

    @model_validator(mode="after")
    def _esr_requires_programme(self) -> ProposalMetadata:
        # Defensive: Pydantic already enforces ``programme`` because it is
        # required, so this branch is normally unreachable. Kept to make the
        # invariant explicit and to surface a clearer error if a future
        # refactor ever loosens the field.
        if self.outcome is SourceStatus.ESR_NOTE and self.programme is None:  # pragma: no cover
            raise ValueError("ESR notes must still record their parent programme")
        return self


class ChunkMetadata(BaseModel):
    """Per-chunk metadata threaded through retrieval and citation rendering.

    The ``source_status`` field is what the retriever filters and ranks on,
    and what the exporter prints in citation footers. It is duplicated from
    :attr:`ProposalMetadata.outcome` deliberately so a chunk can be passed
    around without its parent proposal record always being in scope; the
    :meth:`_status_matches_proposal` validator guarantees the two values can
    never disagree.
    """

    # ``validate_assignment=True`` is what closes the post-construction
    # drift hole: setting ``chunk.source_status = SourceStatus.REJECTED``
    # after the fact re-runs ``_status_matches_proposal`` and raises.
    # ``frozen=True`` is intentionally NOT used because legitimate ingestion
    # paths still need to decorate a chunk after construction (for example,
    # the chunker assigning ``chunk_index`` once it knows the position).
    #
    # Known Pydantic limitation: assignment validators do NOT propagate
    # through nested models, so mutating ``chunk.proposal.outcome`` will
    # re-fire ``ProposalMetadata`` validators (which is why the same flag is
    # set there) but will NOT re-fire ``ChunkMetadata._status_matches_proposal``.
    # The realistic drift path is via the outer model, which is covered.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    proposal: ProposalMetadata = Field(description="Source proposal this chunk belongs to.")
    section_type: SectionType = Field(default=SectionType.OTHER)
    parent_section_heading: str | None = Field(default=None)
    chunk_index: int = Field(ge=0, description="0-indexed position of the chunk in the document.")
    anchor: CitationAnchor = Field(description="Where in the source the chunk came from.")
    source_status: SourceStatus = Field(
        description="Retrieval-facing source status; must equal proposal.outcome."
    )

    @model_validator(mode="after")
    def _status_matches_proposal(self) -> ChunkMetadata:
        if self.source_status is not self.proposal.outcome:
            raise ValueError(
                "source_status drift detected: "
                f"chunk.source_status={self.source_status.value} "
                f"!= proposal.outcome={self.proposal.outcome.value}. "
                "A chunk's source_status must equal its parent proposal's outcome."
            )
        return self
