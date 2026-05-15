"""Pydantic models for the call/topic intake module.

The :class:`TopicContext` record carries structured Work Programme topic
information into the section-generation prompt. It is the single record
that the API / CLI / future React UI all hand to
:class:`~eurpe.generation.GenerationRequest` so the generator can cite
or reference the supplied topic requirements end-to-end (issue #9 AC #3).

Why a separate record rather than threading individual fields through
``GenerationRequest`` directly?

* Acceptance criterion #2 names a fixed set of fields (programme, topic
  ID / title, expected outcomes, section guidance) that must travel as
  a group; promoting them to a sibling record keeps ``GenerationRequest``
  focussed on per-call knobs.
* The same record is produced from two input paths — pasted text and
  PDF excerpt — so the field-set is the single contract those two
  extractors fulfil. A future intake source (web form, API JSON body)
  drops in by producing a :class:`TopicContext`.
* ``raw_text`` is preserved on the record for audit so an operator can
  inspect what the LLM actually saw without re-running the extractor.

``raw_text`` allows an empty string by design: an empty paste yields a
:class:`TopicContext` with all-empty extracted fields rather than
raising. This is best-effort intake — the extractor never throws when
text is missing, it just produces an empty record (mirrors how the
filename parser returns ``{}`` when it can't parse).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from eurpe.schema import Programme, SectionType


class TopicSource(StrEnum):
    """Where a :class:`TopicContext` originated from.

    ``PASTED_TEXT`` is the user-pastes-into-a-form path; ``PDF_EXCERPT``
    is the upload-the-topic-page-PDF path. The label is preserved on
    the record so the audit trail makes the input mode explicit.
    """

    PASTED_TEXT = "pasted_text"
    PDF_EXCERPT = "pdf_excerpt"


class TopicContext(BaseModel):
    """Structured Work Programme call / topic context.

    Produced by :func:`~eurpe.intake.extractor.extract_topic_context_from_text`
    or :func:`~eurpe.intake.extractor.extract_topic_context_from_pdf`,
    consumed by :class:`~eurpe.generation.GenerationRequest`.

    The record is *best-effort*: any field that the extractor could not
    parse from the source text remains at its default (``None`` or an
    empty collection). Callers should use :meth:`is_empty` to decide
    whether enough was recovered to render the structured prompt block.

    Fields:

    * ``programme`` / ``call_id`` / ``topic_id`` mirror the same fields
      on :class:`~eurpe.schema.ProposalMetadata` so a downstream UI can
      cross-reference indexed proposals by the same identifiers.
    * ``topic_title`` is the human-readable name of the topic, used as a
      heading in the rendered prompt block.
    * ``expected_outcomes`` is a list of bullet strings — each entry is
      one outcome the call expects funded proposals to deliver. This is
      the field that AC #3 hangs off: when present, the prompt explicitly
      instructs the LLM to reference the outcomes in its draft.
    * ``scope`` is free-text describing what the topic covers; rendered
      verbatim in the prompt.
    * ``destination`` records the cluster / destination heading (e.g.,
      "Cluster 3 — Civil Security for Society") when present.
    * ``section_guidance`` carries section-specific guidance keyed by
      :class:`~eurpe.schema.SectionType`. When the request's section
      matches a key, the corresponding text appears under the section
      guidance block in the prompt with a ``Topic requirements for this
      section:`` prefix.
    * ``raw_text`` is the normalised source text the extractor saw —
      preserved for audit so an operator can replay extraction without
      re-uploading the PDF / re-pasting the topic.
    * ``source`` records the input mode (pasted vs. PDF), and
      ``source_path`` is populated only for PDF excerpts.

    ``extra="forbid"`` matches the convention used across
    :mod:`eurpe.schema`, :mod:`eurpe.ingestion`, and
    :mod:`eurpe.generation` — typos in field names fail loudly.
    """

    model_config = ConfigDict(extra="forbid")

    programme: Programme | None = Field(
        default=None,
        description=(
            "EU programme the call belongs to, parsed from the topic text "
            "(e.g., ``HORIZON-CL3-...`` → :class:`Programme.HORIZON_EUROPE`)."
        ),
    )
    call_id: str | None = Field(
        default=None,
        description=(
            "Call identifier (e.g., ``HORIZON-CL3-2024-CS-01``) parsed from "
            "the topic text. Surfaced verbatim in the prompt."
        ),
    )
    topic_id: str | None = Field(
        default=None,
        description=(
            "Six- or seven-digit Funding & Tenders topic number (e.g., "
            "``952672``) parsed from the topic text."
        ),
    )
    topic_title: str | None = Field(
        default=None,
        description=(
            "Human-readable topic title (e.g., ``Resilient digital "
            "infrastructure for critical sectors``), parsed from a "
            "``Topic:`` or ``Topic title:`` line in the source text."
        ),
    )
    expected_outcomes: list[str] = Field(
        default_factory=list,
        description=(
            "Bullets captured from the ``Expected Outcomes`` heading "
            "block. When non-empty, the prompt's output instructions "
            "explicitly direct the LLM to reference these outcomes "
            "(AC #3 of issue #9)."
        ),
    )
    scope: str | None = Field(
        default=None,
        description=(
            "Free-text scope captured from the ``Scope`` heading block. "
            "Rendered verbatim in the prompt."
        ),
    )
    destination: str | None = Field(
        default=None,
        description=(
            "Cluster / destination label (e.g., ``Cluster 3 — Civil "
            "Security for Society``) parsed from a ``Destination:`` or "
            "``Cluster:`` line."
        ),
    )
    section_guidance: dict[SectionType, str] = Field(
        default_factory=dict,
        description=(
            "Section-type-keyed guidance blocks captured from headings "
            "like ``Methodology guidance:`` / ``Impact guidance:``. "
            "When the request's section_type matches a key, the entry "
            "appears under the prompt's section-guidance block."
        ),
    )
    raw_text: str = Field(
        default="",
        description=(
            "Normalised source text the extractor saw. Preserved for "
            "audit; not rendered in the prompt directly."
        ),
    )
    source: TopicSource = Field(
        description=(
            "Where this context came from — pasted text or PDF excerpt. "
            "Mirrored in the audit trail."
        ),
    )
    source_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to the source PDF, populated only when "
            "``source == TopicSource.PDF_EXCERPT``."
        ),
    )

    def is_empty(self) -> bool:
        """True iff no usable topic information was extracted.

        Returns True when topic_id, topic_title, and expected_outcomes
        are all blank. A caller can use this to decide whether to skip
        rendering the structured prompt block and fall back to the
        free-text ``call_context`` path.
        """

        return (
            not self.topic_id
            and not self.topic_title
            and not self.expected_outcomes
        )
