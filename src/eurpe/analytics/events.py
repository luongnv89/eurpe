"""Pydantic event schemas for the local analytics package.

Every event is a tiny, frozen, content-free record describing one
operationally interesting moment in the drafting workflow:

* :class:`DraftStartedEvent` — a section drafting call has begun.
* :class:`DraftCompletedEvent` — that call returned a draft.
* :class:`ExportEvent` — the user asked the CLI to write rendered
  artefacts to disk.
* :class:`FeedbackEvent` — schema for a future thumbs-up / regenerate
  signal. Schema-only in this issue; no emitter wired up yet.

The privacy contract (issue #13 AC2) is the reason this module exists.
**No field on any event may carry proposal content, prompts, retrieved
passages, generated draft text, citation snippets, user intent text,
partner-confidential data, or anything else that could be reconstructed
into proposal content.** Field names commonly used to smuggle such
content — ``user_intent``, ``prompt``, ``text``, ``snippet``,
``passage``, ``body``, ``payload``, ``content`` — are explicitly
forbidden by the ``extra="forbid"`` model_config and pinned by the
``tests/analytics/test_events.py`` regression battery. Adding a new
field here MUST be reviewed against that contract.

All events inherit a small :class:`BaseAnalyticsEvent` that fixes the
``event_type`` discriminator + UTC ``timestamp`` so the JSONL log can
be sliced by either field without per-event branching. The discriminator
is materialised on each subclass as a ``Literal[EventType.X]`` so the
class is the source of truth and a misordered ``EventType`` argument
at the call site cannot land an event under the wrong label.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """Closed vocabulary of analytics event labels.

    String-valued so the variant name appears verbatim in the JSONL
    log without a custom serialiser. Adding a new variant means
    a new event subclass + new column in any downstream visualisation
    — both are deliberate.
    """

    DRAFT_STARTED = "draft_started"
    DRAFT_COMPLETED = "draft_completed"
    EXPORT = "export"
    FEEDBACK = "feedback"


class BaseAnalyticsEvent(BaseModel):
    """Shared fields for every analytics event.

    ``extra="forbid"`` is what stops a future emitter from silently
    smuggling proposal content under a ``text`` or ``snippet`` kwarg —
    Pydantic refuses unknown fields at construction. ``frozen=True``
    means an event, once built, cannot be mutated to add content
    after the audit-friendly fields were filled in.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: EventType = Field(
        description="Discriminator naming which event subclass this is.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the event was constructed.",
    )


class DraftStartedEvent(BaseAnalyticsEvent):
    """Recorded when :meth:`SectionGenerationWorkflow.run` begins.

    Carries only the operational knobs the workflow was given. No
    field on this event accepts the user intent text, the call/topic
    context, or any retrieved passage — those would defeat AC2 of
    issue #13.
    """

    event_type: Literal[EventType.DRAFT_STARTED] = EventType.DRAFT_STARTED
    section_type: str = Field(
        description="``SectionType`` value the workflow was asked to draft.",
    )
    target_programme: str | None = Field(
        default=None,
        description="Programme filter applied to retrieval, or ``None`` for all.",
    )
    top_k_examples: int = Field(
        ge=1,
        le=20,
        description="Number of past-proposal examples requested from the retriever.",
    )
    lessons_learned: bool = Field(
        description="Whether lessons-learned mode was on for this call.",
    )
    model: str = Field(
        min_length=1,
        description="LLM identifier (e.g., ``llama3.1:8b``, ``deterministic-stub-v1``).",
    )
    drafting_profile: str | None = Field(
        default=None,
        description="Drafting profile name (e.g., ``Horizon Europe Standard``).",
    )
    topic_context_present: bool = Field(
        description="Whether a structured TopicContext was attached to the request.",
    )


class DraftCompletedEvent(BaseAnalyticsEvent):
    """Recorded when the workflow returns a valid draft.

    ``source_status_mix`` is a Counter-shaped dict over the
    ``SourceStatus`` values of the citations attached to the draft —
    enough to plot "what mix of funded vs. rejected vs. ESR examples
    is the model drawing on" without storing any source text.
    ``iteration_count`` is forward-looking: Sprint 3's critic loop
    (issue #16) will increment it; for the Sprint 1 single-pass
    workflow it stays at 1.
    """

    event_type: Literal[EventType.DRAFT_COMPLETED] = EventType.DRAFT_COMPLETED
    section_type: str = Field(
        description="``SectionType`` value of the drafted section.",
    )
    generation_time_ms: int = Field(
        ge=0,
        description="Wall-clock time the run took, in milliseconds.",
    )
    citation_count: int = Field(
        ge=0,
        description="Number of citations attached to the produced draft.",
    )
    source_status_mix: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Counter over the ``SourceStatus`` values of the citations. "
            "Keys are the plain string values (``funded``, ``rejected``, "
            "``esr_note``, ``unknown``). Empty dict when no citations were "
            "produced."
        ),
    )
    model: str = Field(
        min_length=1,
        description="LLM identifier that produced the draft.",
    )
    drafting_profile: str | None = Field(
        default=None,
        description="Drafting profile name, when one was applied.",
    )
    iteration_count: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of draft iterations (critic-loop passes). Forward-looking "
            "for Sprint 3 (issue #16); stays at 1 for the Sprint 1 single-pass "
            "workflow."
        ),
    )


class ExportEvent(BaseAnalyticsEvent):
    """Recorded when the CLI writes a rendered artefact to disk.

    Records *what kind* of artefact was written and *how large* the
    payload was, NOT the payload itself. Used to track "how often do
    people export to Markdown vs. JSON" and "how big are the drafts
    operators typically work with" — neither needs a single byte of
    the content.
    """

    event_type: Literal[EventType.EXPORT] = EventType.EXPORT
    kind: str = Field(
        min_length=1,
        description="Artefact label (e.g., ``markdown``, ``json``).",
    )
    byte_count: int = Field(
        ge=0,
        description="Size of the written artefact in UTF-8 bytes.",
    )
    section_type: str | None = Field(
        default=None,
        description="``SectionType`` value of the drafted section, when applicable.",
    )


class FeedbackEvent(BaseAnalyticsEvent):
    """Recorded when the user signals their feeling about a draft.

    Schema-only in this issue — no emitter is wired up yet. Future
    UI surfaces (thumbs up / down, regenerate) will populate this.
    The forbidden-field set is identical to every other event in this
    module — no comment text, no rationale, no draft body.
    """

    event_type: Literal[EventType.FEEDBACK] = EventType.FEEDBACK
    feedback_type: str = Field(
        min_length=1,
        description=(
            "Label for the feedback signal (e.g., ``thumbs_up``, ``thumbs_down``, "
            "``regenerate``). Closed vocabulary owned by the UI surface."
        ),
    )
    accepted: bool = Field(
        description="Whether the user accepted the draft as-is.",
    )
    section_type: str | None = Field(
        default=None,
        description="``SectionType`` value of the drafted section, when known.",
    )
    iteration_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Index of the draft iteration the feedback applies to. ``None`` "
            "for single-pass workflows; populated by the Sprint 3 critic loop."
        ),
    )
