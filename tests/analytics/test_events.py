"""Tests for :mod:`eurpe.analytics.events` — the schema layer.

These tests pin AC1 (event schema exists for draft start/complete,
feedback, export, generation time, source-status mix) and the AC2
foundation: the forbidden-field set (``user_intent``, ``prompt``,
``text``, ``snippet``, ``passage``, ``body``, ``payload``, ``content``)
cannot land on any event because every model uses
``extra="forbid"``.

The frozen-model regression is verified explicitly so a future
``model_config = ConfigDict(frozen=False)`` typo would surface.

The exact-field-set assertions mirror
``test_audit_log.py::test_log_attempt_writes_exact_field_set``: any
new field on these events forces a review of this test and, by
extension, of the AC2 privacy contract.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from eurpe.analytics.events import (
    BaseAnalyticsEvent,
    DraftCompletedEvent,
    DraftStartedEvent,
    EventType,
    ExportEvent,
    FeedbackEvent,
)

# Forbidden field names. Constructing ANY event with one of these as a
# kwarg must raise pydantic's ValidationError (because every event
# model has ``extra="forbid"``). The list is identical for every event
# subclass so we parametrise over the union.
_FORBIDDEN_FIELD_NAMES = [
    "user_intent",
    "prompt",
    "text",
    "snippet",
    "passage",
    "body",
    "payload",
    "content",
]


def _draft_started_kwargs() -> dict:
    return {
        "event_type": EventType.DRAFT_STARTED,
        "section_type": "methodology",
        "target_programme": "horizon_europe",
        "top_k_examples": 5,
        "lessons_learned": False,
        "model": "deterministic-stub-v1",
        "drafting_profile": None,
        "topic_context_present": False,
    }


def _draft_completed_kwargs() -> dict:
    return {
        "event_type": EventType.DRAFT_COMPLETED,
        "section_type": "methodology",
        "generation_time_ms": 1234,
        "citation_count": 3,
        "source_status_mix": {"funded": 2, "rejected": 1},
        "model": "deterministic-stub-v1",
        "drafting_profile": None,
    }


def _export_kwargs() -> dict:
    return {
        "event_type": EventType.EXPORT,
        "kind": "markdown",
        "byte_count": 4096,
        "section_type": "methodology",
    }


def _feedback_kwargs() -> dict:
    return {
        "event_type": EventType.FEEDBACK,
        "feedback_type": "thumbs_up",
        "accepted": True,
        "section_type": "methodology",
        "iteration_index": 0,
    }


# ---------------------------------------------------------------------------
# AC2 — forbidden field battery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,base_kwargs",
    [
        (DraftStartedEvent, _draft_started_kwargs()),
        (DraftCompletedEvent, _draft_completed_kwargs()),
        (ExportEvent, _export_kwargs()),
        (FeedbackEvent, _feedback_kwargs()),
    ],
)
@pytest.mark.parametrize("forbidden_name", _FORBIDDEN_FIELD_NAMES)
def test_event_rejects_forbidden_field_names(
    build: type[BaseAnalyticsEvent],
    base_kwargs: dict,
    forbidden_name: str,
) -> None:
    """Every event subclass refuses content-bearing field names.

    The closed-vocabulary contract (issue #13 AC2) says no event field
    may carry proposal content. The schema enforces it via
    ``extra="forbid"`` — this test pins the enforcement on every
    forbidden name across every event type.
    """

    bad_kwargs = {**base_kwargs, forbidden_name: "some proposal content"}
    with pytest.raises(ValidationError):
        build(**bad_kwargs)


# ---------------------------------------------------------------------------
# AC1 — frozen=True regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,kwargs",
    [
        (DraftStartedEvent, _draft_started_kwargs()),
        (DraftCompletedEvent, _draft_completed_kwargs()),
        (ExportEvent, _export_kwargs()),
        (FeedbackEvent, _feedback_kwargs()),
    ],
)
def test_event_is_frozen(build: type[BaseAnalyticsEvent], kwargs: dict) -> None:
    """Assigning to a field on a constructed event raises ValidationError.

    Pydantic v2 surfaces frozen-model violations as
    :class:`pydantic.ValidationError`. Switching to a non-frozen
    model_config would let post-construction mutation smuggle
    content; this test prevents that regression.
    """

    event = build(**kwargs)
    with pytest.raises(ValidationError):
        event.section_type = "MUTATED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC1 — JSON shape pinning, one test per event type
# ---------------------------------------------------------------------------


def test_draft_started_json_field_set() -> None:
    """Exact field set for :class:`DraftStartedEvent`.

    Adding a field to this event surfaces here — a reviewer can then
    confirm the field is not a content-bearing leak under the AC2
    contract.
    """

    event = DraftStartedEvent(**_draft_started_kwargs())
    record = event.model_dump(mode="json")
    parsed = json.loads(json.dumps(record))
    assert set(parsed.keys()) == {
        "event_type",
        "timestamp",
        "section_type",
        "target_programme",
        "top_k_examples",
        "lessons_learned",
        "model",
        "drafting_profile",
        "topic_context_present",
    }
    assert parsed["event_type"] == "draft_started"
    assert parsed["section_type"] == "methodology"
    assert parsed["top_k_examples"] == 5


def test_draft_completed_json_field_set() -> None:
    """Exact field set for :class:`DraftCompletedEvent`."""

    event = DraftCompletedEvent(**_draft_completed_kwargs())
    record = event.model_dump(mode="json")
    parsed = json.loads(json.dumps(record))
    assert set(parsed.keys()) == {
        "event_type",
        "timestamp",
        "section_type",
        "generation_time_ms",
        "citation_count",
        "source_status_mix",
        "model",
        "drafting_profile",
        "iteration_count",
    }
    assert parsed["event_type"] == "draft_completed"
    assert parsed["generation_time_ms"] == 1234
    assert parsed["source_status_mix"] == {"funded": 2, "rejected": 1}
    # Sprint 1 default — Sprint 3 critic loop will bump this.
    assert parsed["iteration_count"] == 1


def test_export_json_field_set() -> None:
    """Exact field set for :class:`ExportEvent`."""

    event = ExportEvent(**_export_kwargs())
    record = event.model_dump(mode="json")
    parsed = json.loads(json.dumps(record))
    assert set(parsed.keys()) == {
        "event_type",
        "timestamp",
        "kind",
        "byte_count",
        "section_type",
    }
    assert parsed["event_type"] == "export"
    assert parsed["kind"] == "markdown"
    assert parsed["byte_count"] == 4096


def test_feedback_json_field_set() -> None:
    """Exact field set for :class:`FeedbackEvent`.

    Schema-only in this issue (no emitter), so the test pins the
    shape that future emitters must respect.
    """

    event = FeedbackEvent(**_feedback_kwargs())
    record = event.model_dump(mode="json")
    parsed = json.loads(json.dumps(record))
    assert set(parsed.keys()) == {
        "event_type",
        "timestamp",
        "feedback_type",
        "accepted",
        "section_type",
        "iteration_index",
    }
    assert parsed["event_type"] == "feedback"
    assert parsed["accepted"] is True


# ---------------------------------------------------------------------------
# AC1 — generation_time_ms and citation_count validators
# ---------------------------------------------------------------------------


def test_draft_completed_rejects_negative_generation_time() -> None:
    """``generation_time_ms`` must be non-negative (ge=0)."""

    bad = {**_draft_completed_kwargs(), "generation_time_ms": -1}
    with pytest.raises(ValidationError):
        DraftCompletedEvent(**bad)


def test_draft_completed_rejects_negative_citation_count() -> None:
    """``citation_count`` must be non-negative (ge=0)."""

    bad = {**_draft_completed_kwargs(), "citation_count": -1}
    with pytest.raises(ValidationError):
        DraftCompletedEvent(**bad)


def test_export_rejects_negative_byte_count() -> None:
    """``byte_count`` must be non-negative (ge=0)."""

    bad = {**_export_kwargs(), "byte_count": -1}
    with pytest.raises(ValidationError):
        ExportEvent(**bad)


def test_draft_started_top_k_bounds() -> None:
    """``top_k_examples`` is bounded 1..20, mirroring the GenerationRequest field."""

    bad_low = {**_draft_started_kwargs(), "top_k_examples": 0}
    with pytest.raises(ValidationError):
        DraftStartedEvent(**bad_low)

    bad_high = {**_draft_started_kwargs(), "top_k_examples": 21}
    with pytest.raises(ValidationError):
        DraftStartedEvent(**bad_high)


# ---------------------------------------------------------------------------
# Discriminator pinning — event_type is the class's own value
# ---------------------------------------------------------------------------


def test_draft_started_event_type_is_pinned() -> None:
    """``event_type`` defaults to DRAFT_STARTED for :class:`DraftStartedEvent`."""

    kwargs = _draft_started_kwargs()
    del kwargs["event_type"]
    event = DraftStartedEvent(**kwargs)
    assert event.event_type is EventType.DRAFT_STARTED


def test_draft_started_rejects_mismatched_event_type() -> None:
    """Passing a wrong ``event_type`` to a subclass surfaces a validation error.

    The ``Literal[EventType.X]`` annotation on each subclass means
    only the matching discriminator value is acceptable — a
    misordered constructor cannot land an event under the wrong
    label.
    """

    bad = {**_draft_started_kwargs(), "event_type": EventType.EXPORT}
    with pytest.raises(ValidationError):
        DraftStartedEvent(**bad)


# ---------------------------------------------------------------------------
# Timestamp default — UTC, datetime
# ---------------------------------------------------------------------------


def test_timestamp_default_is_utc_datetime() -> None:
    """The default factory yields a timezone-aware UTC datetime."""

    from datetime import datetime

    event = DraftStartedEvent(**_draft_started_kwargs())
    assert isinstance(event.timestamp, datetime)
    assert event.timestamp.tzinfo is not None
    # JSON form is ISO-8601 with offset; sanity check it serialises.
    record = event.model_dump(mode="json")
    assert isinstance(record["timestamp"], str)


# ---------------------------------------------------------------------------
# AC2 — content-bearing fields are NOT in the schema (positive check)
# ---------------------------------------------------------------------------


def test_no_event_field_matches_forbidden_vocabulary() -> None:
    """Belt-and-braces: enumerate model fields and assert none match the forbidden vocabulary.

    The ``extra="forbid"`` enforcement is the primary guard; this is
    a structural check that the schema author did not name a field
    after one of the forbidden tokens by accident.
    """

    for event_cls in (
        DraftStartedEvent,
        DraftCompletedEvent,
        ExportEvent,
        FeedbackEvent,
    ):
        field_names = set(event_cls.model_fields.keys())
        for forbidden in _FORBIDDEN_FIELD_NAMES:
            assert forbidden not in field_names, (
                f"{event_cls.__name__} declares forbidden field {forbidden!r}; "
                "the AC2 privacy contract forbids it."
            )
