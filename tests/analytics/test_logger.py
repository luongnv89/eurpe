"""Tests for :mod:`eurpe.analytics.logger` — JSONL writer + handler cache.

These tests mirror :file:`tests/security/test_audit_log.py` because
the two modules share the per-path child-logger pattern. The
properties pinned here are the ones that make the analytics log a
trustworthy operational record:

* one JSON object per line, parseable in isolation;
* round-trip via :class:`BaseAnalyticsEvent.model_validate`;
* handler is reused on repeated calls (no duplicate writes);
* per-path isolation (two loggers, two files, no cross-talk);
* ``propagate=False`` keeps analytics lines out of root / stdout;
* keyword-only ``event`` argument forbids positional miswrites;
* :class:`AnalyticsLogger` validates the parent directory at
  construction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from eurpe.analytics.events import (
    DraftCompletedEvent,
    DraftStartedEvent,
    EventType,
)
from eurpe.analytics.logger import (
    _ANALYTICS_LOGGER_NAME,
    AnalyticsLogger,
    _child_logger_name,
    _reset_handlers_for_tests,
    log_event,
)


@pytest.fixture(autouse=True)
def _clean_analytics_handlers() -> None:
    """Detach cached handlers before AND after each test.

    Tests use ``tmp_path``, so handlers attached to a now-deleted file
    would silently fail subsequent writes. The autouse fixture is what
    makes the rest of the file safe to read top-to-bottom.
    """

    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_started_event() -> DraftStartedEvent:
    return DraftStartedEvent(
        event_type=EventType.DRAFT_STARTED,
        section_type="methodology",
        target_programme="horizon_europe",
        top_k_examples=5,
        lessons_learned=False,
        model="deterministic-stub-v1",
        drafting_profile=None,
        topic_context_present=False,
    )


def _make_completed_event() -> DraftCompletedEvent:
    return DraftCompletedEvent(
        event_type=EventType.DRAFT_COMPLETED,
        section_type="methodology",
        generation_time_ms=123,
        citation_count=2,
        source_status_mix={"funded": 2},
        model="deterministic-stub-v1",
        drafting_profile=None,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_log_event_roundtrip(tmp_path: Path) -> None:
    """An event written to the log can be parsed back into a model."""

    log_path = tmp_path / "analytics.log"
    event = _make_started_event()
    log_event(log_path, event=event)

    lines = _read_lines(log_path)
    assert len(lines) == 1
    record = json.loads(lines[0])
    # Round-trip into the model — Pydantic validates every field.
    parsed = DraftStartedEvent.model_validate(record)
    assert parsed.event_type is EventType.DRAFT_STARTED
    assert parsed.section_type == "methodology"
    assert parsed.top_k_examples == 5
    assert parsed.model == "deterministic-stub-v1"


# ---------------------------------------------------------------------------
# Handler reuse / no duplicate writes
# ---------------------------------------------------------------------------


def test_log_event_handler_is_reused_across_calls(tmp_path: Path) -> None:
    """Repeated calls on the same path must NOT attach duplicate handlers.

    Without handler caching, every ``log_event`` call would attach a
    fresh FileHandler and each subsequent line would be written N
    times. Regression guard for that bug class.
    """

    log_path = tmp_path / "analytics.log"
    event = _make_completed_event()
    for _ in range(5):
        log_event(log_path, event=event)

    lines = _read_lines(log_path)
    assert len(lines) == 5

    child = logging.getLogger(_child_logger_name(log_path))
    file_handlers = [
        h
        for h in child.handlers
        if isinstance(h, logging.FileHandler)
        and Path(h.baseFilename).resolve() == log_path.resolve()
    ]
    assert len(file_handlers) == 1


# ---------------------------------------------------------------------------
# Per-path isolation
# ---------------------------------------------------------------------------


def test_log_event_isolates_writes_per_path(tmp_path: Path) -> None:
    """Two analytics loggers with different paths must not cross-contaminate.

    Mirrors the per-path isolation invariant of
    :mod:`eurpe.security.audit`. A write to ``a.log`` appears in
    ``a.log`` only; a write to ``b.log`` appears in ``b.log`` only.
    """

    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    started = _make_started_event()
    completed = _make_completed_event()
    log_event(log_a, event=started)
    log_event(log_b, event=completed)

    a_records = [json.loads(line) for line in _read_lines(log_a)]
    b_records = [json.loads(line) for line in _read_lines(log_b)]
    assert len(a_records) == 1
    assert len(b_records) == 1
    assert a_records[0]["event_type"] == "draft_started"
    assert b_records[0]["event_type"] == "draft_completed"


# ---------------------------------------------------------------------------
# propagate=False — no leak to root logger
# ---------------------------------------------------------------------------


def test_log_event_does_not_propagate_to_root(tmp_path: Path, caplog) -> None:
    """Analytics lines must not leak into root logger / stdout / stderr.

    Mirrors the security-audit invariant: the analytics log is a file
    sink, not a logging channel. caplog captures records that
    propagate to root — if any analytics line surfaces here, the
    isolation is broken.
    """

    log_path = tmp_path / "analytics.log"
    event = _make_started_event()
    with caplog.at_level(logging.DEBUG):
        log_event(log_path, event=event)

    propagated = [r for r in caplog.records if r.name.startswith(_ANALYTICS_LOGGER_NAME)]
    assert propagated == []


# ---------------------------------------------------------------------------
# Keyword-only ``event`` argument
# ---------------------------------------------------------------------------


def test_log_event_keyword_only_event_argument(tmp_path: Path) -> None:
    """``event`` MUST be keyword-only.

    Allowing it positional would let a future regression swap the
    analytics path and the event at the call site without a
    :class:`TypeError`. Keyword-only enforces clarity at the call
    site.
    """

    log_path = tmp_path / "analytics.log"
    event = _make_started_event()
    with pytest.raises(TypeError):
        log_event(log_path, event)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AnalyticsLogger class
# ---------------------------------------------------------------------------


def test_analytics_logger_log_method(tmp_path: Path) -> None:
    """The :class:`AnalyticsLogger` wrapper delegates to :func:`log_event`."""

    log_path = tmp_path / "analytics.log"
    logger_obj = AnalyticsLogger(log_path)
    assert logger_obj.analytics_log_path == log_path

    logger_obj.log(_make_started_event())
    logger_obj.log(_make_completed_event())

    lines = _read_lines(log_path)
    assert len(lines) == 2
    types = [json.loads(line)["event_type"] for line in lines]
    assert types == ["draft_started", "draft_completed"]


def test_analytics_logger_rejects_missing_parent_dir(tmp_path: Path) -> None:
    """Constructing an :class:`AnalyticsLogger` whose parent dir is missing fails fast.

    Loud failure mode: a missing ``runtime_dir`` is a bootstrapping
    bug (``ensure_runtime_dirs`` wasn't called) — silent on-write
    failure would hide the original miswire.
    """

    missing_parent = tmp_path / "does-not-exist" / "analytics.log"
    with pytest.raises(FileNotFoundError):
        AnalyticsLogger(missing_parent)


def test_log_event_rejects_missing_parent_dir(tmp_path: Path) -> None:
    """:func:`log_event` directly also requires the parent dir to exist."""

    missing_parent = tmp_path / "does-not-exist" / "analytics.log"
    with pytest.raises(FileNotFoundError):
        log_event(missing_parent, event=_make_started_event())


# ---------------------------------------------------------------------------
# JSON shape sanity
# ---------------------------------------------------------------------------


def test_log_event_writes_canonical_json(tmp_path: Path) -> None:
    """The written line is canonical JSON: sorted keys, minimal whitespace.

    Canonical form makes the log diffable across runs and ``jq``-able
    without sort-key gymnastics.
    """

    log_path = tmp_path / "analytics.log"
    log_event(log_path, event=_make_started_event())
    line = _read_lines(log_path)[0]
    # No leading/trailing whitespace.
    assert line == line.strip()
    # Sorted keys: the first key alphabetically should appear first.
    record = json.loads(line)
    sorted_re_dump = json.dumps(record, sort_keys=True, separators=(",", ":"))
    assert line == sorted_re_dump
