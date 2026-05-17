"""Unit tests for :mod:`eurpe.pilot.models`.

Pin the structural invariants of the pilot report record:

* Pydantic ``extra="forbid"`` keeps typos in field names loud.
* The :class:`PilotMode` and :class:`GoNoGoVerdict` enums are
  closed vocabularies.
* :class:`SatisfactionRating` accepts ``None`` for rating /
  time-saved (smoke-mode default) and enforces the PRD's 1-5 scale.
* :class:`PilotReport.to_json` produces stable, sort-keyed output.

These tests do NOT exercise the runner — that lives in
``tests/pilot/test_runner.py``. Keeping the model tests cheap and
mock-free means the test suite stays under a second for the package.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eurpe.benchmarks import BenchmarkReport, RuntimeFingerprint
from eurpe.generation.audit_harness import ReleaseAuditReport
from eurpe.pilot.models import (
    CitationIssue,
    GoNoGoVerdict,
    PilotMode,
    PilotReport,
    PilotSectionResult,
    SatisfactionRating,
    SmokeResult,
)


def _make_benchmark_report() -> BenchmarkReport:
    """Helper: minimal BenchmarkReport for embedding in a PilotReport."""

    runtime = RuntimeFingerprint(
        runtime="deterministic",
        llm_model="deterministic-echo",
        embedder="DeterministicHashEmbedder",
        python_version="3.14.0",
        platform="Darwin/arm64",
        cpu_count=8,
    )
    return BenchmarkReport(generated_at=datetime(2026, 5, 17, tzinfo=UTC), runtime=runtime)


def _make_audit_report() -> ReleaseAuditReport:
    """Helper: empty-but-valid ReleaseAuditReport for embedding."""

    return ReleaseAuditReport(
        audit_directory="<test>",
        total_drafts=0,
        audited_drafts=0,
        passed_drafts=0,
        failed_drafts=0,
        citation_count=0,
        unlabeled_citation_count=0,
        passed=True,
    )


# ---------------------------------------------------------------------------
# Enum closure
# ---------------------------------------------------------------------------


def test_pilot_mode_values_are_closed() -> None:
    """PilotMode has exactly the two values the runner branches on."""

    assert {m.value for m in PilotMode} == {"smoke", "coordinator"}


def test_go_no_go_verdict_values_are_closed() -> None:
    """GoNoGoVerdict has exactly the three release verdicts."""

    assert {v.value for v in GoNoGoVerdict} == {"go", "no_go", "conditional"}


# ---------------------------------------------------------------------------
# SatisfactionRating
# ---------------------------------------------------------------------------


def test_satisfaction_rating_accepts_smoke_mode_blanks() -> None:
    """Smoke-mode runs leave rating/time_saved as None (the contract)."""

    rating = SatisfactionRating(coordinator_id="coord-a")
    assert rating.rating is None
    assert rating.time_saved_minutes is None
    assert rating.notes == ""


def test_satisfaction_rating_enforces_1_to_5_scale() -> None:
    """rating must be in the closed interval [1, 5]."""

    with pytest.raises(ValidationError):
        SatisfactionRating(coordinator_id="coord-a", rating=0)
    with pytest.raises(ValidationError):
        SatisfactionRating(coordinator_id="coord-a", rating=6)
    # Boundaries pass.
    SatisfactionRating(coordinator_id="coord-a", rating=1)
    SatisfactionRating(coordinator_id="coord-a", rating=5)


def test_satisfaction_rating_rejects_negative_time_saved() -> None:
    """Coordinators cannot report negative time saved."""

    with pytest.raises(ValidationError):
        SatisfactionRating(coordinator_id="coord-a", time_saved_minutes=-5)


def test_satisfaction_rating_rejects_unknown_field() -> None:
    """extra=forbid catches typos in field names."""

    with pytest.raises(ValidationError):
        SatisfactionRating(coordinator_id="coord-a", grade=4)


# ---------------------------------------------------------------------------
# CitationIssue / SmokeResult / PilotSectionResult — basic invariants
# ---------------------------------------------------------------------------


def test_citation_issue_requires_non_empty_fields() -> None:
    """draft_path / section_type / code / message all reject empty strings."""

    with pytest.raises(ValidationError):
        CitationIssue(draft_path="", section_type="m", code="c", message="msg")
    with pytest.raises(ValidationError):
        CitationIssue(draft_path="p", section_type="", code="c", message="msg")
    with pytest.raises(ValidationError):
        CitationIssue(draft_path="p", section_type="m", code="", message="msg")
    with pytest.raises(ValidationError):
        CitationIssue(draft_path="p", section_type="m", code="c", message="")


def test_smoke_result_pass_path() -> None:
    """The happy path: probe denied, exit 0, recorded as PASS."""

    sr = SmokeResult(passed=True, exit_code=0, detail="TEST-NET denied")
    assert sr.passed is True
    assert sr.exit_code == 0


def test_smoke_result_rejects_negative_exit_code() -> None:
    """``exit_code`` must be >= 0 (POSIX-shaped)."""

    with pytest.raises(ValidationError):
        SmokeResult(passed=False, exit_code=-1)


def test_pilot_section_result_defaults_empty_satisfaction() -> None:
    """satisfaction list defaults to empty (smoke mode contract)."""

    sec = PilotSectionResult(
        section_type="methodology",
        user_intent="x",
        citation_count=2,
        draft_length=300,
        elapsed_ms=10,
        audit_passed=True,
    )
    assert sec.satisfaction == []


# ---------------------------------------------------------------------------
# PilotReport — aggregate + JSON roundtrip
# ---------------------------------------------------------------------------


def test_pilot_report_roundtrips_through_json() -> None:
    """A minimal PilotReport survives a model_dump → JSON → validate cycle."""

    report = PilotReport(
        mode=PilotMode.SMOKE,
        call_id="HORIZON-CL5-2024-D3-02",
        proposal_title="Test proposal",
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.CONDITIONAL,
    )
    payload = json.loads(report.to_json())
    assert payload["mode"] == "smoke"
    assert payload["call_id"] == "HORIZON-CL5-2024-D3-02"
    assert payload["verdict"] == "conditional"
    # Reconstruct: must validate without errors.
    PilotReport.model_validate(payload)


def test_pilot_report_to_json_is_sort_keyed() -> None:
    """``to_json`` sorts keys so diffs are stable across runs."""

    report = PilotReport(
        mode=PilotMode.SMOKE,
        call_id="X",
        proposal_title="Y",
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.CONDITIONAL,
    )
    text = report.to_json()
    # ``call_id`` should sort before ``mode`` before ``verdict``.
    assert text.index('"call_id"') < text.index('"mode"') < text.index('"verdict"')
