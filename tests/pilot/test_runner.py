"""Tests for :mod:`eurpe.pilot.runner` — the orchestrator.

End-to-end coverage of the pilot flow under the deterministic
backends. The runner composes existing primitives (chunker, index,
retriever, workflow, audit, benchmark, smoke), so the test surface is
deliberately behavioural: we assert on the shape of the produced
:class:`PilotReport` (sections, smoke result, verdict precedence)
rather than instrumenting every internal helper.

The default smoke path is offline-by-construction — no network
fixture override needed.
"""

from __future__ import annotations

from pathlib import Path

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
from eurpe.pilot.runner import (
    DEFAULT_SECTION_TYPES,
    PilotConfig,
    PilotRunError,
    _compute_verdict,
    _smoke_audit_blocking,
    attach_satisfaction,
    load_pilot_report,
    render_pilot_report_markdown,
    run_pilot,
)
from eurpe.schema import SectionType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audit_report(*, passed: bool = True) -> ReleaseAuditReport:
    return ReleaseAuditReport(
        audit_directory="<test>",
        total_drafts=0,
        audited_drafts=0,
        passed_drafts=0,
        failed_drafts=0,
        citation_count=0,
        unlabeled_citation_count=0,
        passed=passed,
    )


def _make_benchmark_report() -> BenchmarkReport:
    runtime = RuntimeFingerprint(
        runtime="deterministic",
        llm_model="deterministic-echo",
        embedder="DeterministicHashEmbedder",
        python_version="3.14.0",
        platform="Darwin/arm64",
        cpu_count=8,
    )
    return BenchmarkReport(runtime=runtime)


def _make_section_result(
    section_type: str = "methodology",
    satisfaction: list[SatisfactionRating] | None = None,
) -> PilotSectionResult:
    return PilotSectionResult(
        section_type=section_type,
        user_intent="probe",
        citation_count=2,
        draft_length=200,
        elapsed_ms=10,
        audit_passed=True,
        satisfaction=satisfaction or [],
    )


# ---------------------------------------------------------------------------
# PilotConfig — AC1 enforcement
# ---------------------------------------------------------------------------


def test_pilot_config_enforces_at_least_three_section_types() -> None:
    """AC1 of issue #21 requires ≥3 sections; PilotConfig enforces it."""

    with pytest.raises(ValidationError):
        PilotConfig(section_types=(SectionType.METHODOLOGY,))


def test_pilot_config_default_section_types_meet_ac1() -> None:
    """The default trio satisfies AC1 without explicit configuration."""

    cfg = PilotConfig()
    assert len(cfg.section_types) >= 3
    assert cfg.section_types == DEFAULT_SECTION_TYPES


# ---------------------------------------------------------------------------
# _smoke_audit_blocking — tolerate placeholder_text in smoke mode
# ---------------------------------------------------------------------------


def test_smoke_audit_blocking_tolerates_placeholder_text_only() -> None:
    """The stub's expected placeholder_text finding is the *only* tolerated code."""

    issues = [
        CitationIssue(draft_path="x", section_type="m", code="placeholder_text", message="x"),
    ]
    assert _smoke_audit_blocking(issues) is False


def test_smoke_audit_blocking_flags_any_other_finding() -> None:
    """Any non-tolerated code blocks the smoke verdict from CONDITIONAL."""

    issues = [
        CitationIssue(draft_path="x", section_type="m", code="placeholder_text", message="x"),
        CitationIssue(draft_path="x", section_type="m", code="missing_status", message="x"),
    ]
    assert _smoke_audit_blocking(issues) is True


# ---------------------------------------------------------------------------
# _compute_verdict — verdict precedence
# ---------------------------------------------------------------------------


def test_compute_verdict_smoke_fail_is_no_go() -> None:
    """Smoke probe FAIL always wins, regardless of mode."""

    verdict = _compute_verdict(
        mode=PilotMode.COORDINATOR,
        smoke=SmokeResult(passed=False, exit_code=1),
        audit=_make_audit_report(passed=True),
        citation_issues=[],
        section_results=[],
    )
    assert verdict == GoNoGoVerdict.NO_GO


def test_compute_verdict_coordinator_audit_fail_is_no_go() -> None:
    """Coordinator mode is strict — any audit failure blocks GO."""

    verdict = _compute_verdict(
        mode=PilotMode.COORDINATOR,
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(passed=False),
        citation_issues=[],
        section_results=[],
    )
    assert verdict == GoNoGoVerdict.NO_GO


def test_compute_verdict_smoke_audit_with_only_placeholder_is_conditional() -> None:
    """Smoke mode tolerates placeholder_text only — verdict stays CONDITIONAL."""

    verdict = _compute_verdict(
        mode=PilotMode.SMOKE,
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(passed=False),
        citation_issues=[
            CitationIssue(draft_path="x", section_type="m", code="placeholder_text", message="x")
        ],
        section_results=[_make_section_result()],
    )
    assert verdict == GoNoGoVerdict.CONDITIONAL


def test_compute_verdict_coordinator_with_high_ratings_is_go() -> None:
    """Coordinator pilot with mean ≥4 on every section → GO."""

    sat = SatisfactionRating(coordinator_id="coord-a", rating=4, time_saved_minutes=30)
    verdict = _compute_verdict(
        mode=PilotMode.COORDINATOR,
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(passed=True),
        citation_issues=[],
        section_results=[_make_section_result(satisfaction=[sat])],
    )
    assert verdict == GoNoGoVerdict.GO


def test_compute_verdict_coordinator_with_low_rating_is_no_go() -> None:
    """Coordinator pilot with mean <4 → NO_GO (the PRD success-criteria floor)."""

    sat = SatisfactionRating(coordinator_id="coord-a", rating=3)
    verdict = _compute_verdict(
        mode=PilotMode.COORDINATOR,
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(passed=True),
        citation_issues=[],
        section_results=[_make_section_result(satisfaction=[sat])],
    )
    assert verdict == GoNoGoVerdict.NO_GO


def test_compute_verdict_coordinator_with_missing_rating_is_conditional() -> None:
    """Coordinator mode without ratings is CONDITIONAL — same honesty rule."""

    verdict = _compute_verdict(
        mode=PilotMode.COORDINATOR,
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(passed=True),
        citation_issues=[],
        section_results=[_make_section_result(satisfaction=[])],
    )
    assert verdict == GoNoGoVerdict.CONDITIONAL


# ---------------------------------------------------------------------------
# run_pilot — end-to-end under the deterministic stubs
# ---------------------------------------------------------------------------


@pytest.fixture
def no_network() -> None:
    """Override the parent ``no_network`` fixture.

    The pilot runner uses Chroma's local sqlite-over-socket access
    (same as the E2E suite), so the repo-wide ``socket.socket.connect``
    monkeypatch in ``tests/conftest.py`` would fail the run. This
    no-op override lets the pilot tests run; the *contract* that the
    runner does not touch the public network is enforced by the
    ``offline_mode: true`` default and the in-process smoke probe.
    """


def test_run_pilot_smoke_mode_produces_three_sections() -> None:
    """AC1: smoke mode produces the default trio of sections."""

    report = run_pilot()
    assert report.mode == PilotMode.SMOKE
    assert len(report.section_results) == 3
    section_types = {s.section_type for s in report.section_results}
    assert section_types == {"methodology", "impact", "implementation"}


def test_run_pilot_smoke_mode_verdict_is_conditional() -> None:
    """AC3: smoke mode renders CONDITIONAL (no real coordinator yet)."""

    report = run_pilot()
    assert report.verdict == GoNoGoVerdict.CONDITIONAL


def test_run_pilot_smoke_mode_includes_smoke_probe_pass() -> None:
    """AC3: the network isolation smoke test is recorded as PASS."""

    report = run_pilot()
    assert report.smoke.passed is True
    assert report.smoke.exit_code == 0


def test_run_pilot_smoke_mode_includes_benchmark_snapshot() -> None:
    """AC3: the performance snapshot is embedded."""

    report = run_pilot()
    assert report.benchmark.runtime.runtime == "deterministic"
    assert report.benchmark.indexing is not None
    assert report.benchmark.retrieval is not None
    assert report.benchmark.generation is not None


def test_run_pilot_smoke_mode_records_call_id_on_every_section(tmp_path: Path) -> None:
    """The configured call_id flows into the indexed corpus → citations."""

    cfg = PilotConfig(call_id="HORIZON-CL5-2024-D3-02")
    report = run_pilot(config=cfg, output_dir=tmp_path)
    # The audit report walks the on-disk drafts; every citation it
    # surfaces must carry our configured call_id.
    assert report.audit.rows, "expected at least one citation in the audit report"
    for row in report.audit.rows:
        assert row.call_id == "HORIZON-CL5-2024-D3-02"


def test_run_pilot_persists_artefacts_under_output_dir(tmp_path: Path) -> None:
    """``output_dir`` lays down per-section JSON + Markdown + the report JSON."""

    report = run_pilot(output_dir=tmp_path)
    assert (tmp_path / "pilot-report.json").exists()
    for sec in report.section_results:
        # Per-section base path is ``<output_dir>/<section_type>``.
        base = tmp_path / sec.section_type
        assert base.with_suffix(".json").exists(), f"missing JSON for {sec.section_type}"
        assert base.with_suffix(".md").exists(), f"missing MD for {sec.section_type}"
        assert sec.draft_path == base.with_suffix(".json").as_posix()


def test_run_pilot_report_json_round_trips(tmp_path: Path) -> None:
    """The persisted JSON is loadable back into a PilotReport."""

    run_pilot(output_dir=tmp_path)
    loaded = load_pilot_report(tmp_path / "pilot-report.json")
    assert loaded.mode == PilotMode.SMOKE
    assert len(loaded.section_results) == 3


# ---------------------------------------------------------------------------
# render_pilot_report_markdown — every AC3 field must appear
# ---------------------------------------------------------------------------


def test_render_pilot_report_markdown_includes_all_ac3_sections() -> None:
    """AC3 names five required fields — every one of them has a section header."""

    report = PilotReport(
        mode=PilotMode.SMOKE,
        call_id="HORIZON-CL5-2024-D3-02",
        proposal_title="Test proposal",
        smoke=SmokeResult(passed=True, exit_code=0, detail="OK"),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.CONDITIONAL,
    )
    md = render_pilot_report_markdown(report)
    # AC3's five required sub-reports.
    assert "Coordinator satisfaction" in md
    assert "Citation issues" in md
    assert "Performance" in md
    assert "Network isolation smoke" in md
    assert "Go / No-Go" in md
    # Verdict appears in upper-case for visibility.
    assert "CONDITIONAL" in md


def test_render_pilot_report_marks_pending_satisfaction_for_smoke_runs() -> None:
    """Smoke-mode reports render satisfaction cells as <pending>."""

    report = PilotReport(
        mode=PilotMode.SMOKE,
        call_id="X",
        proposal_title="Y",
        section_results=[_make_section_result()],
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.CONDITIONAL,
    )
    md = render_pilot_report_markdown(report)
    assert "<pending>" in md


def test_render_pilot_report_includes_coordinator_ratings_when_present() -> None:
    """Coordinator-mode reports render the rating row verbatim."""

    sat = SatisfactionRating(
        coordinator_id="coord-a", rating=5, time_saved_minutes=45, notes="great"
    )
    report = PilotReport(
        mode=PilotMode.COORDINATOR,
        call_id="X",
        proposal_title="Y",
        section_results=[_make_section_result(satisfaction=[sat])],
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.GO,
    )
    md = render_pilot_report_markdown(report)
    assert "coord-a" in md
    assert "great" in md
    assert "GO" in md


# ---------------------------------------------------------------------------
# attach_satisfaction
# ---------------------------------------------------------------------------


def test_attach_satisfaction_returns_new_report_without_mutating_input() -> None:
    """The helper is pure — input report is unchanged."""

    base = PilotReport(
        mode=PilotMode.COORDINATOR,
        call_id="X",
        proposal_title="Y",
        section_results=[_make_section_result(section_type="methodology")],
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.CONDITIONAL,
    )
    rating = SatisfactionRating(coordinator_id="coord-a", rating=4)
    updated = attach_satisfaction(report=base, section_type="methodology", rating=rating)
    assert updated is not base
    assert base.section_results[0].satisfaction == []
    assert updated.section_results[0].satisfaction[0].rating == 4
    # Verdict recomputed: rating=4 on the only section → GO.
    assert updated.verdict == GoNoGoVerdict.GO


def test_attach_satisfaction_raises_when_section_missing() -> None:
    """Unknown section_type surfaces a PilotRunError with a hint."""

    base = PilotReport(
        mode=PilotMode.COORDINATOR,
        call_id="X",
        proposal_title="Y",
        section_results=[_make_section_result(section_type="methodology")],
        smoke=SmokeResult(passed=True, exit_code=0),
        audit=_make_audit_report(),
        benchmark=_make_benchmark_report(),
        verdict=GoNoGoVerdict.CONDITIONAL,
    )
    with pytest.raises(PilotRunError, match="not found"):
        attach_satisfaction(
            report=base,
            section_type="nonexistent",
            rating=SatisfactionRating(coordinator_id="coord-a", rating=4),
        )
