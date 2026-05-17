"""Tests for ``eurpe pilot ...`` Typer commands.

Cover the happy paths and the file-system contract:

* ``eurpe pilot run`` produces JSON + Markdown artefacts when
  ``--output-dir`` / ``--output-json`` / ``--output-markdown`` are
  passed.
* The verdict drives the exit code: ``CONDITIONAL`` / ``GO`` → 0;
  ``NO_GO`` → 1.
* ``eurpe pilot rate`` reads a saved report, appends a
  :class:`SatisfactionRating`, and writes back an updated JSON whose
  verdict reflects the new rating.

These tests do NOT use the parent ``no_network`` fixture because the
pilot uses Chroma's local sqlite-over-socket access (same as the E2E
suite); we override it here to a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eurpe.cli import app


@pytest.fixture
def no_network() -> None:
    """Override the repo-wide ``no_network`` fixture for the pilot CLI tests.

    The pilot's Chroma backend talks over a local sqlite socket, which
    the blanket ``socket.socket.connect`` patch would block. The
    offline contract is enforced by ``offline_mode: true`` + the
    in-process smoke probe, not by the patched socket.
    """


# ---------------------------------------------------------------------------
# eurpe pilot run — happy path
# ---------------------------------------------------------------------------


def test_pilot_run_default_smoke_prints_summary() -> None:
    """The default invocation produces a stdout summary and exits 0."""

    runner = CliRunner()
    result = runner.invoke(app, ["pilot", "run"])
    assert result.exit_code == 0, result.stdout
    # Required summary lines.
    assert "Pilot mode      : smoke" in result.stdout
    assert "Sections        : 3" in result.stdout
    assert "Smoke probe     : PASS" in result.stdout
    assert "Verdict         : CONDITIONAL" in result.stdout


def test_pilot_run_writes_aggregate_artefacts(tmp_path: Path) -> None:
    """--output-dir lays down per-section files plus pilot-report.json."""

    runner = CliRunner()
    out = tmp_path / "pilot"
    result = runner.invoke(app, ["pilot", "run", "--output-dir", str(out)])
    assert result.exit_code == 0, result.stdout
    assert (out / "pilot-report.json").exists()
    for section in ("methodology", "impact", "implementation"):
        assert (out / f"{section}.json").exists()
        assert (out / f"{section}.md").exists()


def test_pilot_run_writes_explicit_output_json(tmp_path: Path) -> None:
    """--output-json writes the aggregate JSON to the exact path requested."""

    runner = CliRunner()
    target = tmp_path / "nested" / "report.json"
    result = runner.invoke(app, ["pilot", "run", "--output-json", str(target)])
    assert result.exit_code == 0, result.stdout
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["mode"] == "smoke"
    assert payload["verdict"] == "conditional"


def test_pilot_run_writes_explicit_output_markdown(tmp_path: Path) -> None:
    """--output-markdown writes the rendered report next to (or instead of) JSON."""

    runner = CliRunner()
    target = tmp_path / "report.md"
    result = runner.invoke(app, ["pilot", "run", "--output-markdown", str(target)])
    assert result.exit_code == 0, result.stdout
    md = target.read_text(encoding="utf-8")
    assert "# MVP Pilot Validation Report" in md
    assert "Go / No-Go" in md


def test_pilot_run_accepts_custom_section_types(tmp_path: Path) -> None:
    """Repeated --section-type / -s flags override the default trio."""

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pilot",
            "run",
            "-s",
            "methodology",
            "-s",
            "impact",
            "-s",
            "consortium",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Sections        : 3" in result.stdout


def test_pilot_run_rejects_unknown_mode() -> None:
    """An unknown --mode value surfaces a clean BadParameter."""

    runner = CliRunner()
    result = runner.invoke(app, ["pilot", "run", "--mode", "release"])
    assert result.exit_code != 0
    assert "must be one of" in (result.stdout + result.stderr)


def test_pilot_run_rejects_unknown_section_type() -> None:
    """An unknown --section-type value surfaces a clean BadParameter."""

    runner = CliRunner()
    result = runner.invoke(app, ["pilot", "run", "-s", "methodology", "-s", "foo", "-s", "impact"])
    assert result.exit_code != 0
    assert "must be one of" in (result.stdout + result.stderr)


def test_pilot_run_rejects_fewer_than_three_section_types() -> None:
    """AC1 of issue #21 — pilot must produce at least three section drafts."""

    runner = CliRunner()
    result = runner.invoke(app, ["pilot", "run", "-s", "methodology"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# eurpe pilot rate — happy path
# ---------------------------------------------------------------------------


def test_pilot_rate_attaches_rating_and_updates_verdict(tmp_path: Path) -> None:
    """A coordinator rating ≥4 on every section flips the verdict to GO."""

    runner = CliRunner()
    out = tmp_path / "p"
    # Use coordinator mode so the placeholder-text audit failure does
    # block GO — we feed in a workflow that yields a passing audit by
    # using a custom config and the deterministic stub. Since we run
    # ``--mode coordinator`` but with deterministic runtime, the audit
    # WILL fail; that's why the rate path here pins the smoke case
    # instead. Specifically: run a smoke-mode pilot, manually re-write
    # the JSON to coordinator mode with a clean audit, then rate it.
    init_result = runner.invoke(app, ["pilot", "run", "--output-dir", str(out)])
    assert init_result.exit_code == 0, init_result.stdout
    report_path = out / "pilot-report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    # Patch to coordinator mode + clean audit so the rating drives the
    # verdict. (The runner's audit failure is the deterministic stub's
    # known placeholder; coordinator mode would normally reject it.)
    payload["mode"] = "coordinator"
    payload["audit"]["passed"] = True
    payload["audit"]["failed_drafts"] = 0
    payload["audit"]["passed_drafts"] = payload["audit"]["audited_drafts"]
    payload["audit"]["draft_results"] = [
        {**dr, "passed": True} for dr in payload["audit"]["draft_results"]
    ]
    payload["citation_issues"] = []
    payload["verdict"] = "conditional"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    # Now rate every section; the final write should flip verdict to GO.
    for section in ("methodology", "impact", "implementation"):
        rate_result = runner.invoke(
            app,
            [
                "pilot",
                "rate",
                str(report_path),
                "-s",
                section,
                "--coordinator-id",
                "coord-a",
                "--rating",
                "5",
                "--time-saved",
                "30",
            ],
        )
        assert rate_result.exit_code == 0, rate_result.stdout

    final = json.loads(report_path.read_text(encoding="utf-8"))
    assert final["verdict"] == "go"
    # Every section has at least one rating.
    for sec in final["section_results"]:
        assert sec["satisfaction"]


def test_pilot_rate_fails_when_section_missing(tmp_path: Path) -> None:
    """Unknown --section-type surfaces a [FAIL] message + exit 1."""

    runner = CliRunner()
    out = tmp_path / "p"
    init_result = runner.invoke(app, ["pilot", "run", "--output-dir", str(out)])
    assert init_result.exit_code == 0, init_result.stdout
    report_path = out / "pilot-report.json"

    rate_result = runner.invoke(
        app,
        [
            "pilot",
            "rate",
            str(report_path),
            "-s",
            "nonexistent",
            "--coordinator-id",
            "coord-a",
            "--rating",
            "5",
        ],
    )
    assert rate_result.exit_code != 0
    output = rate_result.stdout + rate_result.stderr
    assert "not found" in output or "[FAIL]" in output


def test_pilot_rate_fails_when_report_missing(tmp_path: Path) -> None:
    """Missing report file surfaces a clean [FAIL] without a traceback."""

    runner = CliRunner()
    rate_result = runner.invoke(
        app,
        [
            "pilot",
            "rate",
            str(tmp_path / "does-not-exist.json"),
            "-s",
            "methodology",
            "--coordinator-id",
            "coord-a",
            "--rating",
            "5",
        ],
    )
    assert rate_result.exit_code != 0
    output = rate_result.stdout + rate_result.stderr
    assert "not found" in output.lower() or "[FAIL]" in output
