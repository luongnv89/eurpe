"""Tests for ``eurpe analytics export`` — the AC3 chokepoint command.

AC3 of issue #13: "Local analytics are disabled from external export
unless a user explicitly exports them." The ``export`` command is the
ONLY code path in the codebase that copies the analytics JSONL file
outside the runtime directory, and these tests pin its behaviour:

* The output flag is REQUIRED (no default destination) — an export
  without an explicit target fails with a Typer usage error.
* The source log must exist; absent → exit 1 with a clear error.
* An existing destination is refused unless ``--overwrite`` is set.
* With ``--overwrite``, the existing destination is replaced atomically.
* On success, the destination contains the same bytes as the source.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eurpe.analytics.logger import _reset_handlers_for_tests
from eurpe.cli import app
from eurpe.config import EXAMPLE_CONFIG_PATH


@pytest.fixture(autouse=True)
def _clean_analytics_handlers() -> None:
    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


@pytest.fixture
def isolated_workspace(tmp_path: Path) -> Path:
    """Copy the example config into ``tmp_path`` and return the workspace.

    Mirrors the pattern in ``tests/test_smoke.py``. Using the example
    config keeps these tests robust to repo-wide config-shape changes:
    the tests only need ``runtime_dir`` and the ``analytics_log_path``
    method, both of which the example config provides.
    """

    target = tmp_path / "config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG_PATH, target)
    # Point the workspace dirs into the tmp tree so a stray run cannot
    # write into the repo's ./data folder.
    content = target.read_text(encoding="utf-8")
    content = content.replace("./data/corpus", str(tmp_path / "corpus"))
    content = content.replace("./data/index", str(tmp_path / "index"))
    content = content.replace("./data/runtime", str(tmp_path / "runtime"))
    target.write_text(content, encoding="utf-8")
    return tmp_path


def _seed_analytics_log(workspace: Path) -> Path:
    """Create the analytics log under the workspace runtime_dir.

    Returns the path to the written file. Two minimal valid JSONL lines
    are enough to exercise the line-count summary.
    """

    runtime_dir = workspace / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "analytics-events.log"
    log_path.write_text(
        '{"event_type":"draft_started"}\n{"event_type":"draft_completed"}\n',
        encoding="utf-8",
    )
    return log_path


# ---------------------------------------------------------------------------
# Happy path — explicit --output
# ---------------------------------------------------------------------------


def test_export_copies_log_to_output(isolated_workspace: Path) -> None:
    """``analytics export --output X`` copies the log to X."""

    source = _seed_analytics_log(isolated_workspace)
    target = isolated_workspace / "out" / "events.jsonl"
    config_path = isolated_workspace / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analytics",
            "export",
            "--output",
            str(target),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    assert target.read_bytes() == source.read_bytes()
    # The summary line includes the event count.
    assert "exported 2 events" in result.output


# ---------------------------------------------------------------------------
# AC3 — --output is REQUIRED (no default destination)
# ---------------------------------------------------------------------------


def test_export_without_output_fails(isolated_workspace: Path) -> None:
    """No ``--output`` → non-zero exit (Typer missing-option error).

    AC3 chokepoint: an export with no explicit target is the user
    being unclear about where the data should land; we refuse rather
    than guess. Typer surfaces this as a usage error before our
    command body runs.
    """

    _seed_analytics_log(isolated_workspace)
    config_path = isolated_workspace / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["analytics", "export", "--config", str(config_path)],
    )
    assert result.exit_code != 0
    # Typer's missing-option message names the flag.
    assert "--output" in (result.output + (result.stderr or ""))


# ---------------------------------------------------------------------------
# Source-missing failure mode
# ---------------------------------------------------------------------------


def test_export_fails_when_source_log_missing(isolated_workspace: Path) -> None:
    """Source log missing → exit 1 with a clear error.

    The runtime_dir exists (smoke creates it), but no events have
    been recorded yet. The CLI must say so explicitly rather than
    creating an empty destination file.
    """

    # Do not seed the log.
    target = isolated_workspace / "out" / "events.jsonl"
    config_path = isolated_workspace / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analytics",
            "export",
            "--output",
            str(target),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 1, result.output
    combined = result.output + (result.stderr or "")
    assert "analytics log not found" in combined
    assert not target.exists()


# ---------------------------------------------------------------------------
# Overwrite handling
# ---------------------------------------------------------------------------


def test_export_refuses_to_clobber_existing_target(isolated_workspace: Path) -> None:
    """Existing destination without ``--overwrite`` → exit 1.

    Defensive: a user who already has an exported analytics file at
    the destination should not lose it to a stray re-export.
    """

    source = _seed_analytics_log(isolated_workspace)
    target = isolated_workspace / "out" / "events.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("PREEXISTING\n", encoding="utf-8")
    pre_existing_bytes = target.read_bytes()
    config_path = isolated_workspace / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analytics",
            "export",
            "--output",
            str(target),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 1, result.output
    combined = result.output + (result.stderr or "")
    assert "already exists" in combined
    # Pre-existing content was preserved.
    assert target.read_bytes() == pre_existing_bytes
    # Source is untouched.
    assert source.exists()


def test_export_overwrites_when_flag_set(isolated_workspace: Path) -> None:
    """Existing destination with ``--overwrite`` → destination is replaced."""

    source = _seed_analytics_log(isolated_workspace)
    target = isolated_workspace / "out" / "events.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("PREEXISTING\n", encoding="utf-8")
    config_path = isolated_workspace / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analytics",
            "export",
            "--output",
            str(target),
            "--overwrite",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert target.read_bytes() == source.read_bytes()


# ---------------------------------------------------------------------------
# Atomic write — tmp file is cleaned up
# ---------------------------------------------------------------------------


def test_export_does_not_leave_tmp_file_behind(isolated_workspace: Path) -> None:
    """After a successful export, no ``.tmp`` sibling lingers."""

    _seed_analytics_log(isolated_workspace)
    target = isolated_workspace / "out" / "events.jsonl"
    config_path = isolated_workspace / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analytics",
            "export",
            "--output",
            str(target),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    tmp_sibling = target.with_suffix(target.suffix + ".tmp")
    assert not tmp_sibling.exists()
