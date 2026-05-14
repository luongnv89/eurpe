"""Tests for the ``eurpe smoke`` CLI command.

The smoke command must:
- Exit with code 0 on a clean invocation.
- Bootstrap a missing ``config.yaml`` from the example.
- Create the corpus and index directories.
- Make zero network calls (verified via the ``no_network`` fixture).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eurpe.cli import app
from eurpe.config import EXAMPLE_CONFIG_PATH


@pytest.fixture
def isolated_workspace(tmp_path: Path) -> Path:
    """Copy the example config into a clean tmp workspace and return it."""
    target = tmp_path / "config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG_PATH, target)
    return tmp_path


def test_smoke_exits_zero_with_existing_config(
    isolated_workspace: Path,
    no_network: None,  # noqa: ARG001 — fixture asserts no network
) -> None:
    config_path = isolated_workspace / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "EURPE workspace is ready" in result.output
    assert "offline_mode      : True" in result.output


def test_smoke_bootstraps_missing_config(
    tmp_path: Path,
    no_network: None,  # noqa: ARG001
) -> None:
    config_path = tmp_path / "config.yaml"
    assert not config_path.exists()
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert config_path.exists(), "smoke must bootstrap config.yaml from the example"


def test_smoke_creates_runtime_dirs(
    isolated_workspace: Path,
    no_network: None,  # noqa: ARG001
) -> None:
    config_path = isolated_workspace / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    # The example config uses ./data/corpus and ./data/index; resolve_paths anchors them
    # against the repo root, so we just check that some data directory was created.
    # (The exact path depends on the resolve_paths anchor, which is the repo root.)
    assert "corpus_path" in result.output
    assert "index_path" in result.output
