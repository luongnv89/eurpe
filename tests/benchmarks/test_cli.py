"""Tests for ``eurpe benchmark ...`` Typer commands.

Cover every sub-command happy path plus the ``--output`` JSON file
contract. Mirrors ``tests/analytics/test_cli_export.py``: uses
:class:`typer.testing.CliRunner`, points the workspace into
``tmp_path`` so a stray run cannot touch the repo's ``data`` dir, and
asserts on both stdout and the file system.

These tests do NOT use ``no_network`` — the CLI's default
``--runtime deterministic`` path does no I/O outside the temp dir,
and confirming that property is the runner-layer's job (it has its
own ``no_network`` test). The CLI-layer tests focus on exit codes,
output formatting, and the JSON contract.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eurpe.cli import app
from eurpe.config import EXAMPLE_CONFIG_PATH


@pytest.fixture
def isolated_workspace(tmp_path: Path) -> Path:
    """Mirror of the fixture from ``tests/analytics/test_cli_export.py``."""

    target = tmp_path / "config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG_PATH, target)
    content = target.read_text(encoding="utf-8")
    content = content.replace("./data/corpus", str(tmp_path / "corpus"))
    content = content.replace("./data/index", str(tmp_path / "index"))
    content = content.replace("./data/runtime", str(tmp_path / "runtime"))
    target.write_text(content, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# eurpe benchmark all
# ---------------------------------------------------------------------------


def test_benchmark_all_prints_full_summary(isolated_workspace: Path) -> None:
    """The default ``all`` invocation produces every section of the summary."""

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "all",
            "--proposals",
            "2",
            "--top-k",
            "3",
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    # The four sections are the structural contract of the summary.
    assert "Runtime" in result.output
    assert "Indexing" in result.output
    assert "Retrieval" in result.output
    assert "Generation" in result.output
    # AC3: model/runtime configuration must appear in the output.
    assert "runtime" in result.output.lower()
    assert "llm_model" in result.output
    assert "embedder" in result.output


def test_benchmark_all_writes_json_when_output_given(
    isolated_workspace: Path,
) -> None:
    """``--output PATH`` writes a parseable JSON report at the requested path."""

    runner = CliRunner()
    output_path = isolated_workspace / "benchmark.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "all",
            "--proposals",
            "2",
            "--output",
            str(output_path),
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists(), "expected JSON report to be written"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    # Structural shape — every measurement is present in the ``all``
    # report and the runtime fingerprint is populated.
    assert payload["runtime"]["runtime"] == "deterministic"
    assert payload["runtime"]["llm_model"]
    assert payload["runtime"]["embedder"]
    assert payload["indexing"]["proposal_count"] == 2
    assert payload["retrieval"]["query_count"] >= 1
    assert payload["generation"]["section_type"] == "methodology"
    # AC3 explicitly: the report names the model + runtime so a
    # reviewer can compare against the PRD targets.
    for key in ("runtime", "llm_model", "embedder", "python_version", "platform"):
        assert payload["runtime"][key], f"missing runtime.{key}"


def test_benchmark_all_rejects_unknown_runtime(isolated_workspace: Path) -> None:
    """Bad runtime value exits non-zero with the expected message."""

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "all",
            "--runtime",
            "not-a-runtime",
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "Unknown runtime" in result.output or "not-a-runtime" in result.output


# ---------------------------------------------------------------------------
# Per-AC sub-commands — each one MUST satisfy its acceptance criterion
# in isolation, because operators may run them individually.
# ---------------------------------------------------------------------------


def test_benchmark_indexing_satisfies_ac1(isolated_workspace: Path) -> None:
    """AC1: 'Benchmark measures initial indexing time for a fixture corpus.'"""

    runner = CliRunner()
    output_path = isolated_workspace / "indexing.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "indexing",
            "--proposals",
            "2",
            "--output",
            str(output_path),
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["indexing"] is not None
    assert payload["indexing"]["proposal_count"] == 2
    assert payload["indexing"]["chunk_count"] > 0
    # ``elapsed_ms`` may be 0 on a fast machine — the validator
    # accepts it; we only assert the field is present.
    assert "elapsed_ms" in payload["indexing"]


def test_benchmark_retrieval_satisfies_ac2(isolated_workspace: Path) -> None:
    """AC2: 'Benchmark measures retrieval latency for top-k retrieval.'"""

    runner = CliRunner()
    output_path = isolated_workspace / "retrieval.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "retrieval",
            "--proposals",
            "2",
            "--top-k",
            "3",
            "--output",
            str(output_path),
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["retrieval"] is not None
    assert payload["retrieval"]["top_k"] == 3
    assert payload["retrieval"]["query_count"] >= 1
    # The distribution stats are the headline numbers for AC2.
    for key in (
        "elapsed_ms_min",
        "elapsed_ms_avg",
        "elapsed_ms_p95",
        "elapsed_ms_max",
    ):
        assert key in payload["retrieval"], f"missing retrieval.{key}"


def test_benchmark_generation_satisfies_ac3(isolated_workspace: Path) -> None:
    """AC3: 'Benchmark measures section generation latency and reports
    model/runtime configuration.'
    """

    runner = CliRunner()
    output_path = isolated_workspace / "generation.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "generation",
            "--proposals",
            "2",
            "--top-k",
            "3",
            "--output",
            str(output_path),
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["generation"] is not None
    assert payload["generation"]["section_type"] == "methodology"
    assert payload["generation"]["top_k_examples"] == 3
    assert payload["generation"]["prompt_length"] > 0
    assert payload["generation"]["draft_length"] > 0
    # AC3 explicitly demands the model + runtime in the output —
    # check the runtime fingerprint is fully populated.
    runtime = payload["runtime"]
    assert runtime["runtime"], "AC3 requires runtime label"
    assert runtime["llm_model"], "AC3 requires llm_model"
    assert runtime["embedder"], "AC3 requires embedder identifier"


# ---------------------------------------------------------------------------
# Atomic-write contract — the .tmp file must not survive a successful run.
# ---------------------------------------------------------------------------


def test_benchmark_output_does_not_leave_tmp_file(
    isolated_workspace: Path,
) -> None:
    """After a successful run only the target file exists, no sibling .tmp."""

    runner = CliRunner()
    output_path = isolated_workspace / "report.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "all",
            "--proposals",
            "2",
            "--output",
            str(output_path),
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output

    sibling_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    assert output_path.exists()
    assert not sibling_tmp.exists(), "atomic write must leave no .tmp behind"


def test_benchmark_creates_missing_output_directory(
    isolated_workspace: Path,
) -> None:
    """``--output`` into a nested non-existent directory creates the parents."""

    runner = CliRunner()
    nested_output = isolated_workspace / "reports" / "nested" / "report.json"
    assert not nested_output.parent.exists()

    result = runner.invoke(
        app,
        [
            "benchmark",
            "all",
            "--proposals",
            "2",
            "--output",
            str(nested_output),
            "--config",
            str(isolated_workspace / "config.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert nested_output.exists()


# ---------------------------------------------------------------------------
# --help wiring — confirms the sub-app is mounted at ``eurpe benchmark``
# ---------------------------------------------------------------------------


def test_benchmark_subapp_is_discoverable() -> None:
    """``eurpe benchmark --help`` lists every sub-command."""

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    # The four sub-commands must be discoverable to satisfy the
    # operator-facing wording in the ACs.
    for name in ("all", "indexing", "retrieval", "generation"):
        assert name in result.output, f"missing sub-command {name!r} in --help"


def test_top_level_help_lists_benchmark() -> None:
    """The top-level CLI surface mounts ``benchmark`` alongside the others."""

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.output
