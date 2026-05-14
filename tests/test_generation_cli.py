"""Tests for ``eurpe.generation.cli``.

Drives the ``eurpe generate section`` Typer command end-to-end with
the fixture corpus and the deterministic LLM stub. Mirrors the
fast-test pattern used in ``test_retrieval_cli.py`` — same offline
config helper, same in-memory chunk fixtures.

Includes the AC3 sanity check that the CLI runs with no Ollama
reachable: the test config points at an unreachable Ollama URL so
the factory falls back to the deterministic stub.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from eurpe.cli import app
from eurpe.retrieval import ChromaIndex, DeterministicHashEmbedder
from tests._chunk_helpers import build_fixture_chunks


def _write_offline_config(tmp_path: Path) -> Path:
    """Write a config.yaml that pins paths under tmp_path and disables Ollama.

    Same pattern as ``test_retrieval_cli._write_offline_config`` —
    pointing at an unreachable Ollama port forces the factory to fall
    back to the deterministic stub, which is exactly what we want for
    fast tests.
    """

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "corpus_path": str(tmp_path / "corpus"),
                "index_path": str(tmp_path / "index"),
                "models": {
                    "runtime": "ollama",
                    "llm_model": "llama3.1:8b",
                    "embedding_model": "nomic-embed-text",
                    "ollama_base_url": "http://localhost:1",  # unreachable
                },
                "offline_mode": True,
                "log_level": "INFO",
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def _seed_index_with_fixtures(tmp_path: Path) -> int:
    """Populate the index at ``tmp_path/index`` with the fixture chunks."""

    embedder = DeterministicHashEmbedder(dimension=384)
    index = ChromaIndex(
        index_path=tmp_path / "index",
        embedder=embedder,
        collection_name="default",
    )
    chunks = build_fixture_chunks()
    index.upsert(chunks)
    return len(chunks)


def test_generate_section_cli_produces_draft(tmp_path: Path) -> None:
    """Happy path: --type + --intent yield a draft + a citations table."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Describe our deep learning approach for methodology",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Generated draft" in result.output
    assert "Citations" in result.output


def test_generate_section_cli_runs_with_unreachable_ollama_offline(tmp_path: Path) -> None:
    """AC3 sanity check at the CLI layer: offline + unreachable Ollama still works.

    The config points at ``http://localhost:1`` which never has a
    listener; the factory must fall back to DeterministicLLMClient.
    """

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Methodology draft",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # The deterministic stub identifies itself in the "Generating..." line.
    assert "deterministic-stub-v1" in result.output


def test_generate_section_cli_with_empty_index(tmp_path: Path) -> None:
    """Empty index → CLI still emits a draft + a "no citations" notice."""

    cfg_path = _write_offline_config(tmp_path)
    # Don't seed anything — the index will be empty.

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Methodology draft",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Generated draft" in result.output
    assert "(none — no evidence retrieved)" in result.output


def test_generate_section_cli_writes_output_atomically(tmp_path: Path) -> None:
    """``--output`` writes a valid JSON dump of the GenerationDraft."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    out_path = tmp_path / "drafts" / "methodology.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--output",
            str(out_path),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["section_type"] == "methodology"
    assert payload["text"]
    assert payload["model"] == "deterministic-stub-v1"
    assert "request" in payload
    # No leftover .tmp file from the atomic write.
    assert not out_path.with_suffix(out_path.suffix + ".tmp").exists()


def test_generate_section_cli_refuses_to_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    """Without ``--overwrite``, an existing output file is preserved."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    out_path = tmp_path / "draft.json"
    out_path.write_text("EXISTING CONTENT", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--output",
            str(out_path),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 1
    assert "already exists" in result.output
    # The original file was preserved.
    assert out_path.read_text(encoding="utf-8") == "EXISTING CONTENT"


def test_generate_section_cli_overwrite_flag_replaces_output(tmp_path: Path) -> None:
    """``--overwrite`` allows clobbering an existing output file."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    out_path = tmp_path / "draft.json"
    out_path.write_text("EXISTING CONTENT", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--output",
            str(out_path),
            "--overwrite",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["section_type"] == "methodology"


def test_generate_section_cli_supports_context_from_file(tmp_path: Path) -> None:
    """``--context @path`` reads call context from disk and injects it into the prompt."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    ctx_path = tmp_path / "call_topic.md"
    ctx_path.write_text(
        "Topic: federated learning for cyber-physical resilience.",
        encoding="utf-8",
    )
    out_path = tmp_path / "draft.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Methodology draft",
            "--context",
            f"@{ctx_path}",
            "--threshold",
            "0.0",
            "--output",
            str(out_path),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    # The file content lands in the prompt under the call context block.
    assert "federated learning for cyber-physical resilience" in payload["prompt_used"]


def test_generate_section_cli_rejects_invalid_section_type(tmp_path: Path) -> None:
    """An unknown ``--type`` value exits non-zero with a helpful message."""

    cfg_path = _write_offline_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "not-a-real-section",
            "--intent",
            "x",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 1
    assert "--type" in result.output


def test_generate_section_cli_rejects_invalid_programme(tmp_path: Path) -> None:
    """An unknown ``--programme`` value exits non-zero with a helpful message."""

    cfg_path = _write_offline_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "x",
            "--programme",
            "not-a-real-programme",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 1
    assert "--programme" in result.output


def test_generate_section_cli_lessons_learned_flag(tmp_path: Path) -> None:
    """``--lessons-learned`` runs without error (observable behaviour is in workflow tests)."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Methodology draft",
            "--lessons-learned",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_generate_section_cli_no_esr_flag(tmp_path: Path) -> None:
    """``--no-esr`` runs without error and excludes ESR notes from citations."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    out_path = tmp_path / "draft.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "excellence",
            "--intent",
            "Frame the excellence section",
            "--no-esr",
            "--threshold",
            "0.0",
            "--output",
            str(out_path),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    statuses = {c["source_status"] for c in payload["citations"]}
    # ESR notes were excluded from retrieval — no citation should carry that label.
    assert "esr_note" not in statuses


def test_generate_section_cli_top_k_caps_evidence(tmp_path: Path) -> None:
    """``--top-k`` limits the number of citations on the resulting draft."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    out_path = tmp_path / "draft.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Methodology draft",
            "--top-k",
            "2",
            "--threshold",
            "0.0",
            "--output",
            str(out_path),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["citations"]) <= 2


def test_generate_section_cli_context_with_missing_file_errors(tmp_path: Path) -> None:
    """``--context @missing-file`` surfaces a clean error from the helper."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            "methodology",
            "--intent",
            "Methodology draft",
            "--context",
            f"@{tmp_path / 'does_not_exist.md'}",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output
