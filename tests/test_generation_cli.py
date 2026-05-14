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
    """Happy path: --type + --intent yield a rendered Markdown draft.

    The default ``--render`` is ``both``; on stdout that prints the
    rendered Markdown form (heading + body + references). The JSON
    form is written under ``--output`` only.
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
            "Describe our deep learning approach for methodology",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Markdown rendering: section heading + references block.
    assert "# Methodology" in result.output
    assert "## References" in result.output


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
    """Empty index → CLI still emits a draft + a "no citations" notice.

    Under the default ``--render both`` the rendered Markdown is what
    lands on stdout; the renderer's "no citations" placeholder is
    ``_No citations._`` (italic Markdown so a casual reader can't miss
    it).
    """

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
    assert "# Methodology" in result.output
    assert "_No citations._" in result.output


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


# ---------------------------------------------------------------------------
# Issue #7 — render mode + audit subcommand
# ---------------------------------------------------------------------------


def _write_clean_draft(tmp_path: Path) -> Path:
    """Write a hand-crafted clean GenerationDraft JSON to disk.

    Used by the ``audit`` subcommand tests so they don't depend on
    re-running the workflow. The shape mirrors what
    ``GenerationDraft.model_dump_json`` produces.
    """

    payload = {
        "section_type": "methodology",
        "text": "We propose [1].",
        "citations": [
            {
                "citation_id": 1,
                "source_status": "funded",
                "programme": "horizon_europe",
                "call_id": "HORIZON-CL5-2024-D3-02",
                "proposal_title": "Test Proposal",
                "section_heading": "Methodology",
                "page": 12,
                "chunk_id": "chunk-1",
                "snippet": "snippet text",
            }
        ],
        "prompt_used": "(elided)",
        "model": "deterministic-stub-v1",
        "request": {
            "section_type": "methodology",
            "user_intent": "test",
            "call_context": "",
            "target_programme": None,
            "top_k_examples": 5,
            "lessons_learned": False,
        },
    }
    path = tmp_path / "clean-draft.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_dirty_draft(tmp_path: Path) -> Path:
    """Write a draft whose text contains a marker [99] that has no citation."""

    payload = {
        "section_type": "methodology",
        "text": "We propose [1] and also [99].",
        "citations": [
            {
                "citation_id": 1,
                "source_status": "funded",
                "programme": "horizon_europe",
                "call_id": "HORIZON-CL5-2024-D3-02",
                "proposal_title": "Test Proposal",
                "section_heading": "Methodology",
                "page": 12,
                "chunk_id": "chunk-1",
                "snippet": "snippet text",
            }
        ],
        "prompt_used": "(elided)",
        "model": "deterministic-stub-v1",
        "request": {
            "section_type": "methodology",
            "user_intent": "test",
            "call_context": "",
            "target_programme": None,
            "top_k_examples": 5,
            "lessons_learned": False,
        },
    }
    path = tmp_path / "dirty-draft.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_cli_renders_markdown_when_render_markdown(tmp_path: Path) -> None:
    """``--render markdown`` emits a Markdown document with a ``## References`` block."""

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
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--render",
            "markdown",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## References" in result.output
    # The plaintext "Generated draft" header used by the json branch
    # must NOT appear when --render=markdown.
    assert "Generated draft" not in result.output


def test_cli_renders_json_summary_when_render_json(tmp_path: Path) -> None:
    """``--render json`` emits the plaintext draft summary, not Markdown."""

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
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--render",
            "json",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Generated draft" in result.output
    # Markdown heading marker must not leak into json mode.
    assert "## References" not in result.output


def test_cli_writes_both_md_and_json(tmp_path: Path) -> None:
    """``--render both --output base`` produces both ``base.md`` and ``base.json``."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)
    out_base = tmp_path / "drafts" / "methodology"

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
            "--render",
            "both",
            "--output",
            str(out_base),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output

    md_path = out_base.with_suffix(".md")
    json_path = out_base.with_suffix(".json")
    assert md_path.exists()
    assert json_path.exists()

    # Markdown file is a valid rendered document.
    md_text = md_path.read_text(encoding="utf-8")
    assert "# Methodology" in md_text
    assert "## References" in md_text

    # JSON file is a valid GenerationDraft.
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["section_type"] == "methodology"
    assert payload["model"] == "deterministic-stub-v1"


def test_cli_audit_subcommand_exits_0_on_clean_draft(tmp_path: Path) -> None:
    """``eurpe generate audit clean.json`` → exit 0."""

    draft_path = _write_clean_draft(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["generate", "audit", str(draft_path)])

    assert result.exit_code == 0, result.output
    # The summary line carries the file path so the operator can tell
    # which draft was checked when piping.
    assert "Audit summary" in result.output


def test_cli_audit_subcommand_exits_1_on_dirty_draft(tmp_path: Path) -> None:
    """``eurpe generate audit dirty.json`` → exit 1, mentions the failure code."""

    draft_path = _write_dirty_draft(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["generate", "audit", str(draft_path)])

    assert result.exit_code == 1, result.output
    # The audit prints findings to stderr; CliRunner captures both
    # streams in result.output by default.
    assert "marker_without_citation" in result.output


def test_cli_audit_subcommand_rejects_invalid_json(tmp_path: Path) -> None:
    """A malformed JSON file → exit 1 with a clean error."""

    draft_path = tmp_path / "broken.json"
    draft_path.write_text("not-json{", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["generate", "audit", str(draft_path)])

    assert result.exit_code == 1, result.output
    assert "not valid JSON" in result.output


def test_cli_section_runs_audit_by_default(tmp_path: Path) -> None:
    """Audit runs after generation by default; passing run prints the summary."""

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
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # The success summary is printed to stderr by ``_print_audit_findings``.
    assert "Audit:" in result.output


def test_cli_no_audit_flag_skips_audit(tmp_path: Path) -> None:
    """``--no-audit`` suppresses the post-generation audit summary."""

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
            "Describe our DL approach",
            "--threshold",
            "0.0",
            "--no-audit",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # No audit summary line when --no-audit is passed.
    assert "Audit:" not in result.output
    assert "Audit summary" not in result.output


def test_cli_rejects_invalid_render_mode(tmp_path: Path) -> None:
    """``--render bogus`` exits non-zero with a friendly message."""

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
            "x",
            "--render",
            "bogus",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "--render" in result.output
