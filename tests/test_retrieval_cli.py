"""Tests for ``eurpe.retrieval.cli``.

Two tiers:

* **Fast tests** — exercise the query path directly against an
  in-memory chunk fixture (no PDF parsing). Run on every install.
* **Slow tests** (``@pytest.mark.docling``) — generate a synthetic
  PDF via :mod:`reportlab`, write a sidecar YAML pointing at it, and
  drive ``eurpe index build`` end-to-end. Mirrors the pattern used in
  ``test_docling_parser.py``.

The CLI tests share Typer's :class:`CliRunner` invocation pattern so a
later refactor can centralise it without touching individual cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from eurpe.cli import app
from eurpe.retrieval import ChromaIndex, DeterministicHashEmbedder
from tests._chunk_helpers import build_fixture_chunks, query_text_for
from tests._helpers.metadata import write_metadata_yaml
from tests._helpers.offline import write_offline_config as _write_offline_config


def _seed_index_with_fixtures(tmp_path: Path, *, collection: str = "default") -> int:
    """Populate the index at ``tmp_path/index`` with the fixture chunks.

    Returns the chunk count so callers can assert on it. We use the
    deterministic embedder directly because the CLI's
    ``make_embedder`` would also pick it up (the config points at an
    unreachable Ollama port).
    """

    embedder = DeterministicHashEmbedder(dimension=384)
    index = ChromaIndex(
        index_path=tmp_path / "index",
        embedder=embedder,
        collection_name=collection,
    )
    chunks = build_fixture_chunks()
    index.upsert(chunks)
    return len(chunks)


def test_index_query_cli_returns_results_for_marker_token(tmp_path: Path) -> None:
    """``eurpe index query`` returns the funded fixture for its marker token.

    The DeterministicHashEmbedder produces modest cosine scores that
    sit below the policy's 0.30 default threshold, so we lower
    ``--threshold`` to exercise the retrieval path itself rather than
    accidentally test the threshold gate.
    """

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            query_text_for("funded_horizon_europe.yaml"),
            "--top-k",
            "3",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # The funded fixture's source-status label MUST appear in the output.
    assert "funded" in result.output


def test_index_query_cli_with_status_filter_excludes_non_matching(tmp_path: Path) -> None:
    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            "any text",
            "--top-k",
            "5",
            "--source-status",
            "rejected",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # With the rejected filter, no funded chunk should appear.
    assert "status=funded" not in result.output


def test_index_query_cli_shows_policy_reason_column(tmp_path: Path) -> None:
    """Output of ``eurpe index query`` includes the ``policy_reason=`` field."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            query_text_for("funded_horizon_europe.yaml"),
            "--top-k",
            "3",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "policy_reason=" in result.output
    # The funded marker query should land at least one funded_primary row.
    assert "policy_reason=funded_primary" in result.output


def test_index_query_cli_lessons_learned_flag_marks_rejected(tmp_path: Path) -> None:
    """``--lessons-learned`` surfaces rejected results with the lessons_learned reason.

    The fixture corpus has exactly one rejected chunk; under
    lessons-learned mode it should appear with ``policy_reason=lessons_learned_mode``
    even though the default policy with its rejected-fraction cap (0.4 of
    top_k=5 → 2 allowed) wouldn't normally privilege it.
    """

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            query_text_for("rejected_horizon_2020.yaml"),
            "--top-k",
            "5",
            "--threshold",
            "0.0",
            "--lessons-learned",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status=rejected" in result.output
    assert "policy_reason=lessons_learned_mode" in result.output


def test_index_query_cli_no_esr_excludes_esr(tmp_path: Path) -> None:
    """``--no-esr`` removes ESR notes from the result list."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            query_text_for("esr_note_horizon_europe.yaml"),
            "--top-k",
            "5",
            "--threshold",
            "0.0",
            "--no-esr",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status=esr_note" not in result.output


def test_index_query_cli_rejects_invalid_source_status(tmp_path: Path) -> None:
    """An unknown ``--source-status`` value exits non-zero with a helpful message."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            "anything",
            "--source-status",
            "not-a-real-status",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 1
    assert "--source-status" in result.output


def test_index_query_cli_handles_no_results_gracefully(tmp_path: Path) -> None:
    """Filtering for a status that has no chunks must not crash."""

    cfg_path = _write_offline_config(tmp_path)
    _seed_index_with_fixtures(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "query",
            "anything",
            "--source-status",
            "funded",
            # A programme that no fixture uses → empty intersection.
            "--programme",
            "horizon_2020",
            "--threshold",
            "0.0",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(no results)" in result.output


# ---------------------------------------------------------------------------
# Slow tests — require Docling + reportlab.
# ---------------------------------------------------------------------------


def _build_synthetic_pdf(path: Path) -> None:
    """Smaller cousin of the helper in ``test_docling_parser.py``.

    Duplicated rather than imported because the module-level import in
    that file would otherwise drag the whole docling-marker suite
    setup into this file's collection cost.
    """

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, 720, "Synthetic Test Proposal")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 670, "1. Excellence")
    c.setFont("Helvetica", 12)
    c.drawString(72, 640, "Excellence body text describing the proposal scientific ambition.")
    c.drawString(72, 620, "It includes a methodology paragraph and impact analysis.")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 580, "2. Impact")
    c.setFont("Helvetica", 12)
    c.drawString(72, 550, "Impact narrative for downstream commercialisation.")
    c.showPage()
    c.save()


def _write_metadata_yaml(yaml_path: Path, pdf_path: Path) -> None:
    """Minimal proposal-metadata YAML pointing at ``pdf_path``.

    Thin wrapper over :func:`tests._helpers.metadata.write_metadata_yaml`
    that pins the historical fixture values (call_id / topic_id /
    consortium acronym) so the existing assertions stay readable in
    context. The shared helper handles the file-write mechanics so
    this module no longer needs the ``yaml`` import for that purpose.
    """

    write_metadata_yaml(
        yaml_path,
        pdf_path,
        call_id="HORIZON-CL5-2024-D3-02",
        topic_id="HORIZON-CL5-2024-D3-02-01",
        proposal_title="Synthetic Test Proposal",
        consortium_acronym="STP",
    )


@pytest.mark.docling
def test_index_build_cli_end_to_end(tmp_path: Path) -> None:
    """``eurpe index build <yaml>`` populates the index from a real PDF.

    Skips if reportlab is not installed (it's a [dev]-only dep).
    Pins the integration: PDF → parser → chunker → embedder → upsert.
    """

    pytest.importorskip(
        "reportlab",
        reason="reportlab not installed; synthetic PDF cannot be generated",
    )

    cfg_path = _write_offline_config(tmp_path)
    pdf_path = tmp_path / "synthetic.pdf"
    _build_synthetic_pdf(pdf_path)
    yaml_path = tmp_path / "synthetic.yaml"
    _write_metadata_yaml(yaml_path, pdf_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "build",
            str(yaml_path),
            "--config",
            str(cfg_path),
            "--collection",
            "synthetic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ingested synthetic.yaml" in result.output

    # Direct count on the index — proves the upsert landed.
    embedder = DeterministicHashEmbedder(dimension=384)
    index = ChromaIndex(
        index_path=tmp_path / "index",
        embedder=embedder,
        collection_name="synthetic",
    )
    assert index.count() > 0


@pytest.mark.docling
def test_index_build_cli_rejects_missing_yaml(tmp_path: Path) -> None:
    """A missing sidecar must produce exit code 1, not a silent skip."""

    cfg_path = _write_offline_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            "build",
            str(tmp_path / "does_not_exist.yaml"),
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 1
    assert "metadata file not found" in result.output
