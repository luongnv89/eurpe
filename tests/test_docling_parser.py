"""Tests for ``eurpe.ingestion.docling_parser``.

Split into two layers:

* **Fast tests** (no Docling import) — exercise the cheap branches of
  :class:`DoclingProposalParser`: extension support, missing files,
  unsupported formats. These run on every CI invocation.
* **Slow tests** (``@pytest.mark.docling``) — generate a synthetic PDF
  via :mod:`reportlab` (skipped if reportlab isn't installed), run the
  real parser end-to-end, and assert the ParsedProposal contract holds.
  CI may opt out via ``-m "not docling"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eurpe.ingestion import (
    DoclingProposalParser,
    ParsedProposal,
    ParserError,
    UnsupportedFormatError,
)

# ---------------------------------------------------------------------------
# Fast tests — no Docling import required.
# ---------------------------------------------------------------------------


def test_supports_pdf_extension() -> None:
    parser = DoclingProposalParser()
    assert parser.supports(Path("/tmp/proposal.pdf")) is True
    # Case-insensitive: real-world filenames come in both casings.
    assert parser.supports(Path("/tmp/PROPOSAL.PDF")) is True
    assert parser.supports(Path("/tmp/Mixed.Pdf")) is True


def test_supports_rejects_non_pdf_extensions() -> None:
    parser = DoclingProposalParser()
    for name in ("a.docx", "b.txt", "c.html", "d", "e.pdfx"):
        assert parser.supports(Path(f"/tmp/{name}")) is False, name


def test_parse_unsupported_extension_raises_unsupported_format(tmp_path: Path) -> None:
    parser = DoclingProposalParser()
    bad = tmp_path / "report.docx"
    bad.write_text("pretend content", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError) as excinfo:
        parser.parse(bad)
    assert ".docx" in str(excinfo.value)


def test_parse_missing_file_raises_parser_error(tmp_path: Path) -> None:
    """Missing-file errors must surface as ``ParserError`` carrying the path."""

    parser = DoclingProposalParser()
    missing = tmp_path / "nope.pdf"
    with pytest.raises(ParserError) as excinfo:
        parser.parse(missing)
    err = excinfo.value
    assert err.source_path == str(missing)
    assert "file not found" in str(err)


def test_ocr_enabled_default_is_false() -> None:
    parser = DoclingProposalParser()
    assert parser.ocr_enabled is False


def test_ocr_enabled_true_when_constructed_with_ocr() -> None:
    # Must opt out of offline mode to combine with OCR — the offline
    # contract treats ``offline=True + ocr_enabled=True`` as a fail-fast
    # ValueError (see ``test_parser_offline_with_ocr_enabled_raises``).
    parser = DoclingProposalParser(offline=False, ocr_enabled=True)
    assert parser.ocr_enabled is True


def test_parser_offline_default_is_true() -> None:
    """Offline mode must be ON by default — the PRD's no-network invariant."""

    parser = DoclingProposalParser()
    assert parser.offline is True


def test_parser_offline_mode_disables_ocr() -> None:
    """``offline=True`` must resolve ``do_ocr`` to False without importing Docling.

    Asserting on the parser's resolved boolean (rather than constructing
    a real ``PdfPipelineOptions``) keeps this test in the fast suite —
    no Docling import, no model download, no marker required. The
    resolved boolean is what the parser eventually passes to
    ``PdfPipelineOptions(do_ocr=...)`` inside ``parse()``.
    """

    parser = DoclingProposalParser(offline=True)
    assert parser.do_ocr is False
    # Even if a caller passes ``ocr_enabled=False`` explicitly while
    # offline, the resolved flag is still False (defence in depth).
    parser2 = DoclingProposalParser(offline=True, ocr_enabled=False)
    assert parser2.do_ocr is False
    # And ``offline=False, ocr_enabled=False`` keeps OCR off — only
    # explicitly enabling OCR while offline=False turns it on.
    parser3 = DoclingProposalParser(offline=False, ocr_enabled=False)
    assert parser3.do_ocr is False
    parser4 = DoclingProposalParser(offline=False, ocr_enabled=True)
    assert parser4.do_ocr is True


def test_parser_offline_with_ocr_enabled_raises() -> None:
    """``offline=True`` + ``ocr_enabled=True`` must fail fast at construction.

    Silently disabling OCR would surprise callers that set the flag
    intentionally; silently downloading weights would break the offline
    contract. ValueError is the right "you misconfigured this" signal.
    """

    with pytest.raises(ValueError, match="OCR"):
        DoclingProposalParser(offline=True, ocr_enabled=True)


def test_importing_ingestion_does_not_import_docling() -> None:
    """The lazy-import pattern is the whole reason the parser is wrapped.

    Confirm that ``import eurpe.ingestion`` does not pull docling into
    ``sys.modules``. Run in a subprocess so we get a clean Python without
    relying on whatever this pytest process has already imported, and so
    we cannot corrupt other tests' module state by reaching into
    ``sys.modules``.
    """

    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import eurpe.ingestion; print('docling' in sys.modules)"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    # The subprocess prints "True" or "False" for whether ``docling`` was
    # imported as a side effect. We require False — anything else means
    # the lazy-import pattern is broken and ``eurpe smoke`` will slow down.
    assert proc.stdout.strip() == "False", (
        "import eurpe.ingestion pulled docling into sys.modules — "
        "the lazy-import pattern is broken; smoke command will slow down."
    )


# ---------------------------------------------------------------------------
# Slow tests — require Docling and (for the synthetic PDF) reportlab.
# ---------------------------------------------------------------------------


def _build_synthetic_pdf(path: Path) -> None:
    """Create a tiny one-page PDF with a title, a heading, body text, and a table.

    Uses ReportLab's low-level canvas rather than Platypus so the file
    structure stays predictable: Docling sees discrete text-positioned
    spans which it can classify as headings / body without a real layout
    pipeline.
    """

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, 720, "Test Proposal")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 670, "1. Introduction")
    c.setFont("Helvetica", 12)
    c.drawString(72, 640, "This is the introduction body text for the proposal document.")
    c.drawString(72, 620, "It spans two lines so we have something meaningful to extract.")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 580, "1.1 Scope")
    c.setFont("Helvetica", 12)
    c.drawString(72, 550, "Scope details follow.")
    # A simple two-row, two-column "table" (positioned text).
    rows = [["Header A", "Header B"], ["cell 1", "cell 2"]]
    y = 480
    for row in rows:
        for i, cell in enumerate(row):
            c.drawString(72 + i * 150, y, cell)
        y -= 20
    c.showPage()
    c.save()


@pytest.mark.docling
def test_parse_real_pdf_returns_parsed_proposal(tmp_path: Path) -> None:
    """End-to-end Docling parse of a synthetic one-page PDF.

    Asserts the structural acceptance criteria: title is preserved, at
    least one section is recovered, and the heading text is recoverable.
    Skips cleanly if reportlab isn't installed (it's a [dev]-only dep).
    """

    pytest.importorskip(
        "reportlab",
        reason="reportlab not installed; synthetic PDF cannot be generated",
    )

    pdf_path = tmp_path / "synthetic.pdf"
    _build_synthetic_pdf(pdf_path)

    parser = DoclingProposalParser()
    parsed = parser.parse(pdf_path)

    assert isinstance(parsed, ParsedProposal)
    assert parsed.parser == "docling"
    # Title falls back to the first H1 (or doc.name); either way it's
    # non-empty for our synthetic doc which has a clear top-line title.
    assert parsed.title, "title should be non-empty for a document with a clear top-line"
    # At least one section was recovered.
    assert len(parsed.sections) >= 1
    # The "1. Introduction" heading must be recoverable verbatim.
    headings = [s.heading for s in parsed.sections]
    assert any("Introduction" in h for h in headings), (
        f"expected an 'Introduction' heading; got {headings}"
    )
    # Page count is at least 1 for our one-page synthetic doc.
    assert parsed.page_count is not None and parsed.page_count >= 1
    # Total text length must be non-zero or we extracted nothing.
    assert parsed.total_text_length() > 0
    # Source path must round-trip to the absolute form Docling was given.
    assert parsed.source_path == str(pdf_path.resolve())


@pytest.mark.docling
def test_parse_real_pdf_via_cli_writes_json(tmp_path: Path) -> None:
    """``eurpe ingest <pdf> --output <dir>`` writes a parsed.json on success."""

    pytest.importorskip(
        "reportlab",
        reason="reportlab not installed; synthetic PDF cannot be generated",
    )

    from typer.testing import CliRunner

    from eurpe.cli import app

    pdf_path = tmp_path / "synthetic-cli.pdf"
    _build_synthetic_pdf(pdf_path)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["ingest", str(pdf_path), "--output", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "Parsed proposal summary" in result.output
    out_file = out_dir / "synthetic-cli.parsed.json"
    assert out_file.exists()
    # JSON content must round-trip into a ParsedProposal.
    payload = out_file.read_text(encoding="utf-8")
    reloaded = ParsedProposal.model_validate_json(payload)
    assert reloaded.parser == "docling"


def test_cli_reports_parser_error_on_missing_file(tmp_path: Path) -> None:
    """The CLI must exit non-zero and not write any output for a missing PDF."""

    from typer.testing import CliRunner

    from eurpe.cli import app

    missing = tmp_path / "missing.pdf"
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["ingest", str(missing), "--output", str(out_dir)],
    )
    assert result.exit_code == 1, result.output
    # The output dir must not have been created — this is the
    # "no partial state on failure" acceptance criterion.
    assert not out_dir.exists() or not any(out_dir.iterdir()), (
        f"output dir should be empty/missing on failure, got: "
        f"{list(out_dir.iterdir()) if out_dir.exists() else 'absent'}"
    )


def test_cli_rejects_unsupported_extension(tmp_path: Path) -> None:
    """``eurpe ingest`` on a non-PDF must exit 1 with an unsupported-format message."""

    from typer.testing import CliRunner

    from eurpe.cli import app

    bogus = tmp_path / "report.docx"
    bogus.write_text("pretend content", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(bogus)])
    assert result.exit_code == 1, result.output
    assert "unsupported format" in result.output.lower()


@pytest.mark.docling
def test_cli_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    """Without ``--overwrite``, the CLI must refuse to clobber an existing parse.

    Marked ``docling`` because we need a real parse to produce the
    second-run write attempt — the ``--overwrite`` gate sits *after*
    a successful parse (we only refuse to write, we don't refuse to
    parse). If the gate were earlier we could test it without Docling.
    """

    pytest.importorskip(
        "reportlab",
        reason="reportlab not installed; synthetic PDF cannot be generated",
    )

    from typer.testing import CliRunner

    from eurpe.cli import app

    pdf_path = tmp_path / "synthetic-overwrite.pdf"
    _build_synthetic_pdf(pdf_path)
    out_dir = tmp_path / "out"

    runner = CliRunner()

    # First run writes the file successfully.
    result1 = runner.invoke(app, ["ingest", str(pdf_path), "--output", str(out_dir)])
    assert result1.exit_code == 0, result1.output
    out_file = out_dir / "synthetic-overwrite.parsed.json"
    assert out_file.exists()
    original_bytes = out_file.read_bytes()

    # Second run must refuse — the existing file would otherwise be
    # silently clobbered.
    result2 = runner.invoke(app, ["ingest", str(pdf_path), "--output", str(out_dir)])
    assert result2.exit_code == 1, result2.output
    assert "already exists" in result2.output.lower()
    # The on-disk bytes must be unchanged (no half-written .tmp left
    # behind either).
    assert out_file.read_bytes() == original_bytes
    assert not (out_dir / "synthetic-overwrite.parsed.json.tmp").exists()


@pytest.mark.docling
def test_cli_overwrite_flag_replaces_existing_output(tmp_path: Path) -> None:
    """With ``--overwrite``/``-f``, the CLI must replace the existing parse.

    Atomic semantics: the target file always exists during the rename
    (no transient empty file), and any sibling ``.tmp`` is gone after
    the run.
    """

    pytest.importorskip(
        "reportlab",
        reason="reportlab not installed; synthetic PDF cannot be generated",
    )

    from typer.testing import CliRunner

    from eurpe.cli import app

    pdf_path = tmp_path / "synthetic-overwrite-allowed.pdf"
    _build_synthetic_pdf(pdf_path)
    out_dir = tmp_path / "out"
    out_file = out_dir / "synthetic-overwrite-allowed.parsed.json"

    runner = CliRunner()

    # Seed an obviously-bogus prior file so we can confirm the second
    # run actually replaces the bytes, rather than accidentally leaving
    # the seed in place.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text("OLD-CONTENT", encoding="utf-8")

    result = runner.invoke(
        app,
        ["ingest", str(pdf_path), "--output", str(out_dir), "--overwrite"],
    )
    assert result.exit_code == 0, result.output
    body = out_file.read_text(encoding="utf-8")
    assert body != "OLD-CONTENT"
    # Round-trips into a ParsedProposal — i.e. it's a real new parse,
    # not just the seed appended to.
    reloaded = ParsedProposal.model_validate_json(body)
    assert reloaded.parser == "docling"
    assert not (out_dir / "synthetic-overwrite-allowed.parsed.json.tmp").exists()
