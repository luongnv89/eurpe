"""Tests for ``eurpe.intake.extractor``.

Two layers, mirroring ``test_docling_parser.py``:

* **Fast tests** — exercise the pure-Python text extractor against the
  synthetic ``tests/fixtures/topics/sample_topic.txt`` fixture and
  against handcrafted edge cases.
* **Slow tests** (``@pytest.mark.docling``) — generate a synthetic
  one-page PDF with reportlab, parse it with the real
  :class:`DoclingProposalParser`, and confirm the PDF path recovers
  the same fields as the text path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eurpe.intake import (
    TopicSource,
    extract_topic_context_from_pdf,
    extract_topic_context_from_text,
)
from eurpe.schema import Programme, SectionType

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "topics" / "sample_topic.txt"


def _load_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fast tests — text extractor.
# ---------------------------------------------------------------------------


def test_extract_text_recovers_programme_call_topic() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert ctx.programme is Programme.HORIZON_EUROPE
    assert ctx.call_id == "HORIZON-CL3-2024-CS-01"
    assert ctx.topic_id == "952672"


def test_extract_text_recovers_topic_title() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert ctx.topic_title == "Resilient digital infrastructure for critical sectors"


def test_extract_text_recovers_expected_outcomes_as_bullets() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert len(ctx.expected_outcomes) == 3
    assert ctx.expected_outcomes[0] == (
        "Reduced mean-time-to-recover for critical-sector outages by 30%."
    )
    assert ctx.expected_outcomes[1] == ("New open standards for resilient communication protocols.")
    assert ctx.expected_outcomes[2] == "Trained workforce across 5+ EU member states."


def test_extract_text_recovers_scope() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert ctx.scope is not None
    assert "resilience of digital infrastructure" in ctx.scope


def test_extract_text_recovers_destination() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert ctx.destination is not None
    assert "Cluster 3" in ctx.destination


def test_extract_text_recovers_section_guidance_for_methodology() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert SectionType.METHODOLOGY in ctx.section_guidance
    assert "real-world pilots" in ctx.section_guidance[SectionType.METHODOLOGY]


def test_extract_text_default_source_is_pasted_text() -> None:
    ctx = extract_topic_context_from_text(_load_fixture())
    assert ctx.source is TopicSource.PASTED_TEXT
    assert ctx.source_path is None


def test_extract_text_records_pdf_source_when_path_provided() -> None:
    """``source_path`` kwarg flips ``source`` to ``PDF_EXCERPT``.

    Mirrors how :func:`extract_topic_context_from_pdf` delegates back to
    the text extractor with the PDF path attached.
    """

    ctx = extract_topic_context_from_text(_load_fixture(), source_path="/tmp/example.pdf")
    assert ctx.source is TopicSource.PDF_EXCERPT
    assert ctx.source_path == "/tmp/example.pdf"


def test_extract_text_preserves_normalised_raw_text() -> None:
    """``raw_text`` carries the normalised input for audit replay."""

    ctx = extract_topic_context_from_text(_load_fixture())
    assert ctx.raw_text.startswith("HORIZON-CL3-2024-CS-01")
    # CRLF was normalised to LF.
    assert "\r" not in ctx.raw_text


def test_extract_text_normalises_crlf() -> None:
    text = "Topic: 952672\r\nTopic title: Foo\r\n"
    ctx = extract_topic_context_from_text(text)
    assert "\r" not in ctx.raw_text
    assert ctx.topic_id == "952672"
    assert ctx.topic_title == "Foo"


def test_extract_text_empty_input_yields_empty_context() -> None:
    """Empty input → empty record (best-effort intake, no raise)."""

    ctx = extract_topic_context_from_text("")
    assert ctx.is_empty() is True
    assert ctx.raw_text == ""
    assert ctx.source is TopicSource.PASTED_TEXT
    assert ctx.programme is None
    assert ctx.call_id is None
    assert ctx.topic_id is None
    assert ctx.expected_outcomes == []
    assert ctx.section_guidance == {}


def test_extract_text_does_not_use_topic_id_as_title() -> None:
    """A bare ``Topic: <digits>`` line must not leak the ID into topic_title.

    Without this guard the prompt would render ``**Topic title:** 952672``
    and the LLM would treat the numeric ID as the human-readable title.
    """

    ctx = extract_topic_context_from_text("Topic: 952672\nExpected Outcomes:\n- foo\n")
    assert ctx.topic_id == "952672"
    assert ctx.topic_title is None


def test_extract_text_outcomes_with_no_bullets_split_on_blank_lines() -> None:
    """A heading block with no bullet markers splits paragraphs."""

    text = (
        "Expected Outcomes:\n"
        "First paragraph outcome that spans\n"
        "two lines.\n"
        "\n"
        "Second paragraph outcome.\n"
        "\n"
        "Scope:\n"
        "something\n"
    )
    ctx = extract_topic_context_from_text(text)
    assert len(ctx.expected_outcomes) == 2
    assert ctx.expected_outcomes[0].startswith("First paragraph outcome")
    assert ctx.expected_outcomes[1] == "Second paragraph outcome."


def test_extract_text_handles_numbered_bullets() -> None:
    text = "Expected Outcomes:\n1. first\n2. second\n3. third\n\nScope: x\n"
    ctx = extract_topic_context_from_text(text)
    assert ctx.expected_outcomes == ["first", "second", "third"]


def test_extract_text_no_section_guidance_when_heading_missing() -> None:
    """Absent ``Methodology guidance:`` heading → empty dict (no infer)."""

    text = "Topic: 952672\nExpected Outcomes:\n- one\nScope:\nx\n"
    ctx = extract_topic_context_from_text(text)
    assert ctx.section_guidance == {}


# ---------------------------------------------------------------------------
# Fast tests — PDF extractor with a stubbed parser.
# ---------------------------------------------------------------------------


class _StubParsedSection:
    """Mimics :class:`ParsedSection` for the stub parser path."""

    def __init__(self, heading: str, text: str) -> None:
        self.heading = heading
        self.text = text


class _StubParsedProposal:
    """Mimics :class:`ParsedProposal` for the stub parser path."""

    def __init__(self, title: str | None, sections: list[_StubParsedSection]) -> None:
        self.title = title
        self.sections = sections


class _StubParser:
    """Parser-shaped stub: a ``parse(Path)`` method returning a parsed object.

    Used to test :func:`extract_topic_context_from_pdf` without touching
    the real Docling import path.
    """

    def __init__(self, parsed: _StubParsedProposal) -> None:
        self._parsed = parsed
        self.parse_calls: list[Path] = []

    def parse(self, path: Path) -> Any:
        self.parse_calls.append(path)
        return self._parsed


def test_extract_pdf_delegates_to_text_extractor_via_stub(tmp_path: Path) -> None:
    """A stubbed parser → flat-text concatenation → topic context shape.

    The Docling parser models a section as ``heading=...`` plus
    ``text=...``; the section heading is NOT repeated inside ``text``.
    The extractor concatenates ``"<heading>\\n<text>"`` for each
    section, so a heading-anchored capture downstream sees the
    heading line followed by body lines (which is what real
    Work-Programme topic PDFs produce).
    """

    parsed = _StubParsedProposal(
        title="HORIZON-CL3-2024-CS-01 — Resilient digital infrastructure",
        sections=[
            _StubParsedSection(
                heading="Programme",
                text=(
                    "Call: HORIZON-CL3-2024-CS-01\n"
                    "Topic: 952672\n"
                    "Topic title: Resilient digital infrastructure for critical sectors"
                ),
            ),
            _StubParsedSection(
                heading="Expected Outcomes",
                text=(
                    "- Reduced mean-time-to-recover for critical-sector outages by 30%.\n"
                    "- New open standards for resilient communication protocols.\n"
                    "- Trained workforce across 5+ EU member states."
                ),
            ),
            _StubParsedSection(
                heading="Scope",
                text=("Proposals should address the resilience of digital infrastructure."),
            ),
        ],
    )
    stub = _StubParser(parsed)
    pdf_path = tmp_path / "topic.pdf"
    pdf_path.write_bytes(b"not really a pdf")

    ctx = extract_topic_context_from_pdf(pdf_path, parser=stub)

    assert stub.parse_calls == [pdf_path]
    assert ctx.source is TopicSource.PDF_EXCERPT
    assert ctx.source_path == str(pdf_path)
    assert ctx.topic_id == "952672"
    assert ctx.topic_title == "Resilient digital infrastructure for critical sectors"
    assert len(ctx.expected_outcomes) == 3
    assert ctx.scope is not None and "resilience" in ctx.scope


# ---------------------------------------------------------------------------
# Slow tests — require Docling + reportlab.
# ---------------------------------------------------------------------------


def _build_synthetic_topic_pdf(path: Path) -> None:
    """Render the sample topic text into a one-page PDF via reportlab.

    Uses the same low-level canvas pattern as
    ``tests/test_docling_parser.py::_build_synthetic_pdf`` so Docling
    sees the text as positioned spans rather than relying on a real
    layout pipeline.
    """

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 760, "HORIZON-CL3-2024-CS-01")
    c.setFont("Helvetica", 11)
    c.drawString(72, 740, "Programme: Horizon Europe")
    c.drawString(72, 725, "Call: HORIZON-CL3-2024-CS-01")
    c.drawString(72, 710, "Topic: 952672")
    c.drawString(72, 695, "Topic title: Resilient digital infrastructure for critical sectors")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 670, "Expected Outcomes")
    c.setFont("Helvetica", 11)
    c.drawString(72, 650, "- Reduced mean-time-to-recover by 30%.")
    c.drawString(72, 635, "- New open standards for protocols.")
    c.drawString(72, 620, "- Trained workforce across 5+ EU member states.")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 590, "Scope")
    c.setFont("Helvetica", 11)
    c.drawString(72, 570, "Proposals should address the resilience of digital infrastructure.")
    c.showPage()
    c.save()


@pytest.mark.docling
def test_extract_pdf_real_docling_recovers_topic_fields(tmp_path: Path) -> None:
    """End-to-end: synthetic PDF → Docling → :class:`TopicContext`.

    What this test pins:

    * The pipeline runs end-to-end without raising (proves the lazy
      Docling import path is wired correctly and offline mode works).
    * The returned record carries ``source=PDF_EXCERPT`` and
      ``source_path`` set to the absolute file path.
    * Programme parsing fires off the call-id-bearing title line
      Docling reliably recovers.
    * ``raw_text`` is non-empty (Docling found body content).

    Field-level fidelity (every outcome bullet round-tripped) is
    deliberately NOT asserted here: Docling's structural parser drops
    the bulk of body text on tightly-positioned reportlab canvases,
    so individual ``Topic: NNN`` line recovery is unreliable. The
    real-PDF round-trip is exercised end-to-end in
    ``tests/e2e`` with proper Work-Programme excerpts.

    Skips cleanly when reportlab is absent (dev-only dep). The
    ``docling`` marker means CI can opt out via ``-m 'not docling'``.
    """

    pytest.importorskip(
        "reportlab",
        reason="reportlab not installed; synthetic PDF cannot be generated",
    )

    pdf_path = tmp_path / "topic.pdf"
    _build_synthetic_topic_pdf(pdf_path)

    ctx = extract_topic_context_from_pdf(pdf_path)

    assert ctx.source is TopicSource.PDF_EXCERPT
    assert ctx.source_path == str(pdf_path)
    # Programme is parsed off the call ID token at the top of the page —
    # Docling preserves the bold heading so this is reliable.
    assert ctx.programme is Programme.HORIZON_EUROPE
    # The raw_text must be non-empty — proves Docling extracted body
    # text and the extractor saw it.
    assert ctx.raw_text
