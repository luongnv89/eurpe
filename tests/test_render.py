"""Tests for ``eurpe.generation.render`` — Markdown rendering with status labels.

Pin Issue #7 acceptance criteria #1 (every citation gets a labelled
reference row) and #3 (funded and rejected render distinctly). The
remaining cases cover edge handling and the determinism property the
audit relies on.

The fabricated drafts here use the deterministic stub-style fields
straight from the Pydantic constructors so no LLM is invoked. Tests
that need a real workflow run live in ``test_generation_workflow.py``.
"""

from __future__ import annotations

from eurpe.generation import (
    STATUS_BADGE,
    STATUS_CAVEAT,
    STATUS_LABEL,
    CitationRef,
    GenerationDraft,
    GenerationRequest,
    MarkdownCitationRenderer,
)
from eurpe.schema import Programme, SectionType, SourceStatus

# ---------------------------------------------------------------------------
# Fixture builders — kept tiny, no I/O, no workflow.
# ---------------------------------------------------------------------------


def _make_citation(
    *,
    citation_id: int,
    status: SourceStatus,
    programme: Programme = Programme.HORIZON_EUROPE,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    proposal_title: str | None = "Sample Proposal",
    section_heading: str | None = "1.2 Methodology",
    page: int | None = 12,
    chunk_id: str | None = None,
    snippet: str = "This is a snippet from the source chunk.",
) -> CitationRef:
    """Construct a :class:`CitationRef` with sensible defaults.

    Each field is overridable so individual tests can vary one
    dimension at a time without re-typing the boilerplate.
    """

    return CitationRef(
        citation_id=citation_id,
        source_status=status,
        programme=programme,
        call_id=call_id,
        proposal_title=proposal_title,
        section_heading=section_heading,
        page=page,
        chunk_id=chunk_id or f"chunk-{citation_id}",
        snippet=snippet,
    )


def _make_draft(
    *,
    citations: list[CitationRef],
    text: str = "We propose [1] and learn from [2].",
    section_type: SectionType = SectionType.METHODOLOGY,
) -> GenerationDraft:
    """Build a :class:`GenerationDraft` from a citation list."""

    request = GenerationRequest(
        section_type=section_type,
        user_intent="Describe the approach.",
    )
    return GenerationDraft(
        section_type=section_type,
        text=text,
        citations=citations,
        prompt_used="(prompt elided in tests)",
        model="deterministic-stub-v1",
        request=request,
    )


# ---------------------------------------------------------------------------
# AC1 — every citation must surface source document, section, and status
# ---------------------------------------------------------------------------


def test_render_includes_section_document_status_for_each_citation() -> None:
    """AC1 keystone: every cited row carries source, section, and status label.

    Three citations of distinct statuses (FUNDED, REJECTED, ESR_NOTE)
    so we can assert each row independently. Parsing is deliberately
    crude (substring) to keep the test resilient to whitespace
    re-flow but strict enough to catch a missing field.
    """

    citations = [
        _make_citation(
            citation_id=1,
            status=SourceStatus.FUNDED,
            proposal_title="Funded Proposal A",
            section_heading="Methodology",
        ),
        _make_citation(
            citation_id=2,
            status=SourceStatus.REJECTED,
            proposal_title="Rejected Proposal B",
            section_heading="Implementation",
        ),
        _make_citation(
            citation_id=3,
            status=SourceStatus.ESR_NOTE,
            proposal_title="Reviewer Notes C",
            section_heading="Comments",
        ),
    ]
    draft = _make_draft(citations=citations, text="See [1], [2], and [3].")

    rendered = MarkdownCitationRenderer().render(draft)

    # The references section header must be present once and only once.
    assert rendered.count("## References") == 1

    # Every citation contributes the source title to the table row.
    assert "Funded Proposal A" in rendered
    assert "Rejected Proposal B" in rendered
    assert "Reviewer Notes C" in rendered

    # Every citation contributes the section heading.
    assert "Methodology" in rendered
    assert "Implementation" in rendered
    assert "Comments" in rendered

    # Every citation contributes its source-status label (visible in
    # both the table and the notes block).
    assert STATUS_LABEL[SourceStatus.FUNDED] in rendered
    assert STATUS_LABEL[SourceStatus.REJECTED] in rendered
    assert STATUS_LABEL[SourceStatus.ESR_NOTE] in rendered


# ---------------------------------------------------------------------------
# AC3 — funded and rejected MUST render differently
# ---------------------------------------------------------------------------


def test_render_funded_and_rejected_render_differently() -> None:
    """AC3 keystone: same evidence, different status → different bytes.

    Build two drafts that differ only in source_status. The rendered
    outputs must differ in (a) the status badge and (b) the caveat
    line — the funded variant has no caveat, the rejected one carries
    the cautionary text.
    """

    funded_citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.FUNDED),
    ]
    rejected_citations = [
        _make_citation(citation_id=1, status=SourceStatus.REJECTED),
        _make_citation(citation_id=2, status=SourceStatus.REJECTED),
    ]

    funded = MarkdownCitationRenderer().render(_make_draft(citations=funded_citations))
    rejected = MarkdownCitationRenderer().render(_make_draft(citations=rejected_citations))

    # 1. Bytes differ at all.
    assert funded != rejected

    # 2. Funded badge appears in the funded variant only; rejected
    #    badge appears in the rejected variant only.
    assert STATUS_BADGE[SourceStatus.FUNDED] in funded
    assert STATUS_BADGE[SourceStatus.REJECTED] not in funded
    assert STATUS_BADGE[SourceStatus.REJECTED] in rejected
    assert STATUS_BADGE[SourceStatus.FUNDED] not in rejected

    # 3. Caveat line distinguishes the two — funded has no caveat,
    #    rejected has the "NOT funded" cautionary text.
    assert STATUS_CAVEAT[SourceStatus.FUNDED] == ""
    assert STATUS_CAVEAT[SourceStatus.REJECTED] in rejected
    assert STATUS_CAVEAT[SourceStatus.REJECTED] not in funded


# ---------------------------------------------------------------------------
# Inline marker preservation
# ---------------------------------------------------------------------------


def test_render_preserves_inline_markers() -> None:
    """The renderer must NOT strip ``[N]`` markers from the draft text."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.FUNDED),
    ]
    draft = _make_draft(
        citations=citations,
        text="We propose [1] and build on the prior work in [2].",
    )

    rendered = MarkdownCitationRenderer().render(draft)

    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "We propose [1] and build on the prior work in [2]." in rendered


# ---------------------------------------------------------------------------
# PRD invariant: ESR notes are advisory only
# ---------------------------------------------------------------------------


def test_render_esr_marked_advisory_only() -> None:
    """ESR citations must surface the advisory-only caveat AND the badge."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.ESR_NOTE),
    ]
    draft = _make_draft(citations=citations, text="Per the reviewer note [1], ...")

    rendered = MarkdownCitationRenderer().render(draft)

    assert STATUS_BADGE[SourceStatus.ESR_NOTE] in rendered
    assert STATUS_CAVEAT[SourceStatus.ESR_NOTE] in rendered
    # Defence in depth: the cautionary phrase must be present even if
    # someone changes the caveat constant.
    assert "advisory only" in rendered.lower()


def test_render_unknown_status_marked() -> None:
    """UNKNOWN citations must surface the ``?`` badge AND the caveat."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.UNKNOWN),
    ]
    draft = _make_draft(citations=citations, text="See [1] for context.")

    rendered = MarkdownCitationRenderer().render(draft)

    assert STATUS_BADGE[SourceStatus.UNKNOWN] in rendered
    assert STATUS_CAVEAT[SourceStatus.UNKNOWN] in rendered
    assert "treat with caution" in rendered.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_render_no_citations_still_produces_valid_markdown() -> None:
    """An empty citation list renders an explicit ``_No citations._`` line."""

    draft = _make_draft(citations=[], text="Standalone draft text.")

    rendered = MarkdownCitationRenderer().render(draft)

    assert "## References" in rendered
    assert "_No citations._" in rendered
    # No empty table header should leak through when there are no rows.
    assert "| # | Status |" not in rendered


def test_render_section_type_other_uses_other_title() -> None:
    """``SectionType.OTHER`` falls back to a clean ``Other`` heading."""

    citations = [_make_citation(citation_id=1, status=SourceStatus.FUNDED)]
    draft = _make_draft(
        citations=citations,
        text="See [1].",
        section_type=SectionType.OTHER,
    )

    rendered = MarkdownCitationRenderer().render(draft)

    assert rendered.startswith("# Other")


def test_render_handles_missing_optional_fields() -> None:
    """``page=None``, ``section_heading=None``, ``proposal_title=None`` → placeholders."""

    citations = [
        _make_citation(
            citation_id=1,
            status=SourceStatus.FUNDED,
            proposal_title=None,
            section_heading=None,
            page=None,
        ),
    ]
    draft = _make_draft(citations=citations, text="See [1].")

    rendered = MarkdownCitationRenderer().render(draft)

    # Placeholders, NOT the literal Python ``None`` repr.
    assert "None" not in rendered
    assert "untitled" in rendered
    assert "n/a" in rendered
    # The notes block uses the ``p. n/a`` form for missing pages.
    assert "p. n/a" in rendered


def test_render_table_columns_are_consistent() -> None:
    """Every Markdown table row must have the same column count.

    A future refactor that drops a column from one branch (e.g.,
    "no source for ESR notes") would silently break the table; this
    test fails loudly when that happens.
    """

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.REJECTED),
        _make_citation(citation_id=3, status=SourceStatus.ESR_NOTE),
        _make_citation(citation_id=4, status=SourceStatus.UNKNOWN),
        _make_citation(
            citation_id=5,
            status=SourceStatus.FUNDED,
            proposal_title=None,
            page=None,
        ),
    ]
    draft = _make_draft(
        citations=citations,
        text="[1] [2] [3] [4] [5]",
    )

    rendered = MarkdownCitationRenderer().render(draft)

    # Pull the table block: lines that start AND end with "|".
    table_lines = [
        line for line in rendered.splitlines() if line.startswith("|") and line.endswith("|")
    ]
    # Header + separator + 5 rows.
    assert len(table_lines) == 7
    # Every line splits into the same number of pipe-segments.
    widths = {len(line.split("|")) for line in table_lines}
    assert len(widths) == 1, f"Table column counts diverged: {widths}\n--- Rendered ---\n{rendered}"


def test_render_is_deterministic() -> None:
    """Same draft → byte-equal rendered output across two calls.

    The audit and the output-comparison tests both rely on this; a
    timestamp slipping into the rendered string would silently break
    them.
    """

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.REJECTED),
    ]
    draft = _make_draft(citations=citations)

    renderer = MarkdownCitationRenderer()
    first = renderer.render(draft)
    second = renderer.render(draft)

    assert first == second


def test_render_escapes_pipe_in_snippet_friendly_fields() -> None:
    """A ``|`` in a string field must NOT split the table cell.

    Defensive: a proposal title containing a pipe shouldn't silently
    insert an extra column. The renderer escapes pipes with a
    backslash; this test confirms the escape sequence is present and
    that, when escapes are honoured, every row has the same column
    count.
    """

    citations = [
        _make_citation(
            citation_id=1,
            status=SourceStatus.FUNDED,
            proposal_title="Proposal | with pipe",
            section_heading="Section A | Subsection",
        ),
    ]
    draft = _make_draft(citations=citations, text="See [1].")

    rendered = MarkdownCitationRenderer().render(draft)

    # Escaped form lands in the table.
    assert "Proposal \\| with pipe" in rendered
    # Column count remains constant once the renderer's escape sequence
    # is honoured. Strip ``\|`` before splitting so the test reflects
    # what a Markdown renderer actually shows the user.
    table_lines = [
        line for line in rendered.splitlines() if line.startswith("|") and line.endswith("|")
    ]
    assert table_lines, "expected a Markdown table block"
    widths = {len(line.replace("\\|", "").split("|")) for line in table_lines}
    assert len(widths) == 1, (
        f"Table column counts diverged after honouring escapes: {widths}\n"
        f"--- Rendered ---\n{rendered}"
    )
