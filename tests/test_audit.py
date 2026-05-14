"""Tests for ``eurpe.generation.audit`` — release-blocking citation checks.

Pin Issue #7 acceptance criteria #2 (audit fails on any citation that
lacks a non-empty status tag), AC1 indirectly via ``audit_rendered``,
and a defensive battery of edge cases — including the bounded-regex
``[100]`` hallucination that PR #41 reviewers flagged.

Tests build :class:`GenerationDraft` records by hand (no LLM, no
retriever) so each scenario is a single named invariant. The
``model_construct`` bypass for the AC2 keystone is intentional:
Pydantic enforces non-None ``source_status`` at construction, so the
only way to test the audit's defensive check is to skip validation.
"""

from __future__ import annotations

from eurpe.generation import (
    STATUS_BADGE,
    AuditFinding,
    AuditResult,
    AuditSeverity,
    CitationAudit,
    CitationRef,
    GenerationDraft,
    GenerationRequest,
    MarkdownCitationRenderer,
)
from eurpe.schema import Programme, SectionType, SourceStatus

# ---------------------------------------------------------------------------
# Builders — same shape as test_render._make_*; duplicated so each test
# file is independently readable.
# ---------------------------------------------------------------------------


def _make_citation(
    *,
    citation_id: int,
    status: SourceStatus = SourceStatus.FUNDED,
    programme: Programme = Programme.HORIZON_EUROPE,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    proposal_title: str | None = "Sample Proposal",
    section_heading: str | None = "1.2 Methodology",
    page: int | None = 12,
    chunk_id: str | None = None,
    snippet: str = "This is a snippet from the source chunk.",
) -> CitationRef:
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
# Happy path
# ---------------------------------------------------------------------------


def test_audit_clean_draft_passes() -> None:
    """A draft with valid citations and matching markers passes cleanly."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.REJECTED),
    ]
    draft = _make_draft(citations=citations)

    result = CitationAudit().audit_draft(draft)

    assert result.passed is True
    assert result.errors == []
    assert result.warnings == []


# ---------------------------------------------------------------------------
# Marker / citation matching
# ---------------------------------------------------------------------------


def test_audit_detects_marker_without_citation() -> None:
    """Text contains [99] but no citation has that id → ERROR."""

    citations = [_make_citation(citation_id=1), _make_citation(citation_id=2)]
    draft = _make_draft(citations=citations, text="[1] [2] [99]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "marker_without_citation" in codes
    # The offending citation_id is surfaced for the operator.
    offending = [f for f in result.errors if f.code == "marker_without_citation"]
    assert any(f.citation_id == 99 for f in offending)


def test_audit_detects_duplicate_citation_id() -> None:
    """Two citations with the same id → ERROR."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
    ]
    draft = _make_draft(citations=citations, text="[1]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "duplicate_citation_id" in codes


def test_audit_detects_non_sequential_ids() -> None:
    """Citation ids [1, 3] (gap) → ERROR."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=3, status=SourceStatus.FUNDED),
    ]
    draft = _make_draft(citations=citations, text="[1] [3]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "non_sequential_citation_ids" in codes


def test_audit_detects_unused_citation_warning() -> None:
    """Citation list has [1, 2] but text only references [1] → WARNING."""

    citations = [_make_citation(citation_id=1), _make_citation(citation_id=2)]
    draft = _make_draft(citations=citations, text="[1]")

    result = CitationAudit().audit_draft(draft)

    # Warnings do not fail the audit.
    assert result.passed is True
    codes = {f.code for f in result.warnings}
    assert "unused_citation" in codes
    # The offending id is surfaced for the operator.
    unused = [f for f in result.warnings if f.code == "unused_citation"]
    assert any(f.citation_id == 2 for f in unused)


def test_audit_detects_empty_snippet_warning() -> None:
    """A blank snippet → WARNING (not ERROR)."""

    citations = [
        _make_citation(citation_id=1, snippet="   "),  # whitespace only
    ]
    draft = _make_draft(citations=citations, text="[1]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is True
    codes = {f.code for f in result.warnings}
    assert "empty_snippet" in codes


# ---------------------------------------------------------------------------
# Render-time checks
# ---------------------------------------------------------------------------


def test_audit_rendered_detects_missing_badge() -> None:
    """Mutating the rendered Markdown to drop a badge → ERROR.

    Simulates the case where a future renderer change accidentally
    strips the visible status label. The audit catches it before the
    draft ships.
    """

    citations = [_make_citation(citation_id=1, status=SourceStatus.FUNDED)]
    draft = _make_draft(citations=citations, text="See [1].")
    rendered = MarkdownCitationRenderer().render(draft)
    # Drop the funded badge to simulate a rendering bug.
    mutilated = rendered.replace(STATUS_BADGE[SourceStatus.FUNDED], "")
    # Sanity: the FUNDED label still appears in the table cell, but
    # the badge form is gone — the audit looks for the badge form.
    assert STATUS_BADGE[SourceStatus.FUNDED] not in mutilated

    result = CitationAudit().audit_rendered(draft, mutilated)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "bad_render" in codes


def test_audit_rendered_passes_for_clean_render() -> None:
    """A freshly rendered draft passes ``audit_rendered`` cleanly."""

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.REJECTED),
    ]
    draft = _make_draft(citations=citations)
    rendered = MarkdownCitationRenderer().render(draft)

    result = CitationAudit().audit_rendered(draft, rendered)

    assert result.passed is True
    assert result.errors == []


def test_audit_rendered_detects_stripped_inline_marker() -> None:
    """A rendered output missing a ``[N]`` from the draft text → ERROR.

    Defensive: catches the case where post-processing strips the
    inline marker but leaves the references list intact.
    """

    citations = [_make_citation(citation_id=1, status=SourceStatus.FUNDED)]
    draft = _make_draft(citations=citations, text="See [1] now.")
    rendered = MarkdownCitationRenderer().render(draft)
    # Strip the inline marker only (not the ones in the references block).
    mutilated = rendered.replace("See [1] now.", "See now.", 1)

    result = CitationAudit().audit_rendered(draft, mutilated)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "bad_render" in codes


# ---------------------------------------------------------------------------
# AuditResult.passed plumbing
# ---------------------------------------------------------------------------


def test_audit_passes_property_works() -> None:
    """Hand-constructed AuditResult: passed=True iff no ERROR findings."""

    error_finding = AuditFinding(
        severity=AuditSeverity.ERROR,
        code="missing_status",
        message="x",
    )
    warning_finding = AuditFinding(
        severity=AuditSeverity.WARNING,
        code="empty_snippet",
        message="y",
    )

    # passed=True is computed by the audit; here we directly construct
    # an AuditResult to confirm the contract: errors property filters
    # by severity, passed doesn't auto-derive (caller sets it).
    has_error = AuditResult(findings=[error_finding, warning_finding], passed=False)
    assert has_error.passed is False
    assert has_error.errors == [error_finding]
    assert has_error.warnings == [warning_finding]

    only_warning = AuditResult(findings=[warning_finding], passed=True)
    assert only_warning.passed is True
    assert only_warning.errors == []
    assert only_warning.warnings == [warning_finding]


# ---------------------------------------------------------------------------
# The [100] hallucination hole — closes a gap PR #41 reviewers flagged
# ---------------------------------------------------------------------------


def test_audit_rejects_three_digit_marker() -> None:
    """``[100]`` in text with no citation 100 → ERROR (closes PR #41 gap).

    The workflow's own validator uses ``\\d{1,2}`` and silently
    ignores ``[100]``. The audit uses an unbounded ``\\d+`` regex
    explicitly so this case is caught.
    """

    citations = [_make_citation(citation_id=1)]
    draft = _make_draft(citations=citations, text="[1] [100]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "marker_without_citation" in codes
    offending = [f for f in result.errors if f.code == "marker_without_citation"]
    assert any(f.citation_id == 100 for f in offending)


# ---------------------------------------------------------------------------
# AC2 keystone — defensive: source_status=None must surface as ERROR
# ---------------------------------------------------------------------------


def test_audit_fails_when_any_citation_has_none_status() -> None:
    """AC2 keystone: status=None is release-blocking.

    Bypass Pydantic with ``model_construct`` so a malformed citation
    with ``source_status=None`` is constructible. The audit MUST
    surface this as ERROR — it is the single most important invariant
    of the system per PRD § "Source-status labelling".
    """

    valid = _make_citation(citation_id=1, status=SourceStatus.FUNDED)
    # ``model_construct`` skips validation; this is the only way to
    # produce a CitationRef with source_status=None for the test.
    bypassed = CitationRef.model_construct(
        citation_id=2,
        source_status=None,
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-X",
        proposal_title="Bypassed",
        section_heading="x",
        page=1,
        chunk_id="chunk-2",
        snippet="snippet",
    )
    draft = _make_draft(citations=[valid, bypassed], text="[1] [2]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "missing_status" in codes
    # The offending citation id is surfaced for the operator.
    missing = [f for f in result.errors if f.code == "missing_status"]
    assert any(f.citation_id == 2 for f in missing)


def test_audit_fails_when_call_id_is_blank() -> None:
    """An empty ``call_id`` (bypassing Pydantic) → ERROR."""

    valid = _make_citation(citation_id=1, status=SourceStatus.FUNDED)
    bypassed = CitationRef.model_construct(
        citation_id=2,
        source_status=SourceStatus.FUNDED,
        programme=Programme.HORIZON_EUROPE,
        call_id="   ",
        proposal_title="Bypassed",
        section_heading="x",
        page=1,
        chunk_id="chunk-2",
        snippet="snippet",
    )
    draft = _make_draft(citations=[valid, bypassed], text="[1] [2]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "empty_call_id" in codes


# ---------------------------------------------------------------------------
# Runtime audit gates — issue #45
# ---------------------------------------------------------------------------


def test_audit_fails_on_no_evidence_escape_with_empty_citations() -> None:
    """Escape sentence + zero citations → ``no_evidence_escape`` ERROR.

    AC1 of issue #45: the audit must NOT pass a draft that explicitly
    admits "no retrieved evidence was available" while carrying an
    empty citation table.
    """

    draft = _make_draft(
        citations=[],
        text=(
            "Draft for the Methodology section. "
            "No retrieved evidence was available; expand the index "
            "with relevant past proposals before relying on this draft."
        ),
    )

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "no_evidence_escape" in codes


def test_audit_passes_when_escape_sentence_appears_but_citations_present() -> None:
    """Escape sentence alone does NOT fail when citations exist.

    The dual condition (escape sentence AND empty citations) keeps the
    gate narrow: a real LLM that discusses the absence of corroborating
    evidence while still citing comparators must not be flagged.
    """

    citations = [_make_citation(citation_id=1, status=SourceStatus.FUNDED)]
    draft = _make_draft(
        citations=citations,
        text=(
            "We note that no retrieved evidence was available; expand the index "
            "with relevant past proposals before relying on this draft. "
            "Nevertheless, comparator [1] guides the design."
        ),
    )

    result = CitationAudit().audit_draft(draft)

    codes = {f.code for f in result.errors}
    assert "no_evidence_escape" not in codes


def test_audit_passes_when_empty_citations_without_escape_sentence() -> None:
    """Empty citations alone (no escape sentence) does not trip the gate.

    A LangGraph node that legitimately returns a draft with no citation
    list and no escape sentence (e.g., a planning step that runs before
    retrieval) is out of scope for this gate.
    """

    draft = _make_draft(
        citations=[],
        text="Draft for the Methodology section without any retrieved evidence.",
    )

    result = CitationAudit().audit_draft(draft)

    codes = {f.code for f in result.errors}
    assert "no_evidence_escape" not in codes


def test_audit_fails_on_placeholder_text_even_with_citations() -> None:
    """Stub placeholder leaked into draft body → ``placeholder_text`` ERROR.

    AC2 of issue #45: if the deterministic stub's literal placeholder
    sentence lands in a real run, the stub leaked through the
    offline-fallback path and the draft is not trustworthy regardless
    of how many citations it carries.
    """

    citations = [
        _make_citation(citation_id=1, status=SourceStatus.FUNDED),
        _make_citation(citation_id=2, status=SourceStatus.REJECTED),
    ]
    draft = _make_draft(
        citations=citations,
        text=(
            "Draft for the Methodology section, derived from retrieved evidence. "
            "This sentence references retrieved example [1] as supporting evidence. "
            "This sentence references retrieved example [2] as supporting evidence."
        ),
    )

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "placeholder_text" in codes


def test_audit_passes_when_text_describes_evidence_without_placeholder() -> None:
    """Prose that *paraphrases* the placeholder does not trip the gate.

    The regex is anchored on the literal stub output. A real LLM that
    writes ``This sentence reflects evidence from comparator [1]`` is
    safe; only the verbatim stub phrasing fails.
    """

    citations = [_make_citation(citation_id=1, status=SourceStatus.FUNDED)]
    draft = _make_draft(
        citations=citations,
        text=(
            "This sentence reflects supporting evidence drawn from comparator "
            "[1] in the indexed corpus."
        ),
    )

    result = CitationAudit().audit_draft(draft)

    codes = {f.code for f in result.errors}
    assert "placeholder_text" not in codes


def test_audit_rendered_also_fires_runtime_gates() -> None:
    """The runtime gates fire in :meth:`audit_rendered` as well.

    Same contract as :meth:`audit_draft` — a rendered draft that
    admits "no retrieved evidence" with an empty citation table must
    not pass the audit either.
    """

    draft = _make_draft(
        citations=[],
        text=(
            "Draft for the Methodology section. "
            "No retrieved evidence was available; expand the index "
            "with relevant past proposals before relying on this draft."
        ),
    )
    rendered = MarkdownCitationRenderer().render(draft)

    result = CitationAudit().audit_rendered(draft, rendered)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "no_evidence_escape" in codes


# ---------------------------------------------------------------------------


def test_audit_fails_when_programme_is_none() -> None:
    """A ``None`` programme (bypassing Pydantic) → ``empty_programme`` ERROR."""

    valid = _make_citation(citation_id=1, status=SourceStatus.FUNDED)
    bypassed = CitationRef.model_construct(
        citation_id=2,
        source_status=SourceStatus.FUNDED,
        programme=None,
        call_id="HORIZON-X",
        proposal_title="Bypassed",
        section_heading="x",
        page=1,
        chunk_id="chunk-2",
        snippet="snippet",
    )
    draft = _make_draft(citations=[valid, bypassed], text="[1] [2]")

    result = CitationAudit().audit_draft(draft)

    assert result.passed is False
    codes = {f.code for f in result.errors}
    assert "empty_programme" in codes
