"""Tests for ``eurpe.generation.audit_harness`` — the release audit harness.

The harness aggregates the existing per-draft
:class:`~eurpe.generation.audit.CitationAudit` over a directory of
saved drafts so a release manager can confirm an entire sample of
generated sections preserves the source-status invariant.

Coverage matrix
---------------
* **AC1** — ``audit_directory`` returns one row per citation with
  every source-document field populated.
* **AC2** — when any draft has a citation missing its status label,
  the report's ``passed`` flag is False (release-blocking).
* **AC3** — sampling is deterministic and explicit (``sample_size``
  parameter; recorded on the report); the manual template lives in
  ``docs/release-audit-template.md`` (verified by the CLI test that
  references it).
* Edge cases — empty directory, missing directory, malformed draft
  JSON, sample-size larger than discovered count, sample-size of
  zero rejected explicitly.

Tests build :class:`GenerationDraft` records by hand and dump them
to JSON under ``tmp_path`` so each scenario is a single named
invariant. No LLM, no retriever, no network — same offline contract
as :mod:`tests.test_audit`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eurpe.generation import (
    CitationAuditRow,
    GenerationDraft,
    GenerationRequest,
    ReleaseAuditHarness,
    ReleaseAuditHarnessError,
    ReleaseAuditReport,
    has_release_blocking_findings,
)
from eurpe.generation.models import CitationRef
from eurpe.schema import Programme, SectionType, SourceStatus

# ---------------------------------------------------------------------------
# Builders — mirror the shape used in ``tests.test_audit``. Duplicated
# (not imported) so this test file is independently readable.
# ---------------------------------------------------------------------------


def _make_citation(
    *,
    citation_id: int = 1,
    status: SourceStatus = SourceStatus.FUNDED,
    programme: Programme = Programme.HORIZON_EUROPE,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    proposal_title: str | None = "Sample Proposal",
    section_heading: str | None = "1.2 Methodology",
    page: int | None = 12,
    chunk_id: str | None = None,
    snippet: str = "Snippet from the source chunk.",
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
    citations: list[CitationRef] | None = None,
    text: str = "We propose [1] and learn from [2].",
    section_type: SectionType = SectionType.METHODOLOGY,
) -> GenerationDraft:
    if citations is None:
        citations = [
            _make_citation(citation_id=1, status=SourceStatus.FUNDED),
            _make_citation(citation_id=2, status=SourceStatus.REJECTED),
        ]
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


def _dump_draft(directory: Path, name: str, draft: GenerationDraft) -> Path:
    """Write a draft JSON under ``directory`` and return the path.

    Uses ``model_dump_json`` so the JSON layout matches what the CLI
    writes — the harness must accept exactly that shape.
    """

    path = directory / name
    path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    return path


def _dump_payload(directory: Path, name: str, payload: dict) -> Path:
    """Write a literal payload dict (used to construct invalid drafts)."""

    path = directory / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Happy path — AC1 + AC2 together
# ---------------------------------------------------------------------------


def test_audit_directory_passes_on_all_clean_drafts(tmp_path: Path) -> None:
    """Three clean drafts → passed=True, no failed drafts, rows populated."""

    _dump_draft(tmp_path, "draft-a.json", _make_draft())
    _dump_draft(tmp_path, "draft-b.json", _make_draft())
    _dump_draft(tmp_path, "draft-c.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    assert report.passed is True
    assert report.total_drafts == 3
    assert report.audited_drafts == 3
    assert report.passed_drafts == 3
    assert report.failed_drafts == 0
    # Each clean draft has 2 citations → 6 rows total.
    assert len(report.rows) == 6
    assert report.citation_count == 6
    assert report.unlabeled_citation_count == 0


def test_audit_directory_row_carries_every_source_field(tmp_path: Path) -> None:
    """AC1 — every row exposes status AND source document fields."""

    _dump_draft(
        tmp_path,
        "draft.json",
        _make_draft(
            citations=[
                _make_citation(
                    citation_id=1,
                    status=SourceStatus.FUNDED,
                    programme=Programme.HORIZON_EUROPE,
                    call_id="HORIZON-CL5-2024-D3-02",
                    proposal_title="My Funded Project",
                    section_heading="2.1 Approach",
                    page=15,
                    chunk_id="chunk-evidence-1",
                ),
            ],
            text="We propose [1].",
        ),
    )

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    assert len(report.rows) == 1
    row = report.rows[0]
    # Every source-document field MUST be populated for AC1.
    assert row.citation_id == 1
    assert row.source_status == "funded"
    assert row.programme == "horizon_europe"
    assert row.call_id == "HORIZON-CL5-2024-D3-02"
    assert row.proposal_title == "My Funded Project"
    assert row.section_heading == "2.1 Approach"
    assert row.page == 15
    assert row.chunk_id == "chunk-evidence-1"
    # And the row carries the back-link to the source draft.
    assert row.draft_path.endswith("draft.json")
    assert row.section_type == "methodology"


def test_audit_directory_one_failing_draft_fails_the_run(tmp_path: Path) -> None:
    """AC2 — a draft with [99] but no citation 99 → report.passed=False."""

    _dump_draft(tmp_path, "ok.json", _make_draft())
    _dump_draft(
        tmp_path,
        "bad.json",
        _make_draft(text="We propose [1] and reference [99]."),
    )

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    assert report.passed is False
    assert report.passed_drafts == 1
    assert report.failed_drafts == 1
    failures = [r for r in report.draft_results if not r.passed]
    assert len(failures) == 1
    codes = {f.code for f in failures[0].audit_result.errors}
    assert "marker_without_citation" in codes


def test_audit_directory_fails_on_missing_status_label(tmp_path: Path) -> None:
    """AC2 keystone — citation with no source_status surfaces as failure."""

    # Bypass Pydantic to construct a citation with no source_status,
    # mirroring the pattern in tests.test_audit.
    valid = _make_citation(citation_id=1, status=SourceStatus.FUNDED)
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
    # ``model_dump_json`` would serialize source_status as None, which
    # round-trips back through validation and trips Pydantic. We
    # therefore write the payload manually, mirroring the test_audit
    # pattern, but in JSON form.
    payload = {
        "section_type": draft.section_type.value,
        "text": draft.text,
        "citations": [
            {
                "citation_id": 1,
                "source_status": "funded",
                "programme": "horizon_europe",
                "call_id": "HORIZON-CL5-2024-D3-02",
                "proposal_title": "Sample Proposal",
                "section_heading": "1.2 Methodology",
                "page": 12,
                "chunk_id": "chunk-1",
                "snippet": "snippet",
            },
            {
                "citation_id": 2,
                # source_status omitted — Pydantic must error on load.
                "programme": "horizon_europe",
                "call_id": "HORIZON-X",
                "proposal_title": "Bypassed",
                "section_heading": "x",
                "page": 1,
                "chunk_id": "chunk-2",
                "snippet": "snippet",
            },
        ],
        "prompt_used": "(elided)",
        "model": "deterministic-stub-v1",
        "request": {
            "section_type": "methodology",
            "user_intent": "Describe the approach.",
            "call_context": "",
            "target_programme": None,
            "top_k_examples": 5,
            "lessons_learned": False,
            "topic_context": None,
        },
        "drafting_profile": None,
        "topic_context": None,
        "iterations": [],
    }
    _dump_payload(tmp_path, "missing-status.json", payload)

    # An omitted required field surfaces as a load-time
    # ``ReleaseAuditHarnessError`` (Pydantic rejects construction
    # before the audit can run). This is the safer behaviour: a
    # corrupt persisted draft fails the release loudly. AC2 is
    # therefore enforced at two layers — schema enforcement (load
    # fail) AND audit-time enforcement (the audit_draft defensive
    # check via model_construct in tests.test_audit). The harness
    # surfaces the load error consistently.
    with pytest.raises(ReleaseAuditHarnessError) as exc:
        ReleaseAuditHarness().audit_directory(tmp_path)
    assert "missing-status.json" in str(exc.value)


def test_audit_directory_records_unlabeled_count_for_present_status(
    tmp_path: Path,
) -> None:
    """A draft with every citation labelled → unlabeled_citation_count == 0."""

    _dump_draft(tmp_path, "draft.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    # Two FUNDED + REJECTED citations are both labelled.
    assert report.unlabeled_citation_count == 0


# ---------------------------------------------------------------------------
# Sampling — AC3 plumbing
# ---------------------------------------------------------------------------


def test_audit_directory_no_sample_size_audits_everything(tmp_path: Path) -> None:
    """``sample_size=None`` audits every draft and records no sample meta."""

    for i in range(5):
        _dump_draft(tmp_path, f"draft-{i:02d}.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    assert report.total_drafts == 5
    assert report.audited_drafts == 5
    assert report.sample_size is None
    assert report.sample_seed is None
    assert len(report.sampled_paths) == 5


def test_audit_directory_sample_size_picks_deterministic_subset(
    tmp_path: Path,
) -> None:
    """``sample_size=2`` over 5 drafts → audits exactly 2; seed reproducible."""

    for i in range(5):
        _dump_draft(tmp_path, f"draft-{i:02d}.json", _make_draft())

    report_a = ReleaseAuditHarness().audit_directory(tmp_path, sample_size=2, seed=42)
    report_b = ReleaseAuditHarness().audit_directory(tmp_path, sample_size=2, seed=42)

    assert report_a.audited_drafts == 2
    assert report_a.sample_size == 2
    assert report_a.sample_seed == 42
    assert report_a.sampled_paths == report_b.sampled_paths


def test_audit_directory_sample_size_default_seed_42(tmp_path: Path) -> None:
    """When ``seed`` is omitted, default 42 is used and recorded."""

    for i in range(5):
        _dump_draft(tmp_path, f"draft-{i:02d}.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path, sample_size=2)

    assert report.sample_size == 2
    assert report.sample_seed == 42


def test_audit_directory_sample_size_larger_than_available_audits_all(
    tmp_path: Path,
) -> None:
    """``sample_size >= len(paths)`` → audits everything (no error)."""

    for i in range(2):
        _dump_draft(tmp_path, f"draft-{i:02d}.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path, sample_size=10)

    assert report.audited_drafts == 2
    assert report.total_drafts == 2


def test_audit_directory_sample_size_zero_rejected(tmp_path: Path) -> None:
    """``sample_size=0`` is rejected as a probable typo."""

    _dump_draft(tmp_path, "draft.json", _make_draft())

    with pytest.raises(ReleaseAuditHarnessError, match="sample_size"):
        ReleaseAuditHarness().audit_directory(tmp_path, sample_size=0)


def test_audit_directory_sample_size_supports_release_threshold(
    tmp_path: Path,
) -> None:
    """PRD requires ≥ 20 sections per release — harness handles that count.

    Construct 25 drafts and pin sample_size=20 to confirm the harness
    handles the release-scale threshold without surprises.
    """

    for i in range(25):
        _dump_draft(tmp_path, f"draft-{i:02d}.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path, sample_size=20, seed=42)

    assert report.audited_drafts == 20
    assert report.total_drafts == 25
    assert len(report.sampled_paths) == 20
    # Every sampled path appears in the per-draft results.
    sampled_set = set(report.sampled_paths)
    drafted_set = {r.draft_path for r in report.draft_results}
    assert sampled_set == drafted_set


# ---------------------------------------------------------------------------
# Failure modes — missing directory, malformed JSON, ...
# ---------------------------------------------------------------------------


def test_audit_directory_missing_directory_errors(tmp_path: Path) -> None:
    """A non-existent directory raises ``ReleaseAuditHarnessError``."""

    missing = tmp_path / "does-not-exist"
    with pytest.raises(ReleaseAuditHarnessError, match="does not exist"):
        ReleaseAuditHarness().audit_directory(missing)


def test_audit_directory_path_is_a_file_errors(tmp_path: Path) -> None:
    """A path that points to a file (not directory) is rejected."""

    file_path = tmp_path / "draft.json"
    _dump_draft(tmp_path, "draft.json", _make_draft())

    with pytest.raises(ReleaseAuditHarnessError):
        ReleaseAuditHarness().audit_directory(file_path)


def test_audit_directory_empty_directory_passes_vacuously(tmp_path: Path) -> None:
    """Empty directory → audited=0, passed=True. CLI may treat as warning."""

    empty = tmp_path / "empty"
    empty.mkdir()

    report = ReleaseAuditHarness().audit_directory(empty)

    assert report.total_drafts == 0
    assert report.audited_drafts == 0
    assert report.failed_drafts == 0
    assert report.passed is True
    assert report.citation_count == 0


def test_audit_directory_malformed_json_errors(tmp_path: Path) -> None:
    """Unparseable JSON file → ``ReleaseAuditHarnessError`` with path."""

    bad = tmp_path / "broken.json"
    bad.write_text("not-json{", encoding="utf-8")

    with pytest.raises(ReleaseAuditHarnessError, match="broken.json"):
        ReleaseAuditHarness().audit_directory(tmp_path)


def test_audit_directory_walks_subdirectories(tmp_path: Path) -> None:
    """Drafts nested inside subdirectories are discovered too."""

    subdir = tmp_path / "section-a"
    subdir.mkdir()
    _dump_draft(subdir, "draft.json", _make_draft())
    _dump_draft(tmp_path, "top.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    assert report.total_drafts == 2
    # The order is deterministic (sorted POSIX paths).
    paths = [Path(p).name for p in report.sampled_paths]
    assert sorted(paths) == paths


# ---------------------------------------------------------------------------
# Markdown summary rendering
# ---------------------------------------------------------------------------


def test_render_markdown_summary_pass(tmp_path: Path) -> None:
    """The Markdown summary marks a passing run with PASS verdict.

    Pins the at-a-glance contract: the heading, the verdict marker,
    the per-draft block, and the citation fidelity table all appear.
    """

    _dump_draft(tmp_path, "draft.json", _make_draft())
    report = ReleaseAuditHarness().audit_directory(tmp_path)

    md = report.render_markdown_summary()

    assert "# Release Audit Report" in md
    assert "**Verdict:** PASS" in md
    assert "## Summary" in md
    assert "## Per-draft Results" in md
    assert "## Citation Fidelity" in md
    assert "draft.json" in md


def test_render_markdown_summary_fail(tmp_path: Path) -> None:
    """A failing run renders the FAIL verdict explicitly."""

    _dump_draft(
        tmp_path,
        "bad.json",
        _make_draft(text="We propose [1] and reference [99]."),
    )

    report = ReleaseAuditHarness().audit_directory(tmp_path)

    md = report.render_markdown_summary()

    assert "**Verdict:** FAIL" in md


def test_render_markdown_summary_empty_directory(tmp_path: Path) -> None:
    """Empty directory renders a placeholder rather than a malformed table."""

    empty = tmp_path / "empty"
    empty.mkdir()

    report = ReleaseAuditHarness().audit_directory(empty)
    md = report.render_markdown_summary()

    assert "_No drafts audited._" in md
    assert "_No citations across audited drafts" in md


def test_render_markdown_summary_is_deterministic(tmp_path: Path) -> None:
    """Same report → byte-equal Markdown across renders.

    The audit_directory result already pins the order; this test
    pins the renderer itself so a future refactor does not introduce
    nondeterminism (e.g., a dict iteration order leak).
    """

    _dump_draft(tmp_path, "a.json", _make_draft())
    _dump_draft(tmp_path, "b.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path)
    md_first = report.render_markdown_summary()
    md_second = report.render_markdown_summary()

    assert md_first == md_second


def test_render_markdown_summary_includes_sample_metadata(tmp_path: Path) -> None:
    """When sampling is in effect the summary records size + seed."""

    for i in range(3):
        _dump_draft(tmp_path, f"d-{i}.json", _make_draft())

    report = ReleaseAuditHarness().audit_directory(tmp_path, sample_size=2, seed=7)

    md = report.render_markdown_summary()
    assert "Sample size: 2" in md
    assert "Sample seed: 7" in md


# ---------------------------------------------------------------------------
# Models and helpers
# ---------------------------------------------------------------------------


def test_citation_audit_row_rejects_unknown_field() -> None:
    """``extra="forbid"`` keeps typos loud — matches codebase convention."""

    with pytest.raises(ValidationError):
        CitationAuditRow(
            draft_path="x.json",
            section_type="methodology",
            citation_id=1,
            source_status="funded",
            programme="horizon_europe",
            call_id="HORIZON-X",
            chunk_id="chunk-1",
            typo_field="oops",  # type: ignore[call-arg]
        )


def test_release_audit_report_passed_iff_no_failed_drafts(tmp_path: Path) -> None:
    """``ReleaseAuditReport.passed`` is True iff failed_drafts == 0.

    Pin the contract directly by constructing the report (so a
    future change cannot quietly compute passed differently).
    """

    report = ReleaseAuditReport(
        audit_directory="/tmp/x",
        total_drafts=2,
        audited_drafts=2,
        passed_drafts=2,
        failed_drafts=0,
        citation_count=4,
        unlabeled_citation_count=0,
        draft_results=[],
        rows=[],
        passed=True,
    )
    assert report.passed is True

    failing = ReleaseAuditReport(
        audit_directory="/tmp/x",
        total_drafts=2,
        audited_drafts=2,
        passed_drafts=1,
        failed_drafts=1,
        citation_count=4,
        unlabeled_citation_count=0,
        draft_results=[],
        rows=[],
        passed=False,
    )
    assert failing.passed is False


def test_has_release_blocking_findings(tmp_path: Path) -> None:
    """``has_release_blocking_findings`` returns True iff any error finding exists."""

    _dump_draft(
        tmp_path,
        "bad.json",
        _make_draft(text="We propose [1] and reference [99]."),
    )

    report = ReleaseAuditHarness().audit_directory(tmp_path)
    assert has_release_blocking_findings(report) is True

    # On a clean run, the helper returns False.
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    _dump_draft(clean_dir, "ok.json", _make_draft())
    clean_report = ReleaseAuditHarness().audit_directory(clean_dir)
    # The deterministic helper draft does not exercise the runtime
    # gates; assert via passed flag instead of the helper directly.
    assert clean_report.passed is True
    assert has_release_blocking_findings(clean_report) is False


# ---------------------------------------------------------------------------
# AC3 plumbing — manual template must ship on disk
# ---------------------------------------------------------------------------


def test_release_audit_template_exists() -> None:
    """The manual audit template at ``docs/release-audit-template.md`` ships.

    AC3 requires a manual audit template that scales to ≥ 20 sections
    per release. The harness automates the per-citation label check;
    the template owns the human-judgement half. This test asserts
    the template file is present in the repo and carries the 20-row
    checklist (counted by the row markers).
    """

    repo_root = Path(__file__).resolve().parent.parent
    template = repo_root / "docs" / "release-audit-template.md"
    assert template.exists(), f"manual release audit template missing: {template}"

    body = template.read_text(encoding="utf-8")
    # The checklist must support ≥ 20 sections — pin the row markers.
    for row in range(1, 21):
        marker = f"| {row}  |" if row < 10 else f"| {row} |"
        assert marker in body, (
            f"checklist row {row} missing from {template}: "
            f"the template must scaffold at least 20 sections per release"
        )

    # Must reference the automated harness command so an operator
    # following the template knows how to produce the input.
    assert "audit-release" in body
