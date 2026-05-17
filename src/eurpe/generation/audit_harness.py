"""Release-time citation fidelity and source-label audit harness.

This module exists for one reason: the per-draft :class:`CitationAudit`
proves *one* generated section is trustworthy, but a release-readiness
check needs to confirm that an *entire sample* of generated sections
keeps the source-status invariant. Task 3.4 (issue #18) requires a
harness that walks a directory of saved drafts, lists every citation
with its source document and status, fails the run if any citation
lacks a status label, and is sized so a release manager can audit at
least 20 sections per cut.

Why a separate module from :mod:`eurpe.generation.audit`
--------------------------------------------------------
The per-draft audit is a *semantic* check ("is this single draft
trustworthy?"). The release harness is a *batch orchestration*
check ("is this set of drafts release-ready and what citations did
the LLM pick?"). Conflating the two would make the per-draft module
the only thing the renderer's tests touch and slowly drift it
toward sampling concerns. Keeping them in separate modules also
mirrors the convention used by the per-draft :func:`audit` CLI
subcommand vs. the new batch :func:`audit_release` subcommand.

What the harness produces
-------------------------
Every run produces a single :class:`ReleaseAuditReport` containing:

* ``rows`` — one :class:`CitationAuditRow` per citation in the
  sampled drafts. Each row carries the source-document provenance
  fields (``proposal_title``, ``call_id``, ``chunk_id``, ``page``,
  ``section_heading``) AND the ``source_status`` label, so an
  operator can sanity-check every claim against the cited source.
* ``draft_results`` — one :class:`DraftAuditResult` per draft, with
  the path it came from, the per-draft :class:`AuditResult`, and a
  computed ``passed`` flag. Lets the operator drill from the summary
  back to the offending draft.
* ``sampled_paths`` — the deterministic file list the harness picked.
* ``total_drafts``, ``audited_drafts``, ``passed_drafts``,
  ``failed_drafts``, ``citation_count``, ``unlabeled_citation_count``
  — top-level counters for the release notes.
* ``passed`` — ``True`` iff every audited draft passed. The CLI
  exits non-zero on ``False``.

Sampling contract
-----------------
The harness picks drafts by *sorted-filename ascending* (deterministic
across CI runs). When ``sample_size`` is None or larger than the
available count, every draft is audited. When it is smaller, a
``random.Random`` is seeded by ``seed`` (default 42) and used to
pick a deterministic subset. The output ``sampled_paths`` always
records the exact set audited so a re-run with the same seed is
byte-equal.

PRD anchor: line 21 of ``prd.md`` requires "random sample of ≥20
generated sections per release". The harness defaults to
``sample_size=None`` (audit everything) so the operator does not
silently skip drafts; sampling is opt-in via ``--sample-size``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from eurpe.generation.audit import AuditResult, AuditSeverity, CitationAudit
from eurpe.generation.models import GenerationDraft
from eurpe.generation.render import MarkdownCitationRenderer

#: File-glob the harness walks under the audit directory. ``GenerationDraft``
#: payloads are saved by ``eurpe generate section --output`` as ``.json``;
#: rendering siblings (``.md``) are skipped by the loader because they are
#: not parseable as Pydantic models.
_DRAFT_GLOB = "**/*.json"

#: Default seed for the deterministic-sampling fallback. Hard-coded so
#: a release manager who forgets ``--seed`` still gets a reproducible
#: subset across machines. The choice of 42 is conventional only.
_DEFAULT_SAMPLE_SEED = 42


class CitationAuditRow(BaseModel):
    """One row in the citation fidelity report — one citation, one source.

    AC1 of issue #18 requires the harness to list *every citation*
    with the source document AND the status. This row therefore
    carries the full source-document fingerprint (chunk_id, call_id,
    proposal_title, page, section_heading) AND the source_status
    label, so a release manager can paste the rendered Markdown
    table into release notes and have every claim verifiable from
    the row alone.

    ``frozen=True`` so a list of rows can be sorted, hashed, or
    set-deduplicated without losing identity. ``extra="forbid"``
    keeps typos in field names loud — the convention across the
    codebase (see :mod:`eurpe.schema`, :mod:`eurpe.generation.models`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_path: str = Field(
        min_length=1,
        description=(
            "Path to the draft JSON file the citation came from, as a "
            "POSIX-style string. Recorded so the operator can drill from "
            "a row back to the source draft."
        ),
    )
    section_type: str = Field(
        min_length=1,
        description="Section the cited draft was generated for (e.g., methodology).",
    )
    citation_id: int = Field(
        ge=1,
        description="1-indexed citation id (matches the ``[N]`` marker in the draft text).",
    )
    source_status: str = Field(
        min_length=1,
        description=(
            "Source-status label of the cited chunk (funded/rejected/esr_note/"
            "unknown). Stored as the string value rather than the enum so the "
            "JSON report is self-describing without an enum import."
        ),
    )
    programme: str = Field(
        min_length=1,
        description="EU programme of the source proposal (e.g., horizon_europe).",
    )
    call_id: str = Field(
        min_length=1,
        description="Call identifier (e.g., HORIZON-CL5-2024-D3-02).",
    )
    proposal_title: str | None = Field(
        default=None,
        description="Title of the source proposal, when known.",
    )
    section_heading: str | None = Field(
        default=None,
        description="Section heading the cited snippet was taken from, when known.",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed page number, when the source format exposes pagination.",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Stable id of the source chunk — lets an audit trace back to the index entry.",
    )


class DraftAuditResult(BaseModel):
    """Per-draft slice of the release report.

    Pairs the loaded draft path with the per-draft
    :class:`AuditResult`. The ``passed`` flag is denormalised onto
    this row so a Markdown summary can render one ``✓/✗`` per draft
    without re-counting findings. ``draft_path`` is a POSIX string
    (not a ``Path`` object) so the JSON dump is portable across
    operating systems.
    """

    model_config = ConfigDict(extra="forbid")

    draft_path: str = Field(min_length=1)
    section_type: str = Field(min_length=1)
    citation_count: int = Field(ge=0)
    audit_result: AuditResult
    passed: bool


class ReleaseAuditReport(BaseModel):
    """Aggregate report from one :meth:`ReleaseAuditHarness.audit_directory` run.

    ``passed`` is computed from the per-draft results and is the
    single boolean the CLI reads for its exit code. The counters at
    the top level are duplicated for the convenience of release notes
    — the canonical source of truth remains ``draft_results`` and
    ``rows``.

    Why ``sampled_paths`` is recorded explicitly
    --------------------------------------------
    When ``sample_size`` is set, the harness picks a subset and the
    JSON report needs to record *which* subset so a follow-up
    investigation can find the exact drafts that were audited.
    Storing the paths (not just the count) keeps the report
    self-describing.
    """

    model_config = ConfigDict(extra="forbid")

    audit_directory: str = Field(min_length=1)
    sample_size: int | None = None
    sample_seed: int | None = None
    sampled_paths: list[str] = Field(default_factory=list)
    total_drafts: int = Field(ge=0)
    audited_drafts: int = Field(ge=0)
    passed_drafts: int = Field(ge=0)
    failed_drafts: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    unlabeled_citation_count: int = Field(ge=0)
    draft_results: list[DraftAuditResult] = Field(default_factory=list)
    rows: list[CitationAuditRow] = Field(default_factory=list)
    passed: bool

    def render_markdown_summary(self) -> str:
        """Render the report as a Markdown summary suitable for release notes.

        Produces three blocks:

        * **Summary** — counters (total / audited / passed / failed /
          citations / unlabeled).
        * **Per-draft results** — one row per audited draft with the
          pass flag, section type, and citation count.
        * **Citation fidelity table** — one row per citation, with the
          full source-document fingerprint (AC1).

        The Markdown is *deterministic*: same report → byte-equal
        output. The harness tests rely on this.
        """

        verdict = "PASS" if self.passed else "FAIL"
        head = [
            "# Release Audit Report",
            "",
            f"**Verdict:** {verdict}",
            "",
            "## Summary",
            "",
            f"- Audit directory: `{self.audit_directory}`",
            f"- Total drafts on disk: {self.total_drafts}",
            f"- Audited drafts: {self.audited_drafts}",
            f"- Passed drafts: {self.passed_drafts}",
            f"- Failed drafts: {self.failed_drafts}",
            f"- Citations audited: {self.citation_count}",
            f"- Unlabeled citations: {self.unlabeled_citation_count}",
        ]
        if self.sample_size is not None:
            head.extend(
                [
                    f"- Sample size: {self.sample_size}",
                    f"- Sample seed: {self.sample_seed}",
                ]
            )

        per_draft = ["", "## Per-draft Results", ""]
        if self.draft_results:
            per_draft.extend(
                [
                    "| Status | Section | Citations | Draft |",
                    "|--------|---------|-----------|-------|",
                ]
            )
            for r in self.draft_results:
                badge = "PASS" if r.passed else "FAIL"
                per_draft.append(
                    f"| {badge} | {r.section_type} | {r.citation_count} | `{r.draft_path}` |"
                )
        else:
            per_draft.append("_No drafts audited._")

        citation_table = ["", "## Citation Fidelity", ""]
        if self.rows:
            citation_table.extend(
                [
                    "| Status | Programme | Call | Page | Section | Source | Chunk ID | Draft |",
                    "|--------|-----------|------|------|---------|--------|----------|-------|",
                ]
            )
            for row in self.rows:
                page = str(row.page) if row.page is not None else "n/a"
                section = row.section_heading or "n/a"
                title = row.proposal_title or "untitled"
                citation_table.append(
                    f"| {row.source_status} | {row.programme} | {row.call_id} | "
                    f"{page} | {section} | {title} | {row.chunk_id} | "
                    f"`{row.draft_path}` |"
                )
        else:
            citation_table.append(
                "_No citations across audited drafts — investigate before shipping._"
            )

        return "\n".join(head + per_draft + citation_table) + "\n"


class ReleaseAuditHarnessError(Exception):
    """Raised when the harness cannot run (e.g., audit directory missing)."""


class ReleaseAuditHarness:
    """Walk a directory of saved drafts and emit a :class:`ReleaseAuditReport`.

    Stateless — a single instance is safe to share across runs. The
    audit semantics are entirely delegated to :class:`CitationAudit`,
    so the per-draft contract ("missing source_status → ERROR
    finding") is identical whether the operator runs ``eurpe generate
    audit`` for one file or ``eurpe generate audit-release`` for a
    sample of them.

    Why a class rather than module-level functions
    ----------------------------------------------
    A class lets a future caller swap in a custom :class:`CitationAudit`
    (e.g., a release-only flavour that treats ``unused_citation`` as
    ERROR) without touching the harness signature. The private
    helpers stay private so tests exercise the public surface,
    not implementation details.
    """

    def __init__(
        self,
        audit: CitationAudit | None = None,
        renderer: MarkdownCitationRenderer | None = None,
    ) -> None:
        self._audit = audit or CitationAudit()
        self._renderer = renderer or MarkdownCitationRenderer()

    def audit_directory(
        self,
        directory: Path,
        *,
        sample_size: int | None = None,
        seed: int | None = None,
    ) -> ReleaseAuditReport:
        """Walk ``directory`` for ``*.json`` drafts and audit a sample.

        :param directory: Directory containing one or more saved
            :class:`GenerationDraft` JSON files (as produced by
            ``eurpe generate section --output``).
        :param sample_size: Maximum number of drafts to audit. When
            ``None`` (default), every discovered draft is audited.
            When set and smaller than the discovered count, a
            deterministic random subset is selected using ``seed``.
        :param seed: Random seed for the sampling subset. Defaults to
            42 when sampling is in effect and the caller did not
            supply one. Stored on the report for traceability.

        :raises ReleaseAuditHarnessError: when ``directory`` does not
            exist or is not a directory. An empty directory is *not*
            an error: it produces a report with ``audited_drafts=0``
            and ``passed=True`` AND ``failed_drafts=0`` (vacuously
            true). The CLI surface decides whether to treat that as
            an operator mistake; the harness keeps it side-effect-free.
        """

        if not directory.exists() or not directory.is_dir():
            raise ReleaseAuditHarnessError(
                f"audit directory does not exist or is not a directory: {directory}"
            )

        # Deterministic discovery: sort by relative POSIX path so the
        # ordering is stable across operating systems. The harness's
        # determinism guarantee depends on this.
        discovered = sorted(
            (p for p in directory.glob(_DRAFT_GLOB) if p.is_file()),
            key=lambda p: p.relative_to(directory).as_posix(),
        )

        total = len(discovered)
        sampled = self._sample(discovered, sample_size=sample_size, seed=seed)
        effective_seed = seed if seed is not None else _DEFAULT_SAMPLE_SEED

        draft_results: list[DraftAuditResult] = []
        rows: list[CitationAuditRow] = []
        citation_count = 0
        unlabeled_count = 0
        passed_count = 0
        failed_count = 0

        for draft_path in sampled:
            draft = self._load_draft(draft_path)
            rendered = self._renderer.render(draft)
            result = self._audit.audit_rendered(draft, rendered)
            passed = result.passed
            if passed:
                passed_count += 1
            else:
                failed_count += 1

            citation_count += len(draft.citations)
            unlabeled_count += sum(
                1 for c in draft.citations if getattr(c, "source_status", None) is None
            )

            draft_results.append(
                DraftAuditResult(
                    draft_path=draft_path.as_posix(),
                    section_type=draft.section_type.value,
                    citation_count=len(draft.citations),
                    audit_result=result,
                    passed=passed,
                )
            )

            for citation in draft.citations:
                status = getattr(citation, "source_status", None)
                # Unlabeled citations would already be flagged as ERROR
                # findings by CitationAudit; we still emit a row so the
                # release manager sees the offending citation. The
                # row's ``source_status`` field requires a non-empty
                # string, so we surface the literal ``unknown`` marker
                # to keep the row valid AND make the gap visible.
                status_value = status.value if status is not None else "unknown"
                rows.append(
                    CitationAuditRow(
                        draft_path=draft_path.as_posix(),
                        section_type=draft.section_type.value,
                        citation_id=citation.citation_id,
                        source_status=status_value,
                        programme=citation.programme.value,
                        call_id=citation.call_id,
                        proposal_title=citation.proposal_title,
                        section_heading=citation.section_heading,
                        page=citation.page,
                        chunk_id=citation.chunk_id,
                    )
                )

        overall_passed = failed_count == 0

        return ReleaseAuditReport(
            audit_directory=directory.as_posix(),
            sample_size=sample_size,
            sample_seed=effective_seed if sample_size is not None else None,
            sampled_paths=[p.as_posix() for p in sampled],
            total_drafts=total,
            audited_drafts=len(sampled),
            passed_drafts=passed_count,
            failed_drafts=failed_count,
            citation_count=citation_count,
            unlabeled_citation_count=unlabeled_count,
            draft_results=draft_results,
            rows=rows,
            passed=overall_passed,
        )

    # ------------------------------------------------------------------
    # internal helpers — small, named, testable indirectly via public API
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(
        paths: list[Path],
        *,
        sample_size: int | None,
        seed: int | None,
    ) -> list[Path]:
        """Pick the deterministic subset of ``paths`` to audit.

        Contract:
        * ``sample_size is None`` → return ``paths`` unchanged.
        * ``sample_size >= len(paths)`` → return ``paths`` unchanged.
        * ``sample_size < len(paths)`` → seeded ``random.sample`` then
          re-sort by path so the audit walk order is deterministic.
        * ``sample_size < 1`` → raise :class:`ReleaseAuditHarnessError`
          (an operator who asks for zero drafts probably made a typo;
          the safer behaviour is loud).
        """

        if sample_size is None:
            return paths

        if sample_size < 1:
            raise ReleaseAuditHarnessError(f"sample_size must be >= 1 (got {sample_size})")

        if sample_size >= len(paths):
            return paths

        effective_seed = seed if seed is not None else _DEFAULT_SAMPLE_SEED
        # Deterministic release-audit sampling, not security randomness.
        rng = random.Random(effective_seed)  # nosec B311
        picked = rng.sample(paths, sample_size)
        return sorted(picked, key=lambda p: p.as_posix())

    @staticmethod
    def _load_draft(draft_path: Path) -> GenerationDraft:
        """Parse a draft JSON file into a :class:`GenerationDraft`.

        Wraps the JSON / Pydantic errors in a single
        :class:`ReleaseAuditHarnessError` so the CLI can surface one
        consistent message rather than three different exception
        types. The path is included so the operator knows which file
        broke.
        """

        try:
            payload = json.loads(draft_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseAuditHarnessError(
                f"failed to read or parse draft JSON {draft_path}: {exc}"
            ) from exc

        try:
            return GenerationDraft.model_validate(payload)
        except Exception as exc:  # pragma: no cover - pydantic message
            raise ReleaseAuditHarnessError(
                f"draft JSON {draft_path} does not match GenerationDraft schema: {exc}"
            ) from exc


def has_release_blocking_findings(report: ReleaseAuditReport) -> bool:
    """Convenience: ``True`` iff ``report`` has any ERROR-severity finding.

    Hides the per-draft loop from CLI / API callers that only need a
    boolean. Mirrors the per-draft :attr:`AuditResult.passed` shape
    but raises the question one level up.
    """

    return any(
        any(f.severity is AuditSeverity.ERROR for f in d.audit_result.findings)
        for d in report.draft_results
    )
