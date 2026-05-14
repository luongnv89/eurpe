"""Citation audit — release-blocking checks for source-status compliance.

This module exists for one reason: a generated draft that *looks*
polished but silently drops a source-status label would mislead the
user about whether they are quoting a funded pattern or a cautionary
rejected proposal. The PRD calls source-status preservation a
"release-blocking invariant"; the audit is the automated check
:class:`~eurpe.generation.cli` runs after every generation to make
sure no draft escapes with stripped, missing, or hallucinated labels.

What the audit checks
---------------------
**Errors** (release-blocking — exit 1, fail CI):

* ``missing_status`` — a citation lacks a non-empty
  :class:`~eurpe.schema.SourceStatus` value. Pydantic already enforces
  this at construction, but a defensive check via ``model_construct``
  catches the case where someone bypassed validation.
* ``empty_programme`` — a citation's programme is blank.
* ``empty_call_id`` — call_id is missing or blank after strip.
  (Pydantic enforces ``min_length=1`` already; this catches bypass.)
* ``marker_without_citation`` — text contains ``[N]`` but no citation
  has ``citation_id == N``. The bounded regex used by the workflow
  (``\\d{1,2}``) silently skipped ``[100]`` and similar; this audit
  uses an unbounded regex (``\\d+``) so the hallucination is caught.
* ``duplicate_citation_id`` — two citations share the same
  ``citation_id``. The renderer would emit a confusing reference list;
  the workflow should never produce this.
* ``non_sequential_citation_ids`` — citation ids must be ``1..N`` with
  no gaps. The workflow's prompt builder produces sequential ids; a
  gap means something corrupted the list between build and assembly.
* ``bad_render`` — only fired by :meth:`audit_rendered`. The rendered
  Markdown does NOT contain the expected
  :data:`~eurpe.generation.render.STATUS_BADGE` for at least one cited
  source. This is the signal that the renderer accidentally dropped a
  badge — the second line of defence behind the renderer's own tests.

**Warnings** (advisory — recorded but do not fail the audit):

* ``unused_citation`` — a citation is in ``draft.citations`` but no
  ``[N]`` marker references it. Common when the LLM picked fewer
  pieces of evidence than the workflow retrieved; not a bug.
* ``empty_snippet`` — a citation's snippet is blank. The renderer
  copes (it doesn't print the snippet anyway) but it points at a
  malformed retrieval result upstream.

The two audit entry points
--------------------------
* :meth:`CitationAudit.audit_draft` — checks the structured draft
  only. Cheap, runs without the renderer.
* :meth:`CitationAudit.audit_rendered` — runs the same checks PLUS
  verifies the rendered Markdown contains the badges and the inline
  ``[N]`` markers from the draft text. This is the form
  :class:`~eurpe.generation.cli` uses by default after every
  generation.

Why ``AuditResult.passed`` is computed
--------------------------------------
``passed`` is ``True`` iff there are no ERROR findings. Warnings do
not block release because they describe quality issues, not safety
issues. Callers that want to be paranoid can check
``len(result.warnings) == 0`` themselves; the default contract is
"errors fail, warnings inform".
"""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from eurpe.generation.models import CitationRef, GenerationDraft
from eurpe.generation.render import STATUS_BADGE
from eurpe.schema import SourceStatus

#: Unbounded marker regex. The workflow uses a bounded ``\\d{1,2}``
#: regex which silently drops ``[100]``-style hallucinations; the audit
#: uses ``\\d+`` so we can fail explicitly on out-of-range markers.
#: A previous reviewer flagged this gap on PR #41.
_AUDIT_MARKER = re.compile(r"\[(\d+)\]")


class AuditSeverity(StrEnum):
    """Two-tier severity for audit findings.

    ``ERROR`` is release-blocking; ``WARNING`` is advisory. Kept narrow
    on purpose so a caller can switch on ``severity is ERROR`` rather
    than parsing strings.
    """

    ERROR = "error"
    WARNING = "warning"


class AuditFinding(BaseModel):
    """One issue surfaced by the audit.

    ``code`` is a stable, machine-readable identifier (e.g.,
    ``missing_status``) that callers can switch on without parsing the
    human-readable ``message``. ``citation_id`` is the offending
    citation's id when applicable, or ``None`` when the finding is
    about the draft as a whole (e.g., ``bad_render``).

    ``frozen=True`` so a list of findings can be hashed or
    set-deduplicated; ``extra="forbid"`` matches the project-wide
    convention from :mod:`eurpe.schema`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: AuditSeverity
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    citation_id: int | None = None


class AuditResult(BaseModel):
    """Aggregate result of one audit run.

    Carries every finding (errors AND warnings) and a single boolean
    ``passed`` so callers don't have to count error severities
    themselves. The two filtered properties ``errors`` and ``warnings``
    are convenience views — the canonical source of truth is
    ``findings``.
    """

    model_config = ConfigDict(extra="forbid")

    findings: list[AuditFinding] = Field(default_factory=list)
    passed: bool

    @property
    def errors(self) -> list[AuditFinding]:
        """Findings with severity ERROR. Computed each call (cheap)."""

        return [f for f in self.findings if f.severity is AuditSeverity.ERROR]

    @property
    def warnings(self) -> list[AuditFinding]:
        """Findings with severity WARNING. Computed each call (cheap)."""

        return [f for f in self.findings if f.severity is AuditSeverity.WARNING]


class CitationAudit:
    """Inspect a :class:`GenerationDraft` for source-status compliance.

    Stateless — a single instance is safe to share across calls. The
    two public methods (:meth:`audit_draft` and :meth:`audit_rendered`)
    return :class:`AuditResult` records that callers serialise to JSON,
    print to stderr, or use to drive an exit code.

    Why a class rather than module-level functions
    ----------------------------------------------
    A class lets a future audit grow optional configuration (e.g.,
    "treat unused_citation as ERROR for releases, WARNING for drafts")
    without breaking the call sites. The private helpers stay private
    so tests exercise the public surface, not implementation details.
    """

    def audit_draft(self, draft: GenerationDraft) -> AuditResult:
        """Check the structured draft. Does not render."""

        findings: list[AuditFinding] = []
        findings.extend(self._check_citations(draft.citations))
        findings.extend(self._check_markers(draft.text, draft.citations))
        findings.extend(self._check_unused_citations(draft.text, draft.citations))
        return self._build_result(findings)

    def audit_rendered(
        self,
        draft: GenerationDraft,
        rendered_markdown: str,
    ) -> AuditResult:
        """Check the structured draft AND verify the rendered Markdown.

        Runs the same checks as :meth:`audit_draft`, then layers two
        additional render-time checks:

        * Every cited source must contribute its
          :data:`~eurpe.generation.render.STATUS_BADGE` to the rendered
          output. A missing badge means the renderer dropped a label.
        * Every ``[N]`` marker present in ``draft.text`` must also
          appear in the rendered output (the renderer must not strip
          inline markers).
        """

        findings: list[AuditFinding] = []
        findings.extend(self._check_citations(draft.citations))
        findings.extend(self._check_markers(draft.text, draft.citations))
        findings.extend(self._check_unused_citations(draft.text, draft.citations))
        findings.extend(self._check_rendered(draft, rendered_markdown))
        return self._build_result(findings)

    # ------------------------------------------------------------------
    # internal checks — each returns a list of findings (possibly empty)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_citations(citations: list[CitationRef]) -> list[AuditFinding]:
        """Per-citation field-level checks (status, programme, ids)."""

        findings: list[AuditFinding] = []

        seen_ids: dict[int, int] = {}
        for citation in citations:
            cid = citation.citation_id

            # Track duplicates as we go; one finding per duplicate id.
            seen_ids[cid] = seen_ids.get(cid, 0) + 1

            # ``source_status`` is the release-blocking invariant.
            # ``getattr`` with a default of None covers the
            # ``model_construct`` bypass case the AC2 keystone test
            # exercises (Pydantic normally enforces non-None at
            # construction).
            status = getattr(citation, "source_status", None)
            if status is None:
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="missing_status",
                        message=(
                            f"Citation [{cid}] has no source_status — every citation "
                            "MUST carry a non-empty status tag (PRD invariant)."
                        ),
                        citation_id=cid,
                    )
                )

            programme = getattr(citation, "programme", None)
            programme_value = (
                programme.value
                if hasattr(programme, "value") and programme is not None
                else programme
            )
            if programme is None or (
                isinstance(programme_value, str) and not programme_value.strip()
            ):
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="empty_programme",
                        message=f"Citation [{cid}] has an empty or missing programme.",
                        citation_id=cid,
                    )
                )

            call_id = getattr(citation, "call_id", "") or ""
            if not call_id.strip():
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="empty_call_id",
                        message=f"Citation [{cid}] has an empty or missing call_id.",
                        citation_id=cid,
                    )
                )

            snippet = getattr(citation, "snippet", "") or ""
            if not snippet.strip():
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.WARNING,
                        code="empty_snippet",
                        message=f"Citation [{cid}] has an empty snippet.",
                        citation_id=cid,
                    )
                )

        # Duplicate-id findings — emitted once per offending id.
        for cid, count in seen_ids.items():
            if count > 1:
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="duplicate_citation_id",
                        message=(
                            f"Citation id [{cid}] appears {count} times — ids must "
                            "be unique within a draft."
                        ),
                        citation_id=cid,
                    )
                )

        # Non-sequential id check. The prompt builder always emits
        # 1..N; a gap means something dropped a citation between build
        # and assembly. Empty list is allowed (no citations is a valid
        # state for a draft with no retrieved evidence).
        if citations:
            ids = sorted(c.citation_id for c in citations)
            expected = list(range(1, len(ids) + 1))
            if ids != expected:
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="non_sequential_citation_ids",
                        message=(
                            f"Citation ids must be consecutive 1..N; got {ids} "
                            f"(expected {expected})."
                        ),
                    )
                )

        return findings

    @staticmethod
    def _check_markers(
        text: str,
        citations: list[CitationRef],
    ) -> list[AuditFinding]:
        """Every ``[N]`` in ``text`` must reference a real citation id.

        Uses :data:`_AUDIT_MARKER` (unbounded ``\\d+``) rather than the
        workflow's bounded regex so ``[100]`` and similar hallucinations
        surface here even though the workflow's validator silently
        ignored them.
        """

        findings: list[AuditFinding] = []
        valid_ids = {c.citation_id for c in citations}
        for match in _AUDIT_MARKER.finditer(text):
            n = int(match.group(1))
            if n not in valid_ids:
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="marker_without_citation",
                        message=(
                            f"Text contains marker [{n}] but no citation has that id "
                            f"(valid ids: {sorted(valid_ids) if valid_ids else 'none'})."
                        ),
                        citation_id=n,
                    )
                )
        return findings

    @staticmethod
    def _check_unused_citations(
        text: str,
        citations: list[CitationRef],
    ) -> list[AuditFinding]:
        """Citations in the list with no ``[N]`` marker → WARNING."""

        emitted = {int(m.group(1)) for m in _AUDIT_MARKER.finditer(text)}
        findings: list[AuditFinding] = []
        for citation in citations:
            if citation.citation_id not in emitted:
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.WARNING,
                        code="unused_citation",
                        message=(
                            f"Citation [{citation.citation_id}] is attached to the "
                            "draft but never referenced inline in the text."
                        ),
                        citation_id=citation.citation_id,
                    )
                )
        return findings

    @staticmethod
    def _check_rendered(
        draft: GenerationDraft,
        rendered_markdown: str,
    ) -> list[AuditFinding]:
        """Render-time checks: badges present, inline markers preserved."""

        findings: list[AuditFinding] = []
        # Track which badges we've already complained about so a draft
        # with three citations of the same status doesn't produce three
        # identical findings.
        reported_status: set[SourceStatus] = set()
        for citation in draft.citations:
            status = getattr(citation, "source_status", None)
            if status is None or status in reported_status:
                # Missing status is already flagged by _check_citations;
                # avoid double-reporting.
                continue
            badge = STATUS_BADGE.get(status)
            if badge is None or badge not in rendered_markdown:
                reported_status.add(status)
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="bad_render",
                        message=(
                            f"Rendered output does not contain the expected status "
                            f"badge {badge!r} for source_status={status.value!r}."
                        ),
                        citation_id=citation.citation_id,
                    )
                )

        # Inline marker preservation: count ``[N]`` occurrences in the
        # rendered body (everything BEFORE the ``## References``
        # section header) and compare against the draft-text counts.
        # The references block legitimately contains ``[N]`` markers
        # too, so isolating the body is the only honest check — a
        # whole-document count would always pass because the references
        # list adds one occurrence per citation.
        body_split = rendered_markdown.split("## References", 1)
        rendered_body = body_split[0]
        text_counts = Counter(
            int(m.group(1)) for m in _AUDIT_MARKER.finditer(draft.text)
        )
        body_counts = Counter(
            int(m.group(1)) for m in _AUDIT_MARKER.finditer(rendered_body)
        )
        for n, expected in text_counts.items():
            if body_counts.get(n, 0) < expected:
                findings.append(
                    AuditFinding(
                        severity=AuditSeverity.ERROR,
                        code="bad_render",
                        message=(
                            f"Rendered output is missing inline marker [{n}] "
                            f"(draft text contains {expected} occurrence(s); "
                            f"rendered body contains "
                            f"{body_counts.get(n, 0)})."
                        ),
                        citation_id=n,
                    )
                )
        return findings

    @staticmethod
    def _build_result(findings: list[AuditFinding]) -> AuditResult:
        """Wrap findings in an :class:`AuditResult` and compute ``passed``."""

        passed = not any(f.severity is AuditSeverity.ERROR for f in findings)
        return AuditResult(findings=findings, passed=passed)
