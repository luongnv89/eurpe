"""Pydantic models for the MVP pilot validation report.

This module defines the durable on-disk shape of a pilot run: what was
exercised, what came back, and what a coordinator still owes the
release. The PRD's MVP success criterion is "coordinators can generate
real proposal sections with ≥4/5 satisfaction on at least one active
call, with citations that clearly label funded/rejected/ESR source
status" (``prd.md`` line 173); the pilot report is the artefact that
proves (or denies) that criterion at release time.

Why a dedicated models module
-----------------------------
The pilot run composes three existing reports — the citation audit
:class:`~eurpe.generation.audit_harness.ReleaseAuditReport`, the
benchmark :class:`~eurpe.benchmarks.BenchmarkReport`, and a smoke
result — into one cohesive deliverable. Embedding the composition as
typed Pydantic records keeps the JSON output explicit, gives Markdown
rendering a single source of truth, and lets the test suite assert on
named fields rather than ad-hoc dictionaries.

What this module does NOT contain
---------------------------------
* Orchestration — that lives in :mod:`eurpe.pilot.runner`.
* CLI wiring — that lives in :mod:`eurpe.pilot.cli`.
* User satisfaction scoring logic — satisfaction is collected from a
  human coordinator post-run; the model carries a list of optional
  :class:`SatisfactionRating` records and lets the renderer flag them
  ``<pending>`` until filled in.

Auto-mode honesty
-----------------
The MVP pilot's AC2 ("Users rate each draft and report approximate
time saved against manual drafting") requires real human input. An
automated pilot run cannot fabricate that data without invalidating
the release gate, so the model treats ``satisfaction`` as an
*optional* per-section field and ``mode`` as a load-bearing label
that distinguishes ``smoke`` (deterministic, no coordinator) from
``coordinator`` (real human ratings present). The CLI renders a
``conditional`` go/no-go verdict when satisfaction is missing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from eurpe.benchmarks import BenchmarkReport
from eurpe.generation.audit_harness import ReleaseAuditReport


class PilotMode(StrEnum):
    """Identifier for the *kind* of pilot run captured in a report.

    The MVP go/no-go gate needs to know whether the report was
    produced by an automated smoke run (deterministic backends, no
    coordinator) or a real-coordinator pilot (Ollama + real human
    ratings). Conflating the two would let a smoke run masquerade as
    a release sign-off; the explicit enum prevents that.
    """

    SMOKE = "smoke"
    COORDINATOR = "coordinator"


class GoNoGoVerdict(StrEnum):
    """Closed vocabulary for the pilot's release recommendation.

    ``GO`` and ``NO_GO`` are the only terminal verdicts a coordinator
    pilot can render. ``CONDITIONAL`` is the verdict an automated
    smoke run produces by construction (no human ratings yet) and is
    explicitly NOT release-approving — it means "the deterministic
    plumbing held; final go/no-go pending coordinator pilot".
    """

    GO = "go"
    NO_GO = "no_go"
    CONDITIONAL = "conditional"


class SmokeResult(BaseModel):
    """Outcome of the network-isolation smoke probe.

    The pilot run shells ``eurpe smoke`` (or invokes the same probe
    in-process) and records the exit code. AC3 of issue #21 names
    "network isolation smoke test result" as a required pilot-report
    field — this record is the typed form of that requirement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool = Field(
        description=(
            "True when the smoke probe denied the TEST-NET egress as "
            "expected. False is release-blocking; the pilot run "
            "surfaces the failure in the rendered Markdown report."
        ),
    )
    exit_code: int = Field(
        ge=0,
        description=(
            "Exit code returned by the smoke probe. ``0`` is the "
            "expected success path; non-zero values map to the "
            "regression message in :func:`eurpe.cli._smoke_egress_probe`."
        ),
    )
    detail: str = Field(
        default="",
        description=(
            "Short, content-safe human note (e.g., 'TEST-NET probe "
            "denied' or 'config file missing'). Must not echo proposal "
            "content; same privacy contract as analytics events."
        ),
    )


class SatisfactionRating(BaseModel):
    """One coordinator's rating of one generated section draft.

    ``rating`` follows the PRD's 1–5 scale (``prd.md`` line 173); a
    rating below 4 is a release-blocker. ``time_saved_minutes`` is
    the coordinator's own estimate vs. drafting the same section
    manually — the AC2 "approximate time saved" field. Both are
    optional because a smoke-mode run produces no human input; the
    renderer treats unset values as ``<pending>``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinator_id: str = Field(
        min_length=1,
        description=(
            "Anonymous coordinator identifier (e.g., ``coord-a``). "
            "MUST NOT be a real name, email, or any other identifier "
            "that could deanonymise the rater."
        ),
    )
    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "1–5 Likert satisfaction rating. ``None`` means 'not yet "
            "captured' (smoke mode) — the renderer prints ``<pending>``."
        ),
    )
    time_saved_minutes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Approximate minutes saved against manual drafting. ``None`` means 'not yet captured'."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Short coordinator note (e.g., 'wording too academic'). "
            "Same privacy contract as analytics events — no proposal "
            "content, prompt text, or generated draft snippets."
        ),
    )


class CitationIssue(BaseModel):
    """One specific citation-fidelity issue surfaced by the audit harness.

    The :class:`~eurpe.generation.audit_harness.ReleaseAuditReport`
    already records full audit findings; this record flattens the
    subset a pilot reader cares about so the rendered Markdown can
    list "what went wrong" without forcing the reader to crawl the
    audit JSON.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_path: str = Field(min_length=1)
    section_type: str = Field(min_length=1)
    code: str = Field(
        min_length=1,
        description=(
            "Audit finding code (e.g., ``missing_source_status``, "
            "``placeholder_text``). Stable identifier the audit module owns."
        ),
    )
    message: str = Field(
        min_length=1,
        description="Short human-readable message from the audit finding.",
    )


class PilotSectionResult(BaseModel):
    """Per-section result of a pilot run.

    Records what the pilot driver asked for, what came back, and
    where the artefacts live on disk. The same JSON shape is used in
    both smoke and coordinator modes; the satisfaction list is
    populated only in coordinator mode.
    """

    model_config = ConfigDict(extra="forbid")

    section_type: str = Field(
        min_length=1,
        description="``SectionType.value`` exercised for this draft.",
    )
    user_intent: str = Field(
        min_length=1,
        description=(
            "Free-text intent the driver passed to the workflow. Kept "
            "short and generic ('Describe the proposed methodology for "
            "this work') so the pilot run does not echo proposal "
            "content into the report."
        ),
    )
    citation_count: int = Field(
        ge=0,
        description="Number of citations attached to the produced draft.",
    )
    draft_length: int = Field(
        ge=0,
        description="Character length of the produced draft text.",
    )
    elapsed_ms: int = Field(
        ge=0,
        description=(
            "Wall-clock milliseconds for the per-section generation "
            "call. Aggregated into the pilot's performance summary."
        ),
    )
    draft_path: str | None = Field(
        default=None,
        description=(
            "POSIX path to the rendered Markdown draft, when "
            "``--output-dir`` is set. ``None`` for in-memory runs."
        ),
    )
    audit_passed: bool = Field(
        description=(
            "Per-section audit verdict. ``False`` means the section "
            "draft has at least one finding above WARNING; ``True`` "
            "means no findings of any severity."
        ),
    )
    satisfaction: list[SatisfactionRating] = Field(
        default_factory=list,
        description=(
            "One :class:`SatisfactionRating` per coordinator who rated "
            "this section. Empty in smoke mode."
        ),
    )


class PilotReport(BaseModel):
    """Aggregate report from one pilot validation run.

    Designed to be persisted as JSON next to a Markdown summary so the
    release notes carry both a machine-readable record and a
    human-readable artefact. ``mode`` is the load-bearing field that
    distinguishes a smoke pilot from a coordinator pilot; ``verdict``
    is what the release gate reads.
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp the pilot completed.",
    )
    mode: PilotMode = Field(
        description=(
            "``smoke`` for the deterministic-stub default; "
            "``coordinator`` for a real-LLM run with human ratings."
        ),
    )
    call_id: str = Field(
        min_length=1,
        description=(
            "EU call identifier the pilot exercised (e.g., "
            "``HORIZON-CL5-2024-D3-02``). AC1 of issue #21: 'at least "
            "one real call topic'."
        ),
    )
    proposal_title: str = Field(
        min_length=1,
        description="Title of the proposal used as the indexed evidence base.",
    )
    section_results: list[PilotSectionResult] = Field(
        default_factory=list,
        description=(
            "One :class:`PilotSectionResult` per generated section. "
            "AC1 of issue #21 requires at least three section drafts."
        ),
    )
    smoke: SmokeResult = Field(
        description="Network-isolation smoke probe result.",
    )
    audit: ReleaseAuditReport = Field(
        description=(
            "Citation-fidelity audit of every section produced by "
            "this pilot run. Drives the ``citation issues`` field of "
            "the rendered Markdown report."
        ),
    )
    citation_issues: list[CitationIssue] = Field(
        default_factory=list,
        description=(
            "Flat list of citation-fidelity issues (derived from "
            "``audit``). Stored explicitly so the JSON consumer does "
            "not have to re-derive it from per-draft findings."
        ),
    )
    benchmark: BenchmarkReport = Field(
        description=(
            "Performance snapshot taken during the same pilot run. "
            "Drives the ``performance`` field of the rendered report."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Free-text operator notes (e.g., 'ran on M1 Air, Ollama "
            "warm cache'). Same privacy contract as analytics events."
        ),
    )
    verdict: GoNoGoVerdict = Field(
        description=(
            "Release recommendation derived from the constituent "
            "reports. Smoke-mode runs render ``CONDITIONAL`` by "
            "construction; coordinator-mode runs render ``GO`` only "
            "when every section's mean satisfaction is ≥4 and no "
            "release-blocking findings remain."
        ),
    )

    def to_json(self) -> str:
        """Render the report as pretty-printed JSON.

        Mirrors the
        :meth:`~eurpe.benchmarks.BenchmarkReport.to_json` convenience
        on the embedded benchmark report so callers do not have to
        duplicate the ``model_dump(mode='json')`` + ``indent=2``
        recipe. Pretty printing keeps the file diff-friendly across
        release runs.
        """

        import json

        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)
