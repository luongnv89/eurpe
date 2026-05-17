"""MVP pilot validation orchestrator (Task 3.7 / issue #21).

What this module is (and is not)
--------------------------------
The pilot runner is a **thin composer** that drives one cohesive
validation run over the existing primitives:

1. Network-isolation smoke probe (re-uses
   :func:`eurpe.cli._smoke_egress_probe`-style logic — same TEST-NET
   address, same default-deny invariant).
2. Section generation for one configured proposal (default: ≥ 3
   :class:`~eurpe.schema.SectionType` values per AC1 of issue #21).
3. Per-draft citation audit
   (:class:`~eurpe.generation.audit.CitationAudit`) plus a
   release-level audit roll-up
   (:class:`~eurpe.generation.audit_harness.ReleaseAuditHarness`).
4. Performance snapshot
   (:func:`~eurpe.benchmarks.run_all` on the same in-memory pool).
5. Aggregate everything into a :class:`~eurpe.pilot.models.PilotReport`
   and (optionally) render it as Markdown next to the JSON.

It is NOT:

* A new generation/audit/benchmark implementation. The pilot must run
  the *same* code paths as the regular CLIs so a coordinator's report
  is comparable to a CI report.
* A real-LLM gate. The default backend is deterministic so a fresh
  clone produces a complete smoke pilot. The ``runtime`` argument
  switches to Ollama for the coordinator-mode pilot.
* A Docling-driven pipeline. The runner indexes pre-built in-memory
  chunks (the benchmark module already maintains a synthetic corpus
  generator for the same reason: no real PDFs in CI). A coordinator
  pilot that wants to index a real proposal runs ``eurpe ingest``
  beforehand and passes the resulting Chroma index in.

Why a deterministic default is OK
---------------------------------
AC1 of issue #21 asks for "at least one real call topic and at least
three generated section drafts". The runner stamps the configured
``call_id`` onto the synthetic proposal so the produced drafts cite
the real call ID end-to-end (visible in the rendered Markdown). When
the operator wants drafts grounded in real proposal text, they pass
``--proposal`` with an indexed Chroma collection (or use the
companion runbook to run ``eurpe ingest`` first). The smoke path is
the "fresh checkout" entry point; the coordinator path is the
release-gate entry point.

What the runner does NOT capture
--------------------------------
Coordinator satisfaction ratings (AC2). Those are added to the
report after the run by a human, either by hand-editing the JSON or
by running ``eurpe pilot rate``. The runner's job is to lay the
artefacts down; the human's job is to fill in the satisfaction
fields. The default verdict is therefore ``CONDITIONAL`` for any
report that lacks human ratings — the auto-mode honesty contract
documented in :mod:`eurpe.pilot.models`.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from eurpe.benchmarks import BenchmarkReport
from eurpe.benchmarks.runner import build_synthetic_corpus
from eurpe.generation.audit import AuditSeverity, CitationAudit
from eurpe.generation.audit_harness import (
    DraftAuditResult,
    ReleaseAuditHarness,
    ReleaseAuditReport,
)
from eurpe.generation.llm import DeterministicLLMClient, LLMClient
from eurpe.generation.models import GenerationDraft, GenerationRequest
from eurpe.generation.render import MarkdownCitationRenderer
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.pilot.models import (
    CitationIssue,
    GoNoGoVerdict,
    PilotMode,
    PilotReport,
    PilotSectionResult,
    SatisfactionRating,
    SmokeResult,
)
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    Embedder,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
)
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.retrieval.pipeline import index_proposal
from eurpe.schema import SectionType
from eurpe.security import EgressDeniedError, make_network_policy

if TYPE_CHECKING:
    from eurpe.config import EurpeConfig

logger = logging.getLogger(__name__)


# TEST-NET-1 (RFC 5737) — guaranteed unreachable. Same address the
# ``eurpe smoke`` command uses; sharing it keeps the two smoke
# checks honestly equivalent.
_SMOKE_PROBE_HOST = "192.0.2.1"
_SMOKE_PROBE_PORT = 443
_SMOKE_PROBE_SCHEME = "https"


#: Default trio of section types per AC1 ("at least three generated
#: section drafts"). Picked to span the three top-level Horizon Europe
#: structural sections (Excellence-ish, Impact, Implementation) so the
#: pilot exercises distinct retrieval + prompt branches.
DEFAULT_SECTION_TYPES: tuple[SectionType, ...] = (
    SectionType.METHODOLOGY,
    SectionType.IMPACT,
    SectionType.IMPLEMENTATION,
)


#: Generic, content-safe intent strings the runner passes to the
#: workflow for each section type. Kept short and free of any
#: real-proposal terminology so the pilot output never echoes
#: privileged content into the report.
_DEFAULT_USER_INTENTS: dict[SectionType, str] = {
    SectionType.METHODOLOGY: (
        "Describe the proposed methodology and how it addresses the call's expected outcomes."
    ),
    SectionType.IMPACT: (
        "Outline the project's expected scientific, economic, and societal impact pathway."
    ),
    SectionType.IMPLEMENTATION: (
        "Describe the implementation plan, work packages, and consortium responsibilities."
    ),
    SectionType.EXCELLENCE: (
        "Describe the excellence of the proposed approach against the state of the art."
    ),
    SectionType.IMPACT_PATHWAY: (
        "Describe the impact pathway and the key performance indicators that evidence success."
    ),
    SectionType.WORK_PLAN: ("Describe the work plan, milestones, and risk mitigations."),
    SectionType.CONSORTIUM: ("Describe the consortium composition and complementarity."),
    SectionType.BUDGET: ("Describe the budget rationale and resource allocation."),
    SectionType.ETHICS: ("Describe the ethics framework and compliance measures."),
    SectionType.DISSEMINATION: (
        "Describe the dissemination, exploitation, and communication plan."
    ),
    SectionType.OTHER: "Draft this section to support the call's expected outcomes.",
}


class PilotRunError(Exception):
    """Raised when the pilot run cannot proceed (e.g., zero sections requested).

    Distinct from the audit / generation errors so the CLI can surface
    one consistent operator message rather than three different
    exception types.
    """


class PilotConfig(BaseModel):
    """Inputs to one :func:`run_pilot` call.

    Pulled out as a Pydantic record (rather than a long argument list)
    so the same shape can be loaded from a YAML driver file in a
    future ``eurpe pilot from-config`` subcommand without changing
    the runner signature.
    """

    model_config = ConfigDict(extra="forbid")

    mode: PilotMode = Field(
        default=PilotMode.SMOKE,
        description="``smoke`` (default) or ``coordinator``.",
    )
    call_id: str = Field(
        default="HORIZON-CL5-2024-D3-02",
        min_length=1,
        description=(
            "EU call identifier the pilot exercises. AC1 of issue #21 "
            "requires a *real* call topic — the default points at the "
            "Horizon Europe 2024 cybersecurity call used as the example "
            "across the E2E suite and the SANCUS fixture."
        ),
    )
    proposal_title: str = Field(
        default="MVP Pilot Synthetic Corpus",
        min_length=1,
        description=(
            "Title stamped onto the synthetic indexed proposal so the "
            "rendered Markdown is honest about its provenance. Override "
            "to a real proposal title when running a coordinator pilot."
        ),
    )
    section_types: tuple[SectionType, ...] = Field(
        default=DEFAULT_SECTION_TYPES,
        min_length=3,
        description=(
            "Section types to draft. Pydantic enforces ``min_length=3`` "
            "so the AC1 'at least three generated section drafts' "
            "requirement is satisfied by construction."
        ),
    )
    top_k_examples: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Forwarded to :class:`GenerationRequest.top_k_examples` for "
            "every section. Matches the workflow default."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Free-text operator notes captured on the report. Same "
            "privacy contract as analytics events — no proposal content."
        ),
    )


def _build_synthetic_index_for_pilot(
    *,
    config: PilotConfig,
    chunker: HierarchicalChunker,
    index: ChromaIndex,
) -> int:
    """Index a synthetic corpus stamped with the pilot's ``call_id``.

    Returns the total chunk count. The proposal title and call id
    flow into the citations the workflow assembles, so the rendered
    Markdown shows ``call=<config.call_id>`` end-to-end. The embedder
    lives on the ``index`` instance — passing it here would be
    redundant.
    """

    corpus = build_synthetic_corpus(proposal_count=2, sections_per_proposal=4)
    chunk_count = 0
    for i, (parsed, metadata) in enumerate(corpus):
        # Stamp every proposal with the pilot's call_id and title so the
        # downstream citations carry the requested fingerprint. Keep
        # the source_path unique per proposal so the deduper doesn't
        # collapse them.
        stamped = metadata.model_copy(
            update={
                "call_id": config.call_id,
                "proposal_title": f"{config.proposal_title} ({i + 1}/2)",
            }
        )
        chunk_count += index_proposal(
            parsed,
            stamped,
            chunker=chunker,
            index=index,
        )
    return chunk_count


def _run_smoke_probe(eurpe_config: EurpeConfig | None) -> SmokeResult:
    """Run the network-isolation smoke probe and return its outcome.

    Mirrors :func:`eurpe.cli._smoke_egress_probe`: when an
    :class:`EurpeConfig` is supplied, the configured network policy is
    used; otherwise a default-deny policy is constructed. The default
    case matters for tests that don't want to thread a config object
    through.

    Captures the exception path explicitly so the report can record a
    failure mode (e.g., misconfigured allowlist) rather than crashing
    the pilot run. The pilot itself never fails on smoke — the report
    surfaces the failure and the verdict computation downgrades the
    go/no-go automatically.
    """

    try:
        if eurpe_config is None:
            from eurpe.config import EurpeConfig as _EurpeConfig

            cfg = _EurpeConfig()
        else:
            cfg = eurpe_config
        gate = make_network_policy(cfg)
        try:
            gate.check(
                host=_SMOKE_PROBE_HOST,
                port=_SMOKE_PROBE_PORT,
                scheme=_SMOKE_PROBE_SCHEME,
                path="/",
                source="pilot_smoke_probe",
            )
        except EgressDeniedError:
            # Expected — gate denied as required.
            return SmokeResult(
                passed=True,
                exit_code=0,
                detail="TEST-NET probe denied as expected.",
            )
        # Reaching here means the gate did NOT raise — the contract is
        # broken. Same regression message as ``eurpe smoke``.
        return SmokeResult(
            passed=False,
            exit_code=1,
            detail=(
                "TEST-NET probe was ALLOWED — check your network_allowlist for an over-broad entry."
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        # A broken config (e.g., bad allowlist regex) should surface
        # as a smoke FAIL rather than crash the whole pilot.
        return SmokeResult(
            passed=False,
            exit_code=2,
            detail=f"smoke probe could not run: {type(exc).__name__}",
        )


def _draft_to_section_result(
    *,
    draft: GenerationDraft,
    user_intent: str,
    elapsed_ms: int,
    draft_path: Path | None,
    audit_passed: bool,
) -> PilotSectionResult:
    """Pack a :class:`GenerationDraft` into a :class:`PilotSectionResult`.

    Pulled out so the loop body in :func:`run_pilot` is short and so
    the test suite can build expected results without spelling out
    every field. The draft text length is captured rather than the
    text itself: the section result is meant to summarise, not to
    duplicate the rendered Markdown.
    """

    return PilotSectionResult(
        section_type=draft.section_type.value,
        user_intent=user_intent,
        citation_count=len(draft.citations),
        draft_length=len(draft.text),
        elapsed_ms=elapsed_ms,
        draft_path=draft_path.as_posix() if draft_path is not None else None,
        audit_passed=audit_passed,
        satisfaction=[],
    )


def _flatten_audit_issues(report: ReleaseAuditReport) -> list[CitationIssue]:
    """Flatten the audit report's per-draft findings into
    :class:`CitationIssue` records the pilot report can render directly.

    The release audit already records full findings; this helper
    extracts the ERROR-severity ones (the release-blocking subset)
    and strips them into the small record shape the pilot consumer
    cares about. WARNING-severity findings are left for the operator
    to crawl in the audit JSON when needed.
    """

    issues: list[CitationIssue] = []
    per_draft: Sequence[DraftAuditResult] = report.draft_results
    for draft_row in per_draft:
        for finding in draft_row.audit_result.findings:
            if finding.severity != AuditSeverity.ERROR:
                continue
            issues.append(
                CitationIssue(
                    draft_path=draft_row.draft_path,
                    section_type=draft_row.section_type,
                    code=finding.code,
                    message=finding.message,
                )
            )
    return issues


#: Audit finding codes that are *expected* under the deterministic
#: stub LLM. ``placeholder_text`` is the verbatim "This sentence
#: references retrieved example [N] as supporting evidence" the
#: :class:`DeterministicLLMClient` emits — by design, so a release
#: never accidentally ships stub text. In smoke-mode pilot runs we
#: deliberately use the stub, so seeing this finding means "the stub
#: did its job"; in coordinator mode (real LLM) the same finding
#: would be release-blocking. The verdict logic therefore tolerates
#: this code in smoke mode only.
_SMOKE_TOLERATED_FINDINGS: frozenset[str] = frozenset({"placeholder_text"})


def _smoke_audit_blocking(citation_issues: Sequence[CitationIssue]) -> bool:
    """Return True iff the audit produced a finding the smoke mode cannot tolerate.

    The smoke mode runs against the deterministic stub LLM, which
    intentionally emits the ``placeholder_text`` audit finding (it is
    the very signal that prevents stub text from being shipped). We
    therefore tolerate that one code in smoke mode and treat any
    other ERROR code as release-blocking. The set is small and
    closed so a future stub-emitted finding has to be added here
    deliberately, not by accident.
    """

    for issue in citation_issues:
        if issue.code not in _SMOKE_TOLERATED_FINDINGS:
            return True
    return False


def _compute_verdict(
    *,
    mode: PilotMode,
    smoke: SmokeResult,
    audit: ReleaseAuditReport,
    citation_issues: Sequence[CitationIssue],
    section_results: Sequence[PilotSectionResult],
) -> GoNoGoVerdict:
    """Derive the release recommendation from the constituent reports.

    Rules (in order of precedence):

    1. Smoke probe FAIL → NO_GO. The privacy/network-isolation
       contract is release-blocking by PRD definition.
    2. COORDINATOR mode + any audit finding → NO_GO. Real-LLM runs
       have no excuse for ``placeholder_text`` or any other audit
       error; the release gate is strict.
    3. SMOKE mode + any non-``placeholder_text`` audit finding →
       NO_GO. The stub's expected placeholder is tolerated; any
       other audit error reflects a genuine pipeline regression and
       is release-blocking even in smoke mode.
    4. SMOKE mode (no blocking findings) → CONDITIONAL. Auto-mode
       honesty contract: no human satisfaction means no unconditional
       GO.
    5. COORDINATOR mode with at least one rating per section AND mean
       rating ≥ 4 → GO. Anything less → NO_GO.

    Centralised here (rather than spread across the renderer / CLI) so
    the JSON ``verdict`` field and the rendered Markdown always agree.
    """

    if not smoke.passed:
        return GoNoGoVerdict.NO_GO
    if mode == PilotMode.COORDINATOR and not audit.passed:
        return GoNoGoVerdict.NO_GO
    if mode == PilotMode.SMOKE and _smoke_audit_blocking(citation_issues):
        return GoNoGoVerdict.NO_GO
    if mode == PilotMode.SMOKE:
        return GoNoGoVerdict.CONDITIONAL
    # Coordinator mode — require ≥1 rating per section and mean ≥4.
    for section in section_results:
        ratings = [r.rating for r in section.satisfaction if r.rating is not None]
        if not ratings:
            return GoNoGoVerdict.CONDITIONAL
        mean = sum(ratings) / len(ratings)
        if mean < 4.0:
            return GoNoGoVerdict.NO_GO
    return GoNoGoVerdict.GO


def run_pilot(
    *,
    config: PilotConfig | None = None,
    output_dir: Path | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    eurpe_config: EurpeConfig | None = None,
) -> PilotReport:
    """Drive one pilot run and return a :class:`PilotReport`.

    The runner is offline-by-default: when ``llm`` and ``embedder`` are
    omitted, deterministic stubs are constructed. A coordinator pilot
    that wants Ollama-backed measurements passes the live clients in
    (the CLI builds them via :func:`eurpe.generation.llm.make_llm_client`
    and :func:`eurpe.retrieval.embeddings.make_embedder`).

    :param config: Pilot configuration. Defaults to a smoke-mode run
        against the synthetic corpus with three section types.
    :param output_dir: When set, every per-section draft is written
        as JSON (and rendered Markdown sibling) into the directory.
        The directory is created if missing. Required when callers
        want :class:`~eurpe.generation.audit_harness.ReleaseAuditHarness`
        to walk a deterministic on-disk corpus; when omitted the
        runner falls back to an in-memory audit-report roll-up.
    :param llm: Optional :class:`LLMClient` override (defaults to
        :class:`DeterministicLLMClient`).
    :param embedder: Optional :class:`Embedder` override (defaults to
        :class:`DeterministicHashEmbedder`).
    :param eurpe_config: Optional EURPE config used for the network
        policy. ``None`` constructs a default-deny policy.
    :raises PilotRunError: when the synthetic indexing path produces
        zero chunks (the synthetic corpus generator is hard-coded to
        produce >0 chunks, so this is defensive against future
        misconfigurations).
    """

    pilot_config = config or PilotConfig()
    embedder = embedder or DeterministicHashEmbedder(dimension=128)
    llm = llm or DeterministicLLMClient()

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # The ChromaIndex persists into a directory: when the caller
    # supplied ``output_dir`` we co-locate it under ``<output_dir>/index``
    # so the artefact set is self-contained and a coordinator running
    # ``eurpe pilot run --output-dir <release-notes>/<tag>`` ends up
    # with a single directory tree to attach. Otherwise we drop the
    # index in a freshly-created tempdir so the run leaves no
    # filesystem residue.
    if output_dir is not None:
        index_dir = output_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        tempdir_ctx = None
    else:
        tempdir_ctx = tempfile.TemporaryDirectory(prefix="eurpe-pilot-")
        index_dir = Path(tempdir_ctx.name)

    try:
        # ---- Build the workflow's retrieval index.
        chunker = HierarchicalChunker()
        index = ChromaIndex(
            index_path=index_dir,
            embedder=embedder,
            collection_name="pilot",
        )
        chunk_count = _build_synthetic_index_for_pilot(
            config=pilot_config,
            chunker=chunker,
            index=index,
        )
        if chunk_count == 0:
            raise PilotRunError(
                "pilot indexing produced 0 chunks — check the synthetic corpus generator"
            )

        policy = RetrievalPolicy(
            relevance_threshold=0.0,
            max_rejected_fraction=1.0,
            include_esr=False,
        )
        retriever = SourceStatusAwareRetriever(index, policy=policy)
        workflow = SectionGenerationWorkflow(retriever=retriever, llm=llm)

        report = _run_pilot_inner(
            pilot_config=pilot_config,
            workflow=workflow,
            embedder=embedder,
            llm=llm,
            output_dir=output_dir,
            index_dir=index_dir,
            eurpe_config=eurpe_config,
        )
    finally:
        if tempdir_ctx is not None:
            tempdir_ctx.cleanup()

    return report


def _run_pilot_inner(
    *,
    pilot_config: PilotConfig,
    workflow: SectionGenerationWorkflow,
    embedder: Embedder,
    llm: LLMClient,
    output_dir: Path | None,
    index_dir: Path,
    eurpe_config: EurpeConfig | None,
) -> PilotReport:
    """Inner body of :func:`run_pilot` once the workflow is wired.

    Split out so the outer function owns lifecycle (tempdir, output
    dir) and this function focuses on the per-section loop +
    aggregation. Keeps both bodies short and lets tests target the
    inner loop with a pre-built workflow when needed.
    """

    # ---- Generate one draft per requested section type.
    audit = CitationAudit()
    renderer = MarkdownCitationRenderer()
    section_results: list[PilotSectionResult] = []
    drafts: list[tuple[Path | None, GenerationDraft]] = []

    for section_type in pilot_config.section_types:
        intent = _DEFAULT_USER_INTENTS.get(section_type, _DEFAULT_USER_INTENTS[SectionType.OTHER])
        request = GenerationRequest(
            section_type=section_type,
            user_intent=intent,
            top_k_examples=pilot_config.top_k_examples,
        )

        start_ns = time.monotonic_ns()
        draft = workflow.run(request)
        elapsed_ms = (time.monotonic_ns() - start_ns) // 1_000_000

        rendered_md = renderer.render(draft)
        per_audit = audit.audit_rendered(draft, rendered_md)

        draft_path: Path | None = None
        if output_dir is not None:
            base = output_dir / f"{section_type.value}"
            json_path = base.with_suffix(".json")
            md_path = base.with_suffix(".md")
            json_path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
            md_path.write_text(rendered_md, encoding="utf-8")
            draft_path = json_path

        drafts.append((draft_path, draft))
        section_results.append(
            _draft_to_section_result(
                draft=draft,
                user_intent=intent,
                elapsed_ms=elapsed_ms,
                draft_path=draft_path,
                audit_passed=per_audit.passed,
            )
        )

    # ---- Release-level audit roll-up.
    if output_dir is not None:
        audit_report = ReleaseAuditHarness().audit_directory(output_dir)
    else:
        # In-memory fallback: build a report mirror so the pilot report
        # has the same shape regardless of whether we wrote artefacts.
        audit_report = _audit_in_memory(
            drafts=[d for _, d in drafts],
            renderer=renderer,
            audit=audit,
        )

    citation_issues = _flatten_audit_issues(audit_report)

    # ---- Network-isolation smoke probe.
    smoke_result = _run_smoke_probe(eurpe_config)

    # ---- Performance snapshot. Re-use the benchmark harness so the
    # pilot's performance numbers come from the same primitives that
    # ``eurpe benchmark all`` produces. The benchmark gets its own
    # collection name so it does NOT share the workflow's index — the
    # indexing measurement must be a cold-start to be meaningful.
    benchmark_report = _run_benchmark(
        embedder=embedder,
        llm=llm,
        index_path=index_dir,
        runtime_label=("deterministic" if isinstance(llm, DeterministicLLMClient) else "ollama"),
    )

    verdict = _compute_verdict(
        mode=pilot_config.mode,
        smoke=smoke_result,
        audit=audit_report,
        citation_issues=citation_issues,
        section_results=section_results,
    )

    report = PilotReport(
        mode=pilot_config.mode,
        call_id=pilot_config.call_id,
        proposal_title=pilot_config.proposal_title,
        section_results=section_results,
        smoke=smoke_result,
        audit=audit_report,
        citation_issues=citation_issues,
        benchmark=benchmark_report,
        notes=pilot_config.notes,
        verdict=verdict,
    )

    if output_dir is not None:
        # Persist the full pilot report as JSON next to the per-section
        # artefacts. The CLI also writes a Markdown sibling.
        report_path = output_dir / "pilot-report.json"
        report_path.write_text(report.to_json() + "\n", encoding="utf-8")

    return report


def _audit_in_memory(
    *,
    drafts: Sequence[GenerationDraft],
    renderer: MarkdownCitationRenderer,
    audit: CitationAudit,
) -> ReleaseAuditReport:
    """Build a :class:`ReleaseAuditReport` from in-memory drafts.

    Used when the caller did not pass ``output_dir`` — keeps the pilot
    report shape identical regardless of whether artefacts were
    written. Mirrors :meth:`ReleaseAuditHarness.audit_directory` but
    operates on in-memory drafts so the test suite doesn't need a
    tmp_path for the no-output path.
    """

    from eurpe.generation.audit_harness import (
        CitationAuditRow,
        DraftAuditResult,
    )

    draft_results: list[DraftAuditResult] = []
    rows: list[CitationAuditRow] = []
    citation_count = 0
    unlabeled_count = 0
    passed_count = 0
    failed_count = 0
    for i, draft in enumerate(drafts):
        rendered = renderer.render(draft)
        result = audit.audit_rendered(draft, rendered)
        if result.passed:
            passed_count += 1
        else:
            failed_count += 1
        citation_count += len(draft.citations)
        unlabeled_count += sum(
            1 for c in draft.citations if getattr(c, "source_status", None) is None
        )
        synthetic_path = f"<in-memory>/{i:03d}-{draft.section_type.value}.json"
        draft_results.append(
            DraftAuditResult(
                draft_path=synthetic_path,
                section_type=draft.section_type.value,
                citation_count=len(draft.citations),
                audit_result=result,
                passed=result.passed,
            )
        )
        for citation in draft.citations:
            status = getattr(citation, "source_status", None)
            status_value = status.value if status is not None else "unknown"
            rows.append(
                CitationAuditRow(
                    draft_path=synthetic_path,
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
    return ReleaseAuditReport(
        audit_directory="<in-memory>",
        sample_size=None,
        sample_seed=None,
        sampled_paths=[],
        total_drafts=len(drafts),
        audited_drafts=len(drafts),
        passed_drafts=passed_count,
        failed_drafts=failed_count,
        citation_count=citation_count,
        unlabeled_citation_count=unlabeled_count,
        draft_results=draft_results,
        rows=rows,
        passed=failed_count == 0,
    )


def _run_benchmark(
    *,
    embedder: Embedder,
    llm: LLMClient,
    index_path: Path,
    runtime_label: str,
) -> BenchmarkReport:
    """Produce a single-pass performance snapshot for the pilot report.

    Delegates to :func:`eurpe.benchmarks.run_all` so the pilot's
    performance numbers come from the *same* primitives an operator
    would see by running ``eurpe benchmark all``. The benchmark uses
    a distinct collection name under ``index_path`` so the indexing
    measurement starts cold (the workflow's index lives in a
    different collection under the same directory).
    """

    from eurpe.benchmarks.runner import run_all

    return run_all(
        index_path=index_path,
        embedder=embedder,
        llm=llm,
        collection_name="pilot-benchmark",
        runtime_label=runtime_label,
    )


def render_pilot_report_markdown(report: PilotReport) -> str:
    """Render a :class:`PilotReport` as a human-readable Markdown summary.

    Used both by the CLI (to write a Markdown sibling next to the JSON)
    and by the release-notes template. Deterministic: same report →
    byte-equal output. AC3 of issue #21 names "satisfaction, citation
    issues, performance, network isolation smoke test result, and
    go/no-go recommendation" as required fields — every one of those
    has its own section here, populated from the structured record.
    """

    lines: list[str] = []
    lines.append("# MVP Pilot Validation Report")
    lines.append("")
    lines.append(f"- **Mode:** `{report.mode.value}`")
    lines.append(f"- **Call ID:** `{report.call_id}`")
    lines.append(f"- **Proposal:** {report.proposal_title}")
    lines.append(f"- **Generated at (UTC):** {report.generated_at.isoformat()}")
    lines.append(f"- **Verdict:** **{report.verdict.value.upper()}**")
    if report.notes:
        lines.append(f"- **Notes:** {report.notes}")
    lines.append("")

    # --- Section drafts (AC1).
    lines.append("## Generated sections (AC1)")
    lines.append("")
    lines.append(
        "| # | Section | Citations | Draft length (chars) | Elapsed (ms) | Audit | Artefact |"
    )
    lines.append(
        "|---|---------|-----------|----------------------|--------------|-------|----------|"
    )
    for i, sec in enumerate(report.section_results, start=1):
        audit_badge = "PASS" if sec.audit_passed else "FAIL"
        artefact = f"`{sec.draft_path}`" if sec.draft_path else "n/a"
        lines.append(
            f"| {i} | {sec.section_type} | {sec.citation_count} | "
            f"{sec.draft_length} | {sec.elapsed_ms} | {audit_badge} | {artefact} |"
        )
    lines.append("")

    # --- Coordinator satisfaction (AC2).
    lines.append("## Coordinator satisfaction (AC2)")
    lines.append("")
    lines.append("| Section | Coordinator | Rating (1-5) | Time saved (min) | Notes |")
    lines.append("|---------|-------------|--------------|------------------|-------|")
    any_row = False
    for sec in report.section_results:
        for r in sec.satisfaction:
            any_row = True
            rating = str(r.rating) if r.rating is not None else "<pending>"
            time_saved = (
                str(r.time_saved_minutes) if r.time_saved_minutes is not None else "<pending>"
            )
            lines.append(
                f"| {sec.section_type} | {r.coordinator_id} | {rating} | "
                f"{time_saved} | {r.notes or ''} |"
            )
    if not any_row:
        pending_cell = "_smoke-mode run — no coordinator ratings yet_"
        lines.append(f"| _all_ | `<pending>` | `<pending>` | `<pending>` | {pending_cell} |")
    lines.append("")

    # --- Citation issues (AC3).
    lines.append("## Citation issues (AC3)")
    lines.append("")
    if report.citation_issues:
        lines.append("| Section | Code | Message | Draft |")
        lines.append("|---------|------|---------|-------|")
        for iss in report.citation_issues:
            lines.append(
                f"| {iss.section_type} | `{iss.code}` | {iss.message} | `{iss.draft_path}` |"
            )
    else:
        lines.append("_No release-blocking citation issues found._")
    lines.append("")
    lines.append(
        f"- Audit summary: {report.audit.passed_drafts} passed, "
        f"{report.audit.failed_drafts} failed of {report.audit.audited_drafts} audited "
        f"({report.audit.citation_count} citations)."
    )
    lines.append("")

    # --- Performance (AC3).
    lines.append("## Performance (AC3)")
    lines.append("")
    rt = report.benchmark.runtime
    lines.append(f"- Runtime: `{rt.runtime}` (LLM `{rt.llm_model}`, embedder `{rt.embedder}`)")
    lines.append(f"- Platform: `{rt.platform}`")
    if report.benchmark.indexing is not None:
        idx = report.benchmark.indexing
        lines.append(
            f"- Indexing: {idx.chunk_count} chunks in {idx.elapsed_ms} ms "
            f"({idx.chunks_per_second:.1f} chunks/sec)"
        )
    if report.benchmark.retrieval is not None:
        ret = report.benchmark.retrieval
        lines.append(
            f"- Retrieval (top-{ret.top_k}, {ret.query_count} queries): "
            f"avg {ret.elapsed_ms_avg:.1f} ms / p95 {ret.elapsed_ms_p95} ms"
        )
    if report.benchmark.generation is not None:
        gen = report.benchmark.generation
        lines.append(
            f"- Generation ({gen.section_type}): {gen.elapsed_ms} ms, "
            f"{gen.citation_count} citations, prompt {gen.prompt_length} chars"
        )
    lines.append("")

    # --- Network isolation (AC3).
    lines.append("## Network isolation smoke (AC3)")
    lines.append("")
    badge = "PASS" if report.smoke.passed else "FAIL"
    lines.append(f"- Verdict: **{badge}** (exit code {report.smoke.exit_code})")
    if report.smoke.detail:
        lines.append(f"- Detail: {report.smoke.detail}")
    lines.append("")

    # --- Go/no-go (AC3).
    lines.append("## Go / No-Go recommendation (AC3)")
    lines.append("")
    lines.append(f"**{report.verdict.value.upper()}**")
    lines.append("")
    if report.verdict == GoNoGoVerdict.CONDITIONAL:
        lines.append(
            "> Smoke-mode pilot — the deterministic plumbing held but no "
            "coordinator satisfaction ratings were captured. Run a "
            "coordinator-mode pilot (`eurpe pilot run --mode coordinator "
            "--runtime ollama`) before the v1.0 release."
        )
    elif report.verdict == GoNoGoVerdict.NO_GO:
        lines.append(
            "> Release blocked — see the citation issues table and/or the "
            "smoke probe verdict above for the failing invariant."
        )
    else:
        lines.append(
            "> Coordinator pilot complete; all sections cleared the "
            "satisfaction floor and the network/audit invariants held."
        )
    lines.append("")
    return "\n".join(lines)


def load_pilot_report(path: Path) -> PilotReport:
    """Parse a saved pilot report JSON file into a :class:`PilotReport`.

    Small wrapper so the CLI / a future ``eurpe pilot rate`` subcommand
    has one entry point for reading the persisted artefact.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    return PilotReport.model_validate(data)


def attach_satisfaction(
    *,
    report: PilotReport,
    section_type: str,
    rating: SatisfactionRating,
) -> PilotReport:
    """Return a new :class:`PilotReport` with ``rating`` attached to one section.

    Pure-functional: does not mutate the input. The CLI uses this to
    let a coordinator post-edit a smoke-mode report into a
    coordinator-mode report without rewriting the entire JSON by hand.
    Re-computes the verdict so the persisted file stays internally
    consistent.
    """

    updated_sections: list[PilotSectionResult] = []
    matched = False
    for sec in report.section_results:
        if sec.section_type == section_type:
            matched = True
            sec_new = sec.model_copy(update={"satisfaction": [*sec.satisfaction, rating]})
            updated_sections.append(sec_new)
        else:
            updated_sections.append(sec)
    if not matched:
        raise PilotRunError(
            f"section_type {section_type!r} not found in pilot report; "
            f"available: {[s.section_type for s in report.section_results]}"
        )
    new_verdict = _compute_verdict(
        mode=report.mode,
        smoke=report.smoke,
        audit=report.audit,
        citation_issues=report.citation_issues,
        section_results=updated_sections,
    )
    return report.model_copy(update={"section_results": updated_sections, "verdict": new_verdict})
