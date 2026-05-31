"""Typer commands for ``eurpe generate ...``.

Two subcommands today:

* ``eurpe generate section`` drives the
  :class:`~eurpe.generation.SectionGenerationWorkflow` end-to-end:

  1. Load config, ensure runtime dirs exist, build the embedder /
     index / retriever (same pattern used by ``eurpe index query``).
  2. Build the LLM client via :func:`~eurpe.generation.make_llm_client`
     so the offline-fallback path is honoured.
  3. Build the workflow, run it, print a draft summary to stdout
     (form depends on ``--render``).
  4. Render the draft to Markdown via
     :class:`~eurpe.generation.MarkdownCitationRenderer`.
  5. Run :class:`~eurpe.generation.CitationAudit` against the draft +
     rendered Markdown unless ``--no-audit`` is passed. Audit errors
     print to stderr and exit 1 (release-blocking by design — see
     ``audit.py`` module docstring for the rationale).
  6. If ``--output`` is set, write the chosen artefacts atomically.

* ``eurpe generate audit`` reads back a previously dumped
  :class:`~eurpe.generation.GenerationDraft` JSON, re-renders it,
  runs the audit, and exits 0 (clean) or 1 (failures). This is the
  CI-friendly entry point for re-checking saved drafts without
  re-running the LLM.

The command lives in its own module so the ``eurpe.cli`` top-level
file stays thin (matching the convention used for ``ingestion`` and
``retrieval``). It is mounted onto the top-level Typer in
``eurpe.cli`` as a sub-Typer at ``generate``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from eurpe.analytics import EventType, ExportEvent, make_analytics_logger
from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.generation.audit import AuditResult, CitationAudit
from eurpe.generation.audit_harness import (
    ReleaseAuditHarness,
    ReleaseAuditHarnessError,
    ReleaseAuditReport,
)
from eurpe.generation.critic import CriticAgent
from eurpe.generation.critic_loop import (
    MAX_ITERATIONS_CEILING,
    CriticLoopWorkflow,
)
from eurpe.generation.errors import GenerationError, LLMUnavailableError
from eurpe.generation.llm import make_llm_client
from eurpe.generation.models import GenerationDraft, GenerationRequest
from eurpe.generation.profiles import DraftingProfile, load_profile
from eurpe.generation.render import MarkdownCitationRenderer
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.ingestion.errors import IngestionError
from eurpe.intake import (
    TopicContext,
    extract_topic_context_from_pdf,
    extract_topic_context_from_text,
)
from eurpe.retrieval import (
    ChromaIndex,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
    make_embedder,
)
from eurpe.schema import Programme, SectionType
from eurpe.security import SecurityError

# A sub-Typer so the CLI surface is ``eurpe generate section``. Wired
# into the top-level app in :mod:`eurpe.cli`.
generate_app = typer.Typer(
    name="generate",
    help="Generate a draft proposal section from indexed evidence.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_context(value: str, *, flag: str = "--context") -> str:
    """Resolve a flag value: literal text or ``@path/to/file`` reference.

    The ``@``-prefixed file form mirrors a common CLI convention
    (curl, gh) and avoids the awkwardness of pasting multi-paragraph
    call text on a command line. A literal ``@`` at start can be
    escaped by doubling (``@@`` → ``@``) for the rare user who
    wants verbatim ``@``-prefixed text.

    ``flag`` names the user-facing option in the error message so the
    helper can be reused by ``--topic-text`` without producing a
    misleading "--context" reference.
    """

    if not value:
        return ""
    if value.startswith("@@"):
        return value[1:]
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            raise typer.BadParameter(f"{flag} points to a file that does not exist: {path}")
        return path.read_text(encoding="utf-8")
    return value


def _print_draft(workflow_output) -> None:  # type: ignore[no-untyped-def]
    """Print a generated draft + citation table to stdout.

    Defined as a free function so tests can call it independently of
    the Typer CliRunner if needed. The format is intentionally
    plaintext (no rich-style formatting) so output is friendly to
    pipes and CI logs.
    """

    typer.echo("Generated draft")
    typer.echo("===============")
    typer.echo(workflow_output.text)
    typer.echo("")
    typer.echo("Citations")
    typer.echo("---------")
    if not workflow_output.citations:
        typer.echo("(none — no evidence retrieved)")
        return
    for c in workflow_output.citations:
        page = f"p. {c.page}" if c.page is not None else "p. ?"
        section = c.section_heading or "(no section heading)"
        typer.echo(
            f"  [{c.citation_id}] {c.source_status.value.upper()} — "
            f"{c.programme.value} call {c.call_id}, {page}, §{section} "
            f"(chunk_id={c.chunk_id})"
        )


def _print_audit_findings(result: AuditResult) -> None:
    """Print audit findings to stderr — errors first, then warnings.

    Using a single named function keeps the formatting consistent
    between the ``section`` post-generation audit and the standalone
    ``audit`` subcommand.
    """

    if result.passed and not result.warnings:
        typer.echo("Audit: passed (no findings).", err=True)
        return

    typer.echo("Audit findings:", err=True)
    for finding in result.errors:
        cid_part = f"[{finding.citation_id}] " if finding.citation_id is not None else ""
        typer.echo(
            f"  ERROR ({finding.code}): {cid_part}{finding.message}",
            err=True,
        )
    for finding in result.warnings:
        cid_part = f"[{finding.citation_id}] " if finding.citation_id is not None else ""
        typer.echo(
            f"  warning ({finding.code}): {cid_part}{finding.message}",
            err=True,
        )

    if result.passed:
        typer.echo(
            f"Audit: passed with {len(result.warnings)} warning(s).",
            err=True,
        )
    else:
        typer.echo(
            f"Audit: FAILED — {len(result.errors)} error(s), {len(result.warnings)} warning(s).",
            err=True,
        )


def _resolve_output_paths(
    output: Path,
    render_mode: str,
) -> dict[str, Path]:
    """Map ``--render`` value → output paths derived from ``--output``.

    Suffix policy is opinionated and consistent across modes: the
    rendered Markdown is always written to ``<base>.md`` and the JSON
    dump to ``<base>.json``. The ``<base>`` is the user's ``--output``
    path with any existing ``.md`` or ``.json`` suffix stripped (any
    other suffix is preserved as part of the base).

    Examples (matrix below holds for any ``base`` with no recognised
    suffix; suffix-bearing inputs are normalised to the same shape):

    * ``--render markdown --output draft``      → ``draft.md``
    * ``--render markdown --output draft.json`` → ``draft.md`` (the
      ``.json`` suffix is stripped because it would mislead the
      reader; the user asked for Markdown)
    * ``--render json --output draft.md``       → ``draft.json``
    * ``--render both --output draft``          → both siblings
    """

    suffix = output.suffix.lower()
    base = output.with_suffix("") if suffix in {".md", ".json"} else output
    if render_mode == "markdown":
        return {"markdown": base.with_suffix(".md")}
    if render_mode == "json":
        return {"json": base.with_suffix(".json")}
    # both
    return {
        "markdown": base.with_suffix(".md"),
        "json": base.with_suffix(".json"),
    }


def _atomic_write(path: Path, content: str) -> None:
    """Atomic write: temp + ``Path.replace``. Mirrors the ingestion CLI."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


@generate_app.command("section")
def section(
    section_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help=(f"Section to draft. One of: {', '.join(s.value for s in SectionType)}."),
    ),
    intent: str = typer.Option(
        ...,
        "--intent",
        "-i",
        help="What this section should communicate (one or two sentences).",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-x",
        help=(
            "Optional call/topic context. Pass literal text, or "
            "``@path/to/file`` to read from a file."
        ),
    ),
    topic_text: str = typer.Option(
        "",
        "--topic-text",
        help=(
            "Structured topic context as plaintext. Pass literal text, or "
            "``@path/to/file`` to read from a file. Mutually exclusive with "
            "--topic-pdf. Coexists with --context."
        ),
    ),
    topic_pdf: Path | None = typer.Option(
        None,
        "--topic-pdf",
        help=(
            "Path to a PDF excerpt of the Work Programme topic page. "
            "Parsed via Docling in offline mode. Mutually exclusive with "
            "--topic-text."
        ),
    ),
    programme: str | None = typer.Option(
        None,
        "--programme",
        "-p",
        help=(f"Filter retrieval by programme. One of: {', '.join(p.value for p in Programme)}."),
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        min=1,
        max=20,
        help="Number of past-proposal examples to retrieve.",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold",
        min=0.0,
        max=1.0,
        help=(
            "Minimum cosine similarity for retrieved evidence. The "
            "deterministic-hash embedder produces modest scores; lower "
            "to 0.0 if no evidence comes back."
        ),
    ),
    lessons_learned: bool = typer.Option(
        False,
        "--lessons-learned",
        help=(
            "Surface rejected examples more aggressively as cautionary "
            "evidence. Relaxes the rejected-chunk threshold and skips "
            "the rejected-fraction cap."
        ),
    ),
    no_esr: bool = typer.Option(
        False,
        "--no-esr",
        help="Exclude ESR (External Subject Reviewer) notes from retrieval.",
    ),
    render: str = typer.Option(
        "both",
        "--render",
        "-r",
        help=(
            "What to emit. ``markdown`` prints a rendered Markdown "
            "document with status badges; ``json`` prints the raw "
            "GenerationDraft summary; ``both`` (default) prints "
            "Markdown to stdout and writes both .md + .json siblings "
            "when --output is set."
        ),
    ),
    no_audit: bool = typer.Option(
        False,
        "--no-audit",
        help=(
            "Skip the post-generation citation audit. By default, the "
            "audit runs on every draft and fails the command if any "
            "citation is missing a status tag, references an "
            "unknown marker, or the rendered output drops a badge."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Path to write the rendered artefacts. Atomic; will not "
            "clobber an existing file unless --overwrite is set. The "
            "exact files written depend on --render."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite existing --output files.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (defaults to the repo-root config.yaml).",
    ),
    collection: str = typer.Option(
        "default",
        "--collection",
        help="Chroma collection name to retrieve from.",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        min=1,
        max=MAX_ITERATIONS_CEILING,
        help=(
            "Number of total drafting passes (1-5). Default 1 (single-pass — "
            "backward compatible). Set >1 to enable the Task 3.2 critic loop: "
            "each additional iteration runs a critic over the prior draft "
            "and regenerates with the critique woven into the intent. The "
            "issue body recommends 3 as a working default for interactive use. "
            "AC #2 (stop after any iteration) — interrupt with Ctrl+C between "
            "iterations; the most recent completed draft is what was written to "
            "stdout / disk."
        ),
    ),
    profile_programme: str | None = typer.Option(
        None,
        "--profile-programme",
        help=(
            "Programme whose drafting profile to apply (e.g., horizon_europe). "
            "When set, the same profile drives both the initial draft and "
            "every critic iteration so the requirements list stays consistent."
        ),
    ),
) -> None:
    """Generate a draft of one section using indexed past-proposal evidence."""

    # Resolve CLI string flags into the typed enums upfront so an
    # invalid value surfaces a friendly error before any retrieval /
    # generation cost is incurred.
    try:
        section_enum = SectionType(section_type)
    except ValueError as exc:
        typer.echo(
            f"error: --type must be one of {[s.value for s in SectionType]}, got {section_type!r}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        programme_enum = Programme(programme) if programme else None
    except ValueError as exc:
        typer.echo(
            f"error: --programme must be one of {[p.value for p in Programme]}, got {programme!r}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    render_mode = render.lower()
    if render_mode not in {"markdown", "json", "both"}:
        typer.echo(
            f"error: --render must be one of ['markdown', 'json', 'both'], got {render!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(cfg)

    embedder = make_embedder(cfg)
    index = ChromaIndex(
        index_path=cfg.index_path,
        embedder=embedder,
        collection_name=collection,
    )
    policy = RetrievalPolicy(
        relevance_threshold=threshold,
        lessons_learned_mode=lessons_learned,
        include_esr=not no_esr,
    )
    retriever = SourceStatusAwareRetriever(index, policy=policy)

    # Analytics is wired here so the workflow emits draft start /
    # complete events for every CLI-driven run; the logger lives under
    # ``runtime_dir`` and never leaves it unless the user explicitly
    # runs ``eurpe analytics export``.
    analytics = make_analytics_logger(cfg)

    try:
        llm = make_llm_client(cfg)
    except LLMUnavailableError as exc:
        typer.echo(f"error: LLM unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except GenerationError as exc:
        typer.echo(f"error: generation setup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except SecurityError as exc:
        typer.echo(f"error: network policy denied generation setup: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    workflow = SectionGenerationWorkflow(retriever=retriever, llm=llm, analytics=analytics)
    # The critic loop reuses the same LLM client for critique by default.
    # See ``GenerationService`` for the rationale; the CLI mirrors that
    # convention so a single ``ollama pull`` is enough to run the loop.
    critic_loop = CriticLoopWorkflow(workflow=workflow, critic=CriticAgent(llm))

    profile: DraftingProfile | None = None
    if profile_programme:
        try:
            profile = load_profile(Programme(profile_programme))
        except ValueError as exc:
            typer.echo(
                f"error: --profile-programme must be one of "
                f"{[p.value for p in Programme]}, got {profile_programme!r}",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        except FileNotFoundError as exc:
            typer.echo(
                f"error: no drafting profile bundled for {profile_programme!r}: {exc}",
                err=True,
            )
            raise typer.Exit(code=1) from exc

    # Resolve optional file-backed --context value. Failures here
    # surface as a clean BadParameter from the helper (Typer turns it
    # into a non-zero exit with a friendly message).
    resolved_context = _load_context(context)

    # ``--topic-text`` and ``--topic-pdf`` cannot coexist: each is a
    # different *input mode* for the same TopicContext slot. We refuse
    # to silently prefer one — operators must pick.
    if topic_text and topic_pdf is not None:
        typer.echo(
            "error: --topic-text and --topic-pdf are mutually exclusive; "
            "pass one of them, not both.",
            err=True,
        )
        raise typer.Exit(code=1)

    topic_context: TopicContext | None = None
    if topic_text:
        resolved_topic_text = _load_context(topic_text, flag="--topic-text")
        topic_context = extract_topic_context_from_text(resolved_topic_text)
    elif topic_pdf is not None:
        try:
            topic_context = extract_topic_context_from_pdf(topic_pdf, config=cfg)
        except IngestionError as exc:
            typer.echo(
                f"error: failed to parse --topic-pdf {topic_pdf}: {exc}",
                err=True,
            )
            raise typer.Exit(code=1) from exc

    request = GenerationRequest(
        section_type=section_enum,
        user_intent=intent,
        call_context=resolved_context,
        target_programme=programme_enum,
        top_k_examples=top_k,
        lessons_learned=lessons_learned,
        topic_context=topic_context,
    )

    typer.echo(
        f"Generating {section_enum.value} draft using LLM={llm.model} "
        f"and embedder={embedder.model_name}..."
    )

    try:
        draft = workflow.run(request, profile=profile)
    except LLMUnavailableError as exc:
        typer.echo(f"error: LLM unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except GenerationError as exc:
        typer.echo(f"error: generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except SecurityError as exc:
        typer.echo(f"error: network policy denied generation: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Critic loop (Task 3.2 / issue #16). When --iterations > 1, run
    # the loop synchronously up to the cap. Ctrl+C between iterations
    # leaves ``draft`` pinned to the most recent completed pass so the
    # downstream audit / write paths still have something to act on
    # (AC #2: stop after any iteration).
    if iterations > 1:
        for next_pass in range(2, iterations + 1):
            typer.echo(f"  iteration {next_pass}/{iterations}: running critic loop...")
            try:
                result = critic_loop.iterate(
                    prior_draft=draft,
                    request=request,
                    max_iterations=iterations,
                    profile=profile,
                )
            except KeyboardInterrupt:
                typer.echo(
                    f"  ✗ interrupted before iteration {next_pass}; "
                    f"keeping the draft from iteration {next_pass - 1}.",
                    err=True,
                )
                break
            except LLMUnavailableError as exc:
                typer.echo(f"error: LLM unavailable during iteration: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            except GenerationError as exc:
                typer.echo(f"error: iteration failed: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            except SecurityError as exc:
                typer.echo(f"error: network policy denied iteration: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            draft = result.draft
            if result.stopped:
                typer.echo(
                    f"  ✓ iteration cap reached (iteration {result.iteration_index} "
                    f"of {result.max_iterations})."
                )
                break

    renderer = MarkdownCitationRenderer()
    rendered_md = renderer.render(draft)

    typer.echo("")
    if render_mode in {"markdown", "both"}:
        # Markdown form for the human reader. The plaintext draft
        # summary is still emitted under the 'json' branch so test
        # tooling that asserts on "Generated draft" / "Citations"
        # markers continues to work.
        typer.echo(rendered_md)
    if render_mode == "json":
        _print_draft(draft)

    # Audit BEFORE writing output: a draft that fails the audit should
    # not be persisted to disk (otherwise the user might pick it up
    # later thinking it passed).
    if not no_audit:
        audit_result = CitationAudit().audit_rendered(draft, rendered_md)
        _print_audit_findings(audit_result)
        if not audit_result.passed:
            raise typer.Exit(code=1)

    if output is not None:
        target_paths = _resolve_output_paths(output, render_mode)
        # Pre-flight: refuse to clobber any of the targets unless
        # --overwrite is set. Doing this before writing avoids the
        # half-written state where ``.md`` was written but ``.json``
        # already existed.
        for path in target_paths.values():
            if path.exists() and not overwrite:
                typer.echo(
                    f"error: output file already exists: {path} "
                    "(pass --overwrite/-f to replace it)",
                    err=True,
                )
                raise typer.Exit(code=1)

        for kind, path in target_paths.items():
            content = rendered_md if kind == "markdown" else draft.model_dump_json(indent=2)
            _atomic_write(path, content)
            typer.echo("")
            typer.echo(f"  wrote {kind:8s}: {path}")
            # Record an ExportEvent for each artefact actually written.
            # Wrapped in try/except so an analytics failure cannot block
            # the user from getting their draft on disk.
            try:
                analytics.log(
                    ExportEvent(
                        event_type=EventType.EXPORT,
                        kind=kind,
                        byte_count=len(content.encode("utf-8")),
                        section_type=section_enum.value,
                    )
                )
            # Analytics failures must not block the user's export artefacts.
            except Exception:  # pragma: no cover  # nosec B110
                pass

    sys.stdout.flush()
    raise typer.Exit(code=0)


@generate_app.command("audit")
def audit(
    draft_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "Path to a previously dumped GenerationDraft JSON file "
            "(e.g., one produced by ``eurpe generate section --output``)."
        ),
    ),
) -> None:
    """Re-render a saved draft and run the citation audit on it.

    Exits 0 if the audit passes; exits 1 with findings on stderr if
    any error finding is recorded. CI-friendly entry point for
    re-checking saved drafts without re-running the LLM.
    """

    try:
        payload = json.loads(draft_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(
            f"error: {draft_path} is not valid JSON: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    try:
        draft = GenerationDraft.model_validate(payload)
    except Exception as exc:  # pragma: no cover - pydantic surfaces a clean message
        typer.echo(
            f"error: {draft_path} does not match the GenerationDraft schema: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    rendered_md = MarkdownCitationRenderer().render(draft)
    result = CitationAudit().audit_rendered(draft, rendered_md)

    typer.echo(
        f"Audit summary for {draft_path}: "
        f"{len(result.errors)} error(s), {len(result.warnings)} warning(s)."
    )
    _print_audit_findings(result)

    raise typer.Exit(code=0 if result.passed else 1)


def _print_release_audit_summary(report: ReleaseAuditReport) -> None:
    """Print a concise release-audit summary to stderr.

    The full row-by-row report is written to disk via ``--output``;
    stderr only carries the top-level verdict + counters so a CI log
    is easy to skim. Mirrors the per-draft ``_print_audit_findings``
    convention (status / counts on stderr; payload on stdout or disk).
    """

    verdict = "passed" if report.passed else "FAILED"
    typer.echo("", err=True)
    typer.echo(f"Release audit {verdict}:", err=True)
    typer.echo(f"  audit_directory     : {report.audit_directory}", err=True)
    typer.echo(f"  total drafts        : {report.total_drafts}", err=True)
    typer.echo(f"  audited drafts      : {report.audited_drafts}", err=True)
    typer.echo(f"  passed drafts       : {report.passed_drafts}", err=True)
    typer.echo(f"  failed drafts       : {report.failed_drafts}", err=True)
    typer.echo(f"  citations audited   : {report.citation_count}", err=True)
    typer.echo(f"  unlabeled citations : {report.unlabeled_citation_count}", err=True)
    if report.sample_size is not None:
        typer.echo(f"  sample size         : {report.sample_size}", err=True)
        typer.echo(f"  sample seed         : {report.sample_seed}", err=True)

    # Per-draft failure list — only printed when at least one failed.
    # An all-pass report keeps the stderr lean.
    if report.failed_drafts:
        typer.echo("", err=True)
        typer.echo("Failed drafts:", err=True)
        for r in report.draft_results:
            if r.passed:
                continue
            codes = sorted({f.code for f in r.audit_result.errors})
            typer.echo(
                f"  - {r.draft_path}  [{', '.join(codes) or 'no-code'}]",
                err=True,
            )


@generate_app.command("audit-release")
def audit_release(
    directory: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help=(
            "Directory containing one or more GenerationDraft JSON files "
            "(typically the ``--output`` target of ``eurpe generate "
            "section``). Subdirectories are walked recursively."
        ),
    ),
    sample_size: int | None = typer.Option(
        None,
        "--sample-size",
        "-n",
        min=1,
        help=(
            "Audit a deterministic random sample of N drafts. When unset "
            "(default), every discovered draft is audited. Combine with "
            "``--seed`` for byte-equal subsets across re-runs."
        ),
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help=(
            "Random seed for the sampling subset when ``--sample-size`` is "
            "set. Defaults to 42 so a release manager who omits this flag "
            "still gets reproducible output across machines. Recorded in "
            "the report for traceability."
        ),
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help=(
            "Write the full ReleaseAuditReport as JSON to this path. "
            "Atomic. Refuses to overwrite an existing file unless "
            "``--overwrite`` is set."
        ),
    ),
    output_markdown: Path | None = typer.Option(
        None,
        "--output-markdown",
        help=(
            "Write the Markdown release-audit summary to this path. "
            "Atomic. Refuses to overwrite an existing file unless "
            "``--overwrite`` is set. Operators paste this into release "
            "notes."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite existing --output-json / --output-markdown files.",
    ),
    print_summary: bool = typer.Option(
        True,
        "--summary/--no-summary",
        help=(
            "Print the Markdown summary to stdout in addition to writing "
            "any --output-* files. Default on; set --no-summary for a "
            "quiet CI run."
        ),
    ),
) -> None:
    """Audit every saved draft under DIRECTORY for citation fidelity.

    The harness loads each ``*.json`` GenerationDraft, runs the same
    :class:`~eurpe.generation.CitationAudit` checks the per-draft
    ``audit`` subcommand applies, and aggregates the verdict.

    Exits 0 when every audited draft passes. Exits 1 when any draft
    fails OR the harness cannot load a draft file. The release-blocking
    contract from PRD § "Source labeling accuracy" is enforced
    end-to-end: every citation MUST visibly carry a status tag.

    .. note::

        Write ``--output-json`` / ``--output-markdown`` to a path
        *outside* the audited directory. The harness walks ``DIRECTORY``
        recursively for ``*.json`` files; an output file dropped inside
        would be picked up as a draft on the next run and fail the
        schema validation. Release workflows typically put outputs
        under ``release-notes/audits/<release-tag>/`` instead.

    Pairs with the manual audit template at
    ``docs/release-audit-template.md`` for the human-judgement
    portion of the release gate (e.g., "is this quoted passage
    accurate against the cited PDF?"). Issue #18 AC3.
    """

    harness = ReleaseAuditHarness()
    try:
        report = harness.audit_directory(
            directory,
            sample_size=sample_size,
            seed=seed,
        )
    except ReleaseAuditHarnessError as exc:
        typer.echo(f"error: release audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Pre-flight the output paths BEFORE doing any work, so the user
    # doesn't get a "file already exists" error after the audit has
    # already run. Mirrors the section command's overwrite contract.
    if not overwrite:
        for label, path in (
            ("--output-json", output_json),
            ("--output-markdown", output_markdown),
        ):
            if path is not None and path.exists():
                typer.echo(
                    f"error: output file already exists: {path} "
                    f"(pass --overwrite/-f to replace it or pick a different "
                    f"{label} path)",
                    err=True,
                )
                raise typer.Exit(code=1)

    if output_json is not None:
        _atomic_write(output_json, report.model_dump_json(indent=2) + "\n")
        typer.echo(f"  wrote JSON     : {output_json}", err=True)
    if output_markdown is not None:
        _atomic_write(output_markdown, report.render_markdown_summary())
        typer.echo(f"  wrote Markdown : {output_markdown}", err=True)

    if print_summary:
        typer.echo(report.render_markdown_summary())

    _print_release_audit_summary(report)

    raise typer.Exit(code=0 if report.passed else 1)
