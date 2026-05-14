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

from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.generation.audit import AuditResult, CitationAudit
from eurpe.generation.errors import GenerationError, LLMUnavailableError
from eurpe.generation.llm import make_llm_client
from eurpe.generation.models import GenerationDraft, GenerationRequest
from eurpe.generation.render import MarkdownCitationRenderer
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.retrieval import (
    ChromaIndex,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
    make_embedder,
)
from eurpe.schema import Programme, SectionType

# A sub-Typer so the CLI surface is ``eurpe generate section``. Wired
# into the top-level app in :mod:`eurpe.cli`.
generate_app = typer.Typer(
    name="generate",
    help="Generate a draft proposal section from indexed evidence.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_context(value: str) -> str:
    """Resolve ``--context`` value: literal text or ``@path/to/file`` reference.

    The ``@``-prefixed file form mirrors a common CLI convention
    (curl, gh) and avoids the awkwardness of pasting multi-paragraph
    call text on a command line. A literal ``@`` at start can be
    escaped by doubling (``@@`` → ``@``) for the rare user who
    wants verbatim ``@``-prefixed text.
    """

    if not value:
        return ""
    if value.startswith("@@"):
        return value[1:]
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            raise typer.BadParameter(
                f"--context points to a file that does not exist: {path}"
            )
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
        cid_part = (
            f"[{finding.citation_id}] " if finding.citation_id is not None else ""
        )
        typer.echo(
            f"  ERROR ({finding.code}): {cid_part}{finding.message}",
            err=True,
        )
    for finding in result.warnings:
        cid_part = (
            f"[{finding.citation_id}] " if finding.citation_id is not None else ""
        )
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
            f"Audit: FAILED — {len(result.errors)} error(s), "
            f"{len(result.warnings)} warning(s).",
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
        help=(
            f"Section to draft. One of: "
            f"{', '.join(s.value for s in SectionType)}."
        ),
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
    programme: str | None = typer.Option(
        None,
        "--programme",
        "-p",
        help=(
            f"Filter retrieval by programme. One of: "
            f"{', '.join(p.value for p in Programme)}."
        ),
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
) -> None:
    """Generate a draft of one section using indexed past-proposal evidence."""

    # Resolve CLI string flags into the typed enums upfront so an
    # invalid value surfaces a friendly error before any retrieval /
    # generation cost is incurred.
    try:
        section_enum = SectionType(section_type)
    except ValueError as exc:
        typer.echo(
            f"error: --type must be one of {[s.value for s in SectionType]}, "
            f"got {section_type!r}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        programme_enum = Programme(programme) if programme else None
    except ValueError as exc:
        typer.echo(
            f"error: --programme must be one of {[p.value for p in Programme]}, "
            f"got {programme!r}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    render_mode = render.lower()
    if render_mode not in {"markdown", "json", "both"}:
        typer.echo(
            f"error: --render must be one of ['markdown', 'json', 'both'], "
            f"got {render!r}",
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

    llm = make_llm_client(cfg)
    workflow = SectionGenerationWorkflow(retriever=retriever, llm=llm)

    # Resolve optional file-backed --context value. Failures here
    # surface as a clean BadParameter from the helper (Typer turns it
    # into a non-zero exit with a friendly message).
    resolved_context = _load_context(context)

    request = GenerationRequest(
        section_type=section_enum,
        user_intent=intent,
        call_context=resolved_context,
        target_programme=programme_enum,
        top_k_examples=top_k,
        lessons_learned=lessons_learned,
    )

    typer.echo(
        f"Generating {section_enum.value} draft using LLM={llm.model} "
        f"and embedder={embedder.model_name}..."
    )

    try:
        draft = workflow.run(request)
    except LLMUnavailableError as exc:
        typer.echo(f"error: LLM unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except GenerationError as exc:
        typer.echo(f"error: generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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
            content = (
                rendered_md
                if kind == "markdown"
                else draft.model_dump_json(indent=2)
            )
            _atomic_write(path, content)
            typer.echo("")
            typer.echo(f"  wrote {kind:8s}: {path}")

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
