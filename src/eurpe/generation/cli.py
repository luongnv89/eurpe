"""Typer commands for ``eurpe generate ...``.

A single subcommand today (``eurpe generate section``) drives the
:class:`~eurpe.generation.SectionGenerationWorkflow` end-to-end:

1. Load config, ensure runtime dirs exist, build the embedder /
   index / retriever (same pattern used by ``eurpe index query``).
2. Build the LLM client via :func:`~eurpe.generation.make_llm_client`
   so the offline-fallback path is honoured.
3. Build the workflow, run it, print the draft + a citations table
   to stdout.
4. If ``--output`` is given, atomically write a JSON dump of the
   :class:`~eurpe.generation.GenerationDraft` (same atomic-write
   pattern as ``eurpe ingest``).

The command lives in its own module so the ``eurpe.cli`` top-level
file stays thin (matching the convention used for ``ingestion`` and
``retrieval``). It is mounted onto the top-level Typer in
``eurpe.cli`` as a sub-Typer at ``generate``.
"""

from __future__ import annotations

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
from eurpe.generation.errors import GenerationError, LLMUnavailableError
from eurpe.generation.llm import make_llm_client
from eurpe.generation.models import GenerationRequest
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
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Path to write a JSON dump of the GenerationDraft. Atomic; "
            "will not clobber an existing file unless --overwrite is set."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite an existing --output file.",
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

    typer.echo("")
    _print_draft(draft)

    if output is not None:
        if output.exists() and not overwrite:
            typer.echo(
                f"error: output file already exists: {output} "
                "(pass --overwrite/-f to replace it)",
                err=True,
            )
            raise typer.Exit(code=1)
        # Atomic write: temp + Path.replace. Mirrors the pattern in
        # ``eurpe ingest``. ``parent.mkdir(...)`` so the user can pass
        # an output path in a directory that doesn't exist yet.
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output.with_suffix(output.suffix + ".tmp")
        tmp_path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(output)
        typer.echo("")
        typer.echo(f"  wrote         : {output}")

    sys.stdout.flush()
    raise typer.Exit(code=0)
