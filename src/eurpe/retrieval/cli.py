"""Typer commands for ``eurpe index ...``.

Two subcommands wired into the top-level CLI as a sub-Typer at
``index``:

* ``eurpe index build <metadata-yaml>...`` — for each YAML sidecar,
  parses the referenced PDF (Docling), runs the chunker, embeds via
  the configured embedder, and upserts into the local Chroma index.
* ``eurpe index query <text>`` — embeds the query, retrieves the top
  results, and prints rank/score/source-status/programme/page/snippet
  for each.

Both commands honour ``--config/-c`` exactly the way ``eurpe smoke``
and ``eurpe ingest`` do, so a single ``config.yaml`` rules them all.
The embedder is selected by :func:`eurpe.retrieval.make_embedder` from
the loaded config; in offline mode without a reachable Ollama daemon
the deterministic-hash fallback is chosen and a warning is logged.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import typer
import yaml

from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.ingestion.docling_parser import DoclingProposalParser
from eurpe.ingestion.errors import IngestionError
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.retrieval.embeddings import make_embedder
from eurpe.retrieval.errors import IndexingError
from eurpe.retrieval.index import ChromaIndex
from eurpe.retrieval.models import Chunk
from eurpe.retrieval.retriever import (
    RetrievalPolicy,
    RetrievalResult,
    SourceStatusAwareRetriever,
)
from eurpe.schema import Programme, ProposalMetadata, SourceStatus

# A sub-Typer so the CLI surface is ``eurpe index build`` /
# ``eurpe index query``. Wired into the top-level app in
# :mod:`eurpe.cli`.
index_app = typer.Typer(
    name="index",
    help="Build and query the local proposal vector index.",
    no_args_is_help=True,
    add_completion=False,
)


def _resolve_pdf_path(metadata_yaml: Path, source_path: str) -> Path:
    """Resolve ``source_path`` relative to the metadata YAML's directory.

    Sidecars are typically checked-in next to the PDFs they describe,
    so a relative ``source_path`` should be interpreted relative to
    the YAML, not the current working directory. Absolute paths are
    returned untouched.
    """

    candidate = Path(source_path)
    if candidate.is_absolute():
        return candidate
    return (metadata_yaml.parent / candidate).resolve()


def _load_proposal_metadata(metadata_yaml: Path) -> ProposalMetadata:
    raw = yaml.safe_load(metadata_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{metadata_yaml}: expected a YAML mapping at the document root")
    # The metadata YAMLs we accept here are *proposal-shaped*, not
    # *chunk-shaped* — chunk YAMLs nest a ``proposal`` key. Accept
    # both: if ``proposal`` is present, use that subkey; otherwise
    # treat the whole mapping as the proposal record.
    proposal_raw = raw.get("proposal", raw) if isinstance(raw.get("proposal"), dict) else raw
    return ProposalMetadata.model_validate(proposal_raw)


def _format_snippet(text: str, *, max_chars: int = 200) -> str:
    """Single-line snippet for human-readable query output."""

    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


def _print_query_results(results: Iterable[tuple[Chunk, float]], *, max_chars: int = 200) -> None:
    """Legacy formatter for raw ``(chunk, score)`` tuples.

    Kept around for any callers that still consume ``ChromaIndex.query``
    output directly (none today inside the CLI). The retriever-driven
    path uses :func:`_print_retrieval_results` instead, which surfaces
    the policy_reason column.
    """

    rows = list(enumerate(results, start=1))
    if not rows:
        typer.echo("(no results)")
        return
    for rank, (chunk, score) in rows:
        meta = chunk.metadata
        page = meta.anchor.page if meta.anchor.page is not None else "?"
        typer.echo(
            f"#{rank} [{score:+.4f}] "
            f"status={meta.source_status.value} "
            f"programme={meta.proposal.programme.value} "
            f"call={meta.proposal.call_id} "
            f"page={page}"
        )
        typer.echo(f"     {_format_snippet(chunk.text, max_chars=max_chars)}")


def _print_retrieval_results(results: Iterable[RetrievalResult], *, max_chars: int = 200) -> None:
    """Print results returned by :class:`SourceStatusAwareRetriever`.

    Adds a ``policy_reason=`` column so an operator can immediately tell
    why each chunk was kept (funded_primary, rejected_threshold_met,
    lessons_learned_mode, esr_advisory, unknown_low_confidence).
    """

    rows = list(results)
    if not rows:
        typer.echo("(no results)")
        return
    for r in rows:
        meta = r.chunk.metadata
        page = meta.anchor.page if meta.anchor.page is not None else "?"
        typer.echo(
            f"#{r.rank} [{r.score:+.4f}] "
            f"status={meta.source_status.value} "
            f"policy_reason={r.policy_reason} "
            f"programme={meta.proposal.programme.value} "
            f"call={meta.proposal.call_id} "
            f"page={page}"
        )
        typer.echo(f"     {_format_snippet(r.chunk.text, max_chars=max_chars)}")


@index_app.command("build")
def build(
    metadata_yamls: list[Path] = typer.Argument(
        ...,
        help="One or more YAML metadata sidecars to ingest.",
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
        help="Chroma collection name (3-512 chars, [a-zA-Z0-9._-]).",
    ),
) -> None:
    """Build (or update) the local index from one or more YAML sidecars.

    For each YAML the referenced PDF is parsed via Docling, chunked,
    embedded, and upserted into the configured Chroma collection.
    The ``upsert`` operation is idempotent on
    :attr:`Chunk.chunk_id`, so re-running this command on the same
    inputs leaves the collection in the same state.
    """

    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(cfg)

    embedder = make_embedder(cfg)
    typer.echo(f"Using embedder: {embedder.model_name} (dim={embedder.dimension})")

    parser = DoclingProposalParser(offline=cfg.offline_mode)
    chunker = HierarchicalChunker()
    index = ChromaIndex(
        index_path=cfg.index_path,
        embedder=embedder,
        collection_name=collection,
    )

    total_added = 0
    for yaml_path in metadata_yamls:
        if not yaml_path.exists():
            typer.echo(f"error: metadata file not found: {yaml_path}", err=True)
            raise typer.Exit(code=1)

        try:
            proposal_meta = _load_proposal_metadata(yaml_path)
        except Exception as exc:  # noqa: BLE001 - surface any sidecar issue
            typer.echo(f"error: invalid metadata in {yaml_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        pdf_path = _resolve_pdf_path(yaml_path, proposal_meta.source_path)
        try:
            parsed = parser.parse(pdf_path)
        except IngestionError as exc:
            typer.echo(f"error: parsing failed for {pdf_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        chunks = chunker.chunk(parsed, proposal_meta)
        try:
            index.upsert(chunks)
        except IndexingError as exc:
            typer.echo(f"error: index upsert failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        total_added += len(chunks)
        typer.echo(f"  ingested {yaml_path.name}: {len(chunks)} chunks from {pdf_path.name}")

    typer.echo("")
    typer.echo(
        f"Done. {total_added} chunks added; collection {collection!r} now holds {index.count()}."
    )


@index_app.command("query")
def query(
    text: str = typer.Argument(..., help="Free-text query."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Number of results."),
    source_status: str | None = typer.Option(
        None,
        "--source-status",
        help=(
            "Hard-filter by source status (funded | rejected | esr_note | unknown). "
            "Forwarded as a Chroma where clause; the source-status policy still "
            "applies on top of the filtered candidate pool."
        ),
    ),
    programme: str | None = typer.Option(
        None,
        "--programme",
        help="Filter by programme (e.g., horizon_europe).",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold",
        min=0.0,
        max=1.0,
        help=(
            "Minimum cosine similarity for a chunk to be considered topically "
            "relevant. Applied uniformly to all source statuses."
        ),
    ),
    lessons_learned: bool = typer.Option(
        False,
        "--lessons-learned",
        help=(
            "Enable lessons-learned mode: relax the rejected threshold by "
            "--rejected-offset and skip the rejected-fraction cap so cautionary "
            "examples are surfaced more aggressively."
        ),
    ),
    rejected_offset: float = typer.Option(
        -0.10,
        "--rejected-offset",
        min=-0.5,
        max=0.5,
        help=(
            "Offset added to --threshold for REJECTED chunks under "
            "--lessons-learned. Typically negative (default -0.10) to relax the "
            "bar. Ignored without --lessons-learned."
        ),
    ),
    no_esr: bool = typer.Option(
        False,
        "--no-esr",
        help=(
            "Exclude ESR (External Subject Reviewer) notes entirely. Use when "
            "drafting the final version where subjective commentary should not "
            "leak into the retrieved evidence."
        ),
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml.",
    ),
    collection: str = typer.Option(
        "default",
        "--collection",
        help="Chroma collection name.",
    ),
    snippet_chars: int = typer.Option(
        200,
        "--snippet-chars",
        min=20,
        help="Max characters of each result's text snippet.",
    ),
) -> None:
    """Query the local index applying the source-status policy.

    Drives :class:`~eurpe.retrieval.retriever.SourceStatusAwareRetriever`
    so the same policy used by the proposal-drafting pipeline is what an
    operator sees when probing the index from the command line.

    The optional ``--source-status`` and ``--programme`` flags become
    hard server-side filters on the Chroma collection; the policy
    (threshold, funded-first ordering, rejected cap, ESR handling) then
    runs on top of the filtered pool.
    """

    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(cfg)

    embedder = make_embedder(cfg)
    index = ChromaIndex(
        index_path=cfg.index_path,
        embedder=embedder,
        collection_name=collection,
    )

    # Coerce CLI string flags into the typed enums the retriever expects.
    # Invalid values surface a friendly error here rather than as an
    # opaque KeyError deep in the retriever.
    try:
        status_enum = SourceStatus(source_status) if source_status else None
    except ValueError as exc:
        typer.echo(
            f"error: --source-status must be one of "
            f"{[s.value for s in SourceStatus]}, got {source_status!r}",
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

    policy = RetrievalPolicy(
        relevance_threshold=threshold,
        lessons_learned_mode=lessons_learned,
        rejected_threshold_offset=rejected_offset,
        include_esr=not no_esr,
    )
    retriever = SourceStatusAwareRetriever(index, policy=policy)

    try:
        results = retriever.retrieve(
            text,
            top_k=top_k,
            programme=programme_enum,
            source_status=status_enum,
        )
    except IndexingError as exc:
        typer.echo(f"error: query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_retrieval_results(results, max_chars=snippet_chars)
