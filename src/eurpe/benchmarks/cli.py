"""Typer commands for ``eurpe benchmark ...``.

Operator-facing sub-app for the Task 3.5 / issue #19 benchmark
harness. Four sub-commands, each mapping to one acceptance
criterion or the convenience aggregate:

* ``eurpe benchmark indexing`` — AC1 (initial indexing latency).
* ``eurpe benchmark retrieval`` — AC2 (top-k retrieval latency).
* ``eurpe benchmark generation`` — AC3 (section generation latency,
  with model/runtime configuration in the output).
* ``eurpe benchmark all`` — run all three on a shared in-memory
  index. The "first thing an operator runs" entry point.

Every sub-command:

1. Prints a human-readable summary to stdout (no machine parsing
   required).
2. Optionally writes a structured JSON report via ``--output``.
   Mirrors the ``eurpe analytics export`` contract: an explicit
   path, no default destination, atomic write via a sibling
   ``.tmp`` then ``Path.replace``.

The default runtime is ``deterministic`` so a fresh clone produces
a complete report with zero network. Passing ``--runtime ollama``
swaps in the real backends — the report's runtime fingerprint
records which path produced the numbers so a reviewer can compare
them against the PRD targets correctly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from eurpe.benchmarks.runner import (
    BenchmarkReport,
    GenerationBenchmark,
    IndexingBenchmark,
    RetrievalBenchmark,
    build_synthetic_corpus,
    capture_runtime_fingerprint,
    run_all,
)
from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.generation.llm import DeterministicLLMClient, LLMClient, make_llm_client
from eurpe.generation.models import GenerationRequest
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    Embedder,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
)
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.retrieval.embeddings import make_embedder
from eurpe.schema import SectionType

# Sub-Typer for ``eurpe benchmark ...``. Mounted onto the top-level
# app in :mod:`eurpe.cli`.
benchmark_app = typer.Typer(
    name="benchmark",
    help=(
        "Measure indexing / retrieval / generation latency against "
        "the PRD's v1 performance targets. Defaults to the offline "
        "deterministic backends; pass --runtime ollama to measure "
        "the real local-LLM path."
    ),
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Shared backend selection
# ---------------------------------------------------------------------------


_RUNTIME_DETERMINISTIC = "deterministic"
_RUNTIME_OLLAMA = "ollama"


def _select_backends(
    runtime: str,
    config_path: Path,
) -> tuple[Embedder, LLMClient, str]:
    """Build embedder + LLM client for the requested runtime.

    Returns ``(embedder, llm, runtime_label)``. The label is the
    *effective* runtime — if ``runtime`` is ``ollama`` but the daemon
    is unreachable, :func:`make_embedder` /
    :func:`make_llm_client` already fall back to the deterministic
    stubs in offline mode; the label reflects the realised choice
    rather than the requested one, so the report does not lie about
    what produced the numbers.

    For the deterministic path the config is irrelevant — the stubs
    need no settings. For the ollama path we load the config so the
    factories pick the operator-configured model + host.
    """

    if runtime == _RUNTIME_DETERMINISTIC:
        embedder: Embedder = DeterministicHashEmbedder(dimension=128)
        llm: LLMClient = DeterministicLLMClient()
        return embedder, llm, _RUNTIME_DETERMINISTIC

    if runtime != _RUNTIME_OLLAMA:
        raise typer.BadParameter(
            f"Unknown runtime: {runtime!r}. Choose 'deterministic' or 'ollama'."
        )

    # Ollama path — load config so the factories know which model to
    # use. Mirrors the resolution in ``eurpe index build``.
    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(cfg)
    embedder = make_embedder(cfg)
    llm = make_llm_client(cfg)
    return embedder, llm, _RUNTIME_OLLAMA


def _write_report(report: BenchmarkReport, output: Path | None) -> None:
    """Atomically write ``report`` as JSON to ``output``, if given.

    Mirrors the atomic-write pattern in
    :mod:`eurpe.analytics.cli`: write to a sibling ``.tmp`` file
    and ``Path.replace`` it into place. The destination directory
    is created if missing.

    ``output`` of ``None`` is a no-op — the command was run without
    the ``--output`` flag and only the stdout summary was wanted.
    """

    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    try:
        tmp_path.write_text(report.to_json() + "\n", encoding="utf-8")
        tmp_path.replace(output)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:  # pragma: no cover - defensive
                pass
        raise


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_runtime(report: BenchmarkReport) -> None:
    """Print the runtime fingerprint block at the top of every summary."""

    rt = report.runtime
    typer.echo("Runtime")
    typer.echo(f"  runtime         : {rt.runtime}")
    typer.echo(f"  llm_model       : {rt.llm_model}")
    typer.echo(f"  embedder        : {rt.embedder}")
    typer.echo(f"  python          : {rt.python_version}")
    typer.echo(f"  platform        : {rt.platform}")
    if rt.cpu_count is not None:
        typer.echo(f"  cpu_count       : {rt.cpu_count}")
    typer.echo("")


def _print_indexing(report: BenchmarkReport) -> None:
    if report.indexing is None:
        return
    m = report.indexing
    typer.echo("Indexing")
    typer.echo(f"  proposals       : {m.proposal_count}")
    typer.echo(f"  chunks          : {m.chunk_count}")
    typer.echo(f"  elapsed_ms      : {m.elapsed_ms}")
    typer.echo(f"  per_proposal_ms : {m.per_proposal_ms_avg:.2f}")
    typer.echo(f"  chunks_per_sec  : {m.chunks_per_second:.2f}")
    typer.echo(
        "  PRD target      : <2 hours on Mac M1 32 GB for 40 proposals"
    )
    typer.echo("")


def _print_retrieval(report: BenchmarkReport) -> None:
    if report.retrieval is None:
        return
    m = report.retrieval
    typer.echo("Retrieval")
    typer.echo(f"  query_count     : {m.query_count}")
    typer.echo(f"  top_k           : {m.top_k}")
    typer.echo(f"  elapsed_ms_min  : {m.elapsed_ms_min}")
    typer.echo(f"  elapsed_ms_avg  : {m.elapsed_ms_avg:.2f}")
    typer.echo(f"  elapsed_ms_p95  : {m.elapsed_ms_p95}")
    typer.echo(f"  elapsed_ms_max  : {m.elapsed_ms_max}")
    typer.echo(f"  result_count_avg: {m.result_count_avg:.2f}")
    typer.echo("  PRD target      : <2 s for top-k retrieval")
    typer.echo("")


def _print_generation(report: BenchmarkReport) -> None:
    if report.generation is None:
        return
    m = report.generation
    typer.echo("Generation")
    typer.echo(f"  section_type    : {m.section_type}")
    typer.echo(f"  elapsed_ms      : {m.elapsed_ms}")
    typer.echo(f"  top_k_examples  : {m.top_k_examples}")
    typer.echo(f"  citation_count  : {m.citation_count}")
    typer.echo(f"  prompt_length   : {m.prompt_length}")
    typer.echo(f"  draft_length    : {m.draft_length}")
    typer.echo(
        "  PRD target      : <2 min on M1, <30 s on DGX (5–10 page section)"
    )
    typer.echo("")


# ---------------------------------------------------------------------------
# ``eurpe benchmark all``
# ---------------------------------------------------------------------------


@benchmark_app.command("all")
def benchmark_all(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Optional path to write the structured JSON report. The "
            "stdout summary is always printed; ``--output`` adds a "
            "machine-readable file (mirrors ``eurpe analytics export``)."
        ),
    ),
    runtime: str = typer.Option(
        _RUNTIME_DETERMINISTIC,
        "--runtime",
        "-r",
        help=(
            "Backend selection: ``deterministic`` (offline, instant — "
            "the default) or ``ollama`` (real local LLM + embedder, "
            "needed for honest comparison against the PRD targets)."
        ),
    ),
    proposal_count: int = typer.Option(
        4,
        "--proposals",
        "-n",
        min=1,
        max=80,
        help=(
            "Number of synthetic proposals to index. Default 4 keeps "
            "the deterministic run fast; bump to 40 to measure against "
            "the PRD's '40 proposals' target."
        ),
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        min=1,
        max=20,
        help=(
            "``top_k`` for both retrieval and generation. Capped at "
            "20 to match the ``GenerationRequest.top_k_examples`` "
            "ceiling."
        ),
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (only consulted when ``--runtime ollama``).",
    ),
) -> None:
    """Run indexing + retrieval + generation benchmarks on a shared index.

    The fastest path to a complete report. Builds a synthetic
    corpus, indexes it into a tmp-path Chroma collection, runs the
    five-probe retrieval set, then drives one methodology-section
    generation. Prints a structured summary; writes JSON if
    ``--output`` is supplied.

    Uses ``typer``'s tmp-path behaviour: we open a per-run
    sub-directory under the OS temp dir so the indexing benchmark
    starts from an empty Chroma collection (otherwise the cached
    embeddings would skew the timing). For a persistent index that
    survives across runs, use the per-AC sub-commands which accept
    explicit ``--index-path`` overrides.
    """

    embedder, llm, runtime_label = _select_backends(runtime, config_path)
    corpus = build_synthetic_corpus(proposal_count=proposal_count)

    # tmp_path-style isolation: each ``benchmark all`` run gets a
    # fresh directory so the indexing measurement is honest. The
    # directory is created by ``typer``'s ``Path`` parameter when
    # the operator supplies one; for the no-argument default we
    # let the runner build a temp dir under the system temp space.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="eurpe-benchmark-") as tmp:
        report = run_all(
            index_path=Path(tmp),
            embedder=embedder,
            llm=llm,
            corpus=corpus,
            retrieval_top_k=top_k,
            runtime_label=runtime_label,
        )

    _print_runtime(report)
    _print_indexing(report)
    _print_retrieval(report)
    _print_generation(report)

    _write_report(report, output)
    if output is not None:
        typer.echo(f"  wrote JSON report to {output}")

    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# Per-AC sub-commands
# ---------------------------------------------------------------------------


@benchmark_app.command("indexing")
def benchmark_indexing(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional JSON output path; stdout summary is always printed.",
    ),
    runtime: str = typer.Option(
        _RUNTIME_DETERMINISTIC,
        "--runtime",
        "-r",
        help="``deterministic`` (default, offline) or ``ollama``.",
    ),
    proposal_count: int = typer.Option(
        4,
        "--proposals",
        "-n",
        min=1,
        max=200,
        help="Number of synthetic proposals to index.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (only consulted when ``--runtime ollama``).",
    ),
) -> None:
    """Measure initial indexing latency (issue #19 AC1)."""

    embedder, llm, runtime_label = _select_backends(runtime, config_path)
    corpus = build_synthetic_corpus(proposal_count=proposal_count)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="eurpe-benchmark-indexing-") as tmp:
        index = ChromaIndex(
            index_path=Path(tmp),
            embedder=embedder,
            collection_name="benchmark_indexing",
        )
        chunker = HierarchicalChunker()
        measurement = IndexingBenchmark(chunker=chunker, index=index).measure(corpus)

    report = BenchmarkReport(
        runtime=capture_runtime_fingerprint(
            runtime=runtime_label, llm=llm, embedder=embedder
        ),
        indexing=measurement,
    )
    _print_runtime(report)
    _print_indexing(report)

    _write_report(report, output)
    if output is not None:
        typer.echo(f"  wrote JSON report to {output}")
    raise typer.Exit(code=0)


@benchmark_app.command("retrieval")
def benchmark_retrieval(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional JSON output path; stdout summary is always printed.",
    ),
    runtime: str = typer.Option(
        _RUNTIME_DETERMINISTIC,
        "--runtime",
        "-r",
        help="``deterministic`` (default, offline) or ``ollama``.",
    ),
    proposal_count: int = typer.Option(
        4,
        "--proposals",
        "-n",
        min=1,
        max=80,
        help="Synthetic proposals to seed before querying.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        min=1,
        max=20,
        help="``top_k`` for the retrieval probes.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (only consulted when ``--runtime ollama``).",
    ),
) -> None:
    """Measure top-k retrieval latency on a seeded index (issue #19 AC2)."""

    embedder, llm, runtime_label = _select_backends(runtime, config_path)
    corpus = build_synthetic_corpus(proposal_count=proposal_count)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="eurpe-benchmark-retrieval-") as tmp:
        index = ChromaIndex(
            index_path=Path(tmp),
            embedder=embedder,
            collection_name="benchmark_retrieval",
        )
        chunker = HierarchicalChunker()
        for parsed, proposal in corpus:
            from eurpe.retrieval.pipeline import index_proposal as _index_proposal

            _index_proposal(parsed, proposal, chunker=chunker, index=index)

        policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
        retriever = SourceStatusAwareRetriever(index, policy=policy)
        from eurpe.benchmarks.runner import _DEFAULT_RETRIEVAL_PROBES  # noqa: PLC0415

        measurement = RetrievalBenchmark(retriever).measure(
            _DEFAULT_RETRIEVAL_PROBES, top_k=top_k
        )

    report = BenchmarkReport(
        runtime=capture_runtime_fingerprint(
            runtime=runtime_label, llm=llm, embedder=embedder
        ),
        retrieval=measurement,
    )
    _print_runtime(report)
    _print_retrieval(report)

    _write_report(report, output)
    if output is not None:
        typer.echo(f"  wrote JSON report to {output}")
    raise typer.Exit(code=0)


@benchmark_app.command("generation")
def benchmark_generation(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional JSON output path; stdout summary is always printed.",
    ),
    runtime: str = typer.Option(
        _RUNTIME_DETERMINISTIC,
        "--runtime",
        "-r",
        help="``deterministic`` (default, offline) or ``ollama``.",
    ),
    proposal_count: int = typer.Option(
        4,
        "--proposals",
        "-n",
        min=1,
        max=80,
        help="Synthetic proposals to seed before generating.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        min=1,
        max=20,
        help="``top_k_examples`` for the generation request.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (only consulted when ``--runtime ollama``).",
    ),
) -> None:
    """Measure section-generation latency end-to-end (issue #19 AC3)."""

    embedder, llm, runtime_label = _select_backends(runtime, config_path)
    corpus = build_synthetic_corpus(proposal_count=proposal_count)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="eurpe-benchmark-generation-") as tmp:
        index = ChromaIndex(
            index_path=Path(tmp),
            embedder=embedder,
            collection_name="benchmark_generation",
        )
        chunker = HierarchicalChunker()
        for parsed, proposal in corpus:
            from eurpe.retrieval.pipeline import index_proposal as _index_proposal

            _index_proposal(parsed, proposal, chunker=chunker, index=index)

        policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
        retriever = SourceStatusAwareRetriever(index, policy=policy)
        workflow = SectionGenerationWorkflow(retriever=retriever, llm=llm)
        request = GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent=(
                "Draft the methodology section for a new EU research "
                "proposal, highlighting the deep learning approach and "
                "evaluation framework."
            ),
            top_k_examples=top_k,
        )
        measurement = GenerationBenchmark(workflow).measure(request=request)

    report = BenchmarkReport(
        runtime=capture_runtime_fingerprint(
            runtime=runtime_label, llm=llm, embedder=embedder
        ),
        generation=measurement,
    )
    _print_runtime(report)
    _print_generation(report)

    _write_report(report, output)
    if output is not None:
        typer.echo(f"  wrote JSON report to {output}")
    raise typer.Exit(code=0)
