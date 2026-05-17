"""Measurement primitives for the EURPE v1 performance benchmarks.

The runner module is the *engine* of the benchmark harness. It owns
the three measurement classes (indexing / retrieval / generation), the
:class:`BenchmarkReport` aggregate, and a :func:`run_all` convenience
that wires them on a shared in-memory index.

Why three named classes instead of one ``run_all``?
---------------------------------------------------
Each measurement is independently useful:

* CI may run only the retrieval and generation benchmarks (they are
  fast under the deterministic backends).
* An operator measuring a fresh corpus runs the indexing benchmark
  once and then queries the same index many times.
* Tests assert on each measurement in isolation.

A single monolithic function would fold the three signatures into one
hard-to-test entry point. Splitting also lets the CLI's
``eurpe benchmark indexing`` / ``benchmark retrieval`` /
``benchmark generation`` sub-commands each call exactly one primitive.

Timing convention
-----------------
All durations are measured with :func:`time.monotonic_ns` and reported
in milliseconds (integer). ``monotonic_ns`` is the right clock for
elapsed-time measurement — it is not affected by NTP adjustments and
has nanosecond resolution on every supported platform. The PRD
targets are quoted in seconds/minutes; the millisecond units convert
cleanly without loss.

Offline-by-default
------------------
Every backend defaults to deterministic, zero-network implementations.
The CLI may swap real backends (Ollama LLM / embedder, on-disk
ChromaIndex) when ``--runtime ollama`` is passed; the report records
which runtime was used so a reviewer can map numbers to the PRD
targets correctly.
"""

from __future__ import annotations

import logging
import os
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from eurpe.generation.llm import DeterministicLLMClient, LLMClient
from eurpe.generation.models import GenerationDraft, GenerationRequest
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.ingestion.models import ParsedProposal, ParsedSection
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    Embedder,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
)
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.retrieval.pipeline import index_proposal
from eurpe.schema import (
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime fingerprint
# ---------------------------------------------------------------------------


class RuntimeFingerprint(BaseModel):
    """Snapshot of the environment that produced a benchmark report.

    AC3 of issue #19 explicitly requires "reports model/runtime
    configuration." The other two ACs implicitly do too — a "2.4 ms
    retrieval" number means nothing unless the reviewer can tell
    whether it was measured under the deterministic hash embedder or
    a real Ollama daemon. Without this fingerprint a benchmark
    report cannot be honestly compared against PRD targets that
    assume a 14B–32B model.

    The fields are intentionally narrow: no PID, no hostname, no
    username. The hardware fingerprint is the machine *class*
    (``Darwin / arm64``) so two M1 runs are comparable, not a unique
    identifier of the developer's laptop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: str = Field(
        description=(
            "Backend identifier: ``deterministic`` (no network, hash "
            "embedder + echo LLM) or ``ollama`` (real local daemon). "
            "Determines how to interpret the timings against PRD targets."
        ),
    )
    llm_model: str = Field(
        description=(
            "Identifier of the LLM client's ``model`` property. For the "
            "deterministic backend this is ``deterministic-echo``; for "
            "Ollama it is the operator-configured model name."
        ),
    )
    embedder: str = Field(
        description=(
            "Embedder backend identifier: ``deterministic-hash`` or "
            "``ollama:<model>``. Latency and recall differ by ~100×"
            " between the two so the field is essential for comparison."
        ),
    )
    python_version: str = Field(
        description="Output of :func:`platform.python_version`.",
    )
    platform: str = Field(
        description=(
            "``<system>/<machine>`` (e.g., ``Darwin/arm64``, "
            "``Linux/x86_64``). Coarse enough not to identify a "
            "specific machine; precise enough to distinguish the two "
            "PRD target environments (M1 vs DGX)."
        ),
    )
    cpu_count: int | None = Field(
        default=None,
        ge=1,
        description="``os.cpu_count()`` at run time, if reported by the OS.",
    )


def capture_runtime_fingerprint(
    *,
    runtime: str,
    llm: LLMClient,
    embedder: Embedder,
) -> RuntimeFingerprint:
    """Build a :class:`RuntimeFingerprint` from the live backends.

    Pulls the LLM ``model`` property and the embedder's class name +
    optional ``model`` attribute. Used by :func:`run_all` and the CLI
    so every report carries the same shape of metadata regardless of
    which sub-command produced it.
    """

    if hasattr(embedder, "model"):
        embedder_id = f"{embedder.__class__.__name__}:{embedder.model}"
    else:
        embedder_id = embedder.__class__.__name__
    return RuntimeFingerprint(
        runtime=runtime,
        llm_model=llm.model,
        embedder=embedder_id,
        python_version=platform.python_version(),
        platform=f"{platform.system()}/{platform.machine()}",
        cpu_count=os.cpu_count(),
    )


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------


# A small, prose-shaped section template used to bulk out synthetic
# proposals. The text is intentionally generic (deep-learning
# methodology jargon) so the deterministic-hash embedder produces
# non-trivial token-overlap scores against the retrieval probe.
_SYNTHETIC_SECTION_TEMPLATE = (
    "This section discusses the proposed {topic} methodology using a "
    "deep learning approach with attention mechanisms. The work plan "
    "covers data collection, model training, validation against a "
    "domain corpus, and an evaluation framework grounded in EU "
    "research-impact metrics. Risks include dataset bias, model drift, "
    "and partner availability; mitigations are documented in the work "
    "plan and revisited at each milestone."
)


def _synthetic_parsed_proposal(
    document_id: str,
    *,
    section_count: int = 5,
    topic: str = "methodology",
) -> ParsedProposal:
    """Build a small in-memory :class:`ParsedProposal` for benchmarking.

    Each synthetic proposal carries ``section_count`` short sections so
    the chunker emits a handful of chunks per proposal — enough to
    exercise the index and the retriever without depending on real
    PDFs (none of which can ship in this repo).
    """

    sections = [
        ParsedSection(
            heading=f"{i + 1}.{j + 1} {topic.title()} chapter {j + 1}",
            level=2,
            text=_SYNTHETIC_SECTION_TEMPLATE.format(topic=f"{topic} part {j + 1}"),
            page_start=j + 1,
            page_end=j + 1,
        )
        for i in (0,)
        for j in range(section_count)
    ]
    return ParsedProposal(
        source_path=f"synthetic/{document_id}.pdf",
        title=f"Synthetic proposal {document_id}",
        sections=sections,
        page_count=section_count,
        parser="benchmark-synthetic",
    )


def _synthetic_proposal_metadata(
    document_id: str,
    *,
    outcome: SourceStatus = SourceStatus.FUNDED,
    programme: Programme = Programme.HORIZON_EUROPE,
) -> ProposalMetadata:
    """Mirror of the parsed proposal: matching :class:`ProposalMetadata`.

    The chunker pairs ``ParsedProposal`` with ``ProposalMetadata`` to
    stamp every emitted chunk with programme/call/outcome. The two
    halves are produced together so the drift validator on
    :class:`~eurpe.schema.ChunkMetadata` is satisfied by construction.
    """

    return ProposalMetadata(
        programme=programme,
        call_id="HORIZON-CL5-2024-D3-02",
        year=2024,
        outcome=outcome,
        proposal_title=f"Synthetic proposal {document_id}",
        source_path=f"synthetic/{document_id}.pdf",
    )


def build_synthetic_corpus(
    *,
    proposal_count: int = 4,
    sections_per_proposal: int = 5,
) -> list[tuple[ParsedProposal, ProposalMetadata]]:
    """Generate a synthetic ``(ParsedProposal, ProposalMetadata)`` corpus.

    The default size (4 proposals × 5 sections) keeps the test path
    fast — well under a second to index — while still producing enough
    chunks to make retrieval scores non-degenerate. Operators
    measuring against the PRD's "40 proposals" target pass a real
    corpus via the CLI's ``--corpus PATH`` flag; this function is the
    deterministic fallback that always works.
    """

    if proposal_count < 1:
        raise ValueError(f"proposal_count must be >= 1, got {proposal_count}")
    if sections_per_proposal < 1:
        raise ValueError(
            f"sections_per_proposal must be >= 1, got {sections_per_proposal}"
        )
    return [
        (
            _synthetic_parsed_proposal(
                f"synthetic-{i:03d}",
                section_count=sections_per_proposal,
                topic="methodology" if i % 2 == 0 else "impact pathway",
            ),
            _synthetic_proposal_metadata(f"synthetic-{i:03d}"),
        )
        for i in range(proposal_count)
    ]


# ---------------------------------------------------------------------------
# Measurement records
# ---------------------------------------------------------------------------


class IndexingMeasurement(BaseModel):
    """Result of the indexing benchmark.

    ``elapsed_ms`` is the wall-clock time to chunk + embed + upsert
    every proposal in the corpus. ``chunks_per_second`` is the derived
    throughput — easier to read than a raw "8 ms for 12 chunks" line
    when the corpus size varies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_count: int = Field(
        ge=0,
        description="Number of proposals indexed.",
    )
    chunk_count: int = Field(
        ge=0,
        description="Total chunks upserted across all proposals.",
    )
    elapsed_ms: int = Field(
        ge=0,
        description="Wall-clock elapsed milliseconds for the full corpus.",
    )
    per_proposal_ms_avg: float = Field(
        ge=0.0,
        description=(
            "Mean per-proposal elapsed time in milliseconds. ``0`` when "
            "``proposal_count`` is ``0``."
        ),
    )
    chunks_per_second: float = Field(
        ge=0.0,
        description=(
            "Derived throughput (``chunk_count / elapsed_seconds``). "
            "``0`` when ``elapsed_ms`` is ``0``."
        ),
    )


class RetrievalMeasurement(BaseModel):
    """Result of the retrieval benchmark.

    Stores the per-query distribution because retrieval latency has
    long-tailed jitter on cold collections — the mean alone hides
    that. ``p95_ms`` is the headline number for the PRD's "<2 s for
    top-k" target; the user cares about the slow case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_count: int = Field(
        ge=0,
        description="Number of retrieval calls timed.",
    )
    top_k: int = Field(
        ge=1,
        description="``top_k`` passed to each retrieval call.",
    )
    elapsed_ms_min: int = Field(ge=0, description="Fastest query, ms.")
    elapsed_ms_avg: float = Field(ge=0.0, description="Mean query latency, ms.")
    elapsed_ms_p95: int = Field(
        ge=0,
        description=(
            "95th percentile query latency, ms. Headline number for "
            "the PRD's '<2 s top-k' target."
        ),
    )
    elapsed_ms_max: int = Field(ge=0, description="Slowest query, ms.")
    result_count_avg: float = Field(
        ge=0.0,
        description=(
            "Mean number of results returned per query (post-policy "
            "filtering). Sanity-check: a queryset with too many empty "
            "returns invalidates the latency numbers."
        ),
    )


class GenerationMeasurement(BaseModel):
    """Result of the section-generation benchmark.

    Records both ``elapsed_ms`` (wall clock) and the per-step
    ``citation_count`` so a reviewer can tell whether a fast number
    came from a degenerate "no retrieved evidence → tiny prompt" path
    versus an honest end-to-end pass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_type: str = Field(description="``SectionType.value`` exercised.")
    elapsed_ms: int = Field(
        ge=0,
        description=(
            "Wall-clock milliseconds for one ``workflow.run()`` call: "
            "retrieve → prompt → generate → validate → assemble."
        ),
    )
    citation_count: int = Field(
        ge=0,
        description=(
            "Number of citations attached to the produced draft. ``0`` "
            "for a degenerate path; a healthy run produces 1–``top_k``."
        ),
    )
    top_k_examples: int = Field(
        ge=1,
        description="``top_k_examples`` requested for the underlying retrieval.",
    )
    prompt_length: int = Field(
        ge=0,
        description=(
            "Character length of the prompt sent to the LLM. Cheap "
            "sanity-check that the prompt was actually built before "
            "timing the LLM call."
        ),
    )
    draft_length: int = Field(
        ge=0,
        description="Character length of the produced draft text.",
    )


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


class BenchmarkReport(BaseModel):
    """Aggregate report for one benchmark run.

    The shape is forward-compatible: each measurement field is
    optional so a sub-command that runs only one benchmark can
    produce a partial report without sentinel zeros. Persisted as
    JSON by the CLI; Task 3.7 (pilot validation) and future CI
    regression checks will read this file.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp the report was produced.",
    )
    runtime: RuntimeFingerprint = Field(
        description="Environment that produced the measurements.",
    )
    indexing: IndexingMeasurement | None = Field(
        default=None,
        description="Indexing benchmark result; ``None`` if not run.",
    )
    retrieval: RetrievalMeasurement | None = Field(
        default=None,
        description="Retrieval benchmark result; ``None`` if not run.",
    )
    generation: GenerationMeasurement | None = Field(
        default=None,
        description="Generation benchmark result; ``None`` if not run.",
    )

    def to_json(self) -> str:
        """Render the report as pretty-printed JSON.

        Tiny convenience so the CLI does not have to duplicate the
        ``model_dump(mode='json')`` + ``indent=2`` recipe. Pretty
        printing keeps the file diff-friendly across CI runs.
        """

        import json

        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


def _elapsed_ms_since(start_ns: int) -> int:
    """Wall-clock milliseconds between ``start_ns`` and ``monotonic_ns()`` now.

    Centralised so every benchmark uses the same clock and the same
    nanosecond-to-millisecond conversion. Integer division here is
    deliberate: sub-millisecond noise is below the precision the PRD
    targets care about, and the integer makes test assertions stable.
    """

    return (time.monotonic_ns() - start_ns) // 1_000_000


class IndexingBenchmark:
    """Measure end-to-end indexing latency on a corpus.

    Construct once with a pre-built :class:`HierarchicalChunker` and
    :class:`ChromaIndex`; call :meth:`measure` per corpus. The
    benchmark does NOT own the index lifecycle — the caller decides
    whether to use a tmp_path-scoped collection or a persistent one.

    Why this lives behind a class rather than a free function
    ---------------------------------------------------------
    The chunker + index pair are heavy to build (Chroma client open
    + embedder construction). The CLI builds them once and may run
    several measurements (e.g., warm-up + measured) against the same
    pair; tests stub them with deterministic equivalents. A class
    captures that "build once, measure many" lifecycle naturally.
    """

    def __init__(
        self,
        *,
        chunker: HierarchicalChunker,
        index: ChromaIndex,
    ) -> None:
        self._chunker = chunker
        self._index = index

    def measure(
        self,
        corpus: Sequence[tuple[ParsedProposal, ProposalMetadata]],
    ) -> IndexingMeasurement:
        """Time the chunk+embed+upsert chain across ``corpus``.

        ``corpus`` is a sequence of pre-parsed proposals so the
        measurement excludes PDF parsing time (which is heavily
        I/O-bound and would dominate the number under realistic
        Docling settings — out of scope for the indexing latency
        target). When operators care about parsing time they can
        run the parser separately and add the two numbers.

        Returns an :class:`IndexingMeasurement`. Empty corpus is
        valid and returns zeroes — the CLI surfaces a warning when
        it sees a zero result, but the engine does not error.
        """

        proposal_count = len(corpus)
        if proposal_count == 0:
            return IndexingMeasurement(
                proposal_count=0,
                chunk_count=0,
                elapsed_ms=0,
                per_proposal_ms_avg=0.0,
                chunks_per_second=0.0,
            )

        start_ns = time.monotonic_ns()
        chunk_count = 0
        for parsed, proposal in corpus:
            chunk_count += index_proposal(
                parsed,
                proposal,
                chunker=self._chunker,
                index=self._index,
            )
        elapsed_ms = _elapsed_ms_since(start_ns)

        # ``per_proposal_ms_avg`` is the average wall-clock latency
        # per proposal — what operators care about when sizing a
        # batch ingest. Float division here so a fast 5-ms corpus on
        # 4 proposals reports 1.25 ms rather than the integer 1.
        per_proposal_ms_avg = elapsed_ms / proposal_count

        if elapsed_ms == 0:
            # Avoid ZeroDivisionError on the synthetic-tiny corpus
            # path. ``0`` is a defensible default that the CLI's
            # summary line handles ("n/a — too fast to measure").
            chunks_per_second = 0.0
        else:
            chunks_per_second = chunk_count / (elapsed_ms / 1000.0)

        return IndexingMeasurement(
            proposal_count=proposal_count,
            chunk_count=chunk_count,
            elapsed_ms=elapsed_ms,
            per_proposal_ms_avg=per_proposal_ms_avg,
            chunks_per_second=chunks_per_second,
        )


class RetrievalBenchmark:
    """Measure top-k retrieval latency.

    Wraps a pre-built :class:`SourceStatusAwareRetriever`. ``measure``
    runs each probe in ``probes`` once and returns the distribution
    statistics. We don't loop a single probe N times because that
    would hide the cold-vs-warm difference (Chroma caches the
    embedding model on first call); a diverse probe set surfaces it.
    """

    def __init__(self, retriever: SourceStatusAwareRetriever) -> None:
        self._retriever = retriever

    def measure(
        self,
        probes: Sequence[str],
        *,
        top_k: int = 5,
    ) -> RetrievalMeasurement:
        """Time each query in ``probes`` and aggregate the latencies.

        ``probes`` must contain at least one non-empty query string.
        An empty list raises :class:`ValueError` — measuring zero
        queries returns nothing meaningful and would silently hide a
        CLI-wiring bug.

        Returns a :class:`RetrievalMeasurement` with min / mean / p95
        / max in milliseconds plus the average result count. ``p95``
        is computed as :func:`statistics.quantiles` boundary (the
        "inclusive" method) so even a 2-probe set yields a defined
        value rather than raising.
        """

        if not probes:
            raise ValueError("RetrievalBenchmark.measure requires at least one probe.")
        for i, probe in enumerate(probes):
            if not probe:
                raise ValueError(f"probes[{i}] must be a non-empty string.")

        durations: list[int] = []
        result_counts: list[int] = []
        for probe in probes:
            start_ns = time.monotonic_ns()
            results = self._retriever.retrieve(probe, top_k=top_k)
            durations.append(_elapsed_ms_since(start_ns))
            result_counts.append(len(results))

        return RetrievalMeasurement(
            query_count=len(durations),
            top_k=top_k,
            elapsed_ms_min=min(durations),
            elapsed_ms_avg=statistics.fmean(durations),
            elapsed_ms_p95=_p95(durations),
            elapsed_ms_max=max(durations),
            result_count_avg=statistics.fmean(result_counts),
        )


def _p95(values: Sequence[int]) -> int:
    """Return the 95th-percentile value of ``values`` as an int.

    Centralised so :class:`RetrievalBenchmark` and any future
    multi-query measurement use the same definition. We use the
    "inclusive" method (matches NumPy's default
    ``np.percentile(..., method='linear')`` for the common case) and
    round to the nearest integer — the timings are already in
    millisecond units, sub-millisecond precision is noise.

    Special case: with one sample, the p95 is that sample.
    """

    if len(values) == 0:
        raise ValueError("_p95 requires at least one value.")
    if len(values) == 1:
        return int(values[0])
    sorted_values = sorted(values)
    # Linear interpolation between sorted values, matching
    # numpy.percentile's default behaviour.
    pos = 0.95 * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    interp = sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])
    return int(round(interp))


class GenerationBenchmark:
    """Measure end-to-end section-generation latency.

    Wraps a :class:`SectionGenerationWorkflow`. ``measure`` runs one
    ``workflow.run`` call and reports the wall-clock plus a couple of
    sanity-check fields (citation count, prompt + draft length) so a
    reviewer can tell whether a fast result came from an honest run
    or a degenerate empty-retrieval path.

    For the PRD's "<2 min on M1" target, an operator runs this with
    ``--runtime ollama`` and a real model. Under the deterministic
    LLM the number is in the single-digit milliseconds and exists
    primarily to prove the wiring works end-to-end.
    """

    def __init__(self, workflow: SectionGenerationWorkflow) -> None:
        self._workflow = workflow

    def measure(
        self,
        *,
        request: GenerationRequest,
    ) -> GenerationMeasurement:
        """Run one section-generation request and return the timing.

        The ``request`` is supplied by the caller so the CLI can
        drive a representative ``GenerationRequest`` (methodology +
        meaningful intent) and tests can use a tiny fixed request.
        """

        start_ns = time.monotonic_ns()
        draft: GenerationDraft = self._workflow.run(request)
        elapsed_ms = _elapsed_ms_since(start_ns)
        return GenerationMeasurement(
            section_type=request.section_type.value,
            elapsed_ms=elapsed_ms,
            citation_count=len(draft.citations),
            top_k_examples=request.top_k_examples,
            prompt_length=len(draft.prompt_used),
            draft_length=len(draft.text),
        )


# ---------------------------------------------------------------------------
# Convenience: run all three with a shared index
# ---------------------------------------------------------------------------


_DEFAULT_RETRIEVAL_PROBES: tuple[str, ...] = (
    "methodology approach for evaluation",
    "impact pathway expected outcomes",
    "consortium and partner roles",
    "work plan and risk mitigation",
    "data collection and validation framework",
)

_DEFAULT_GENERATION_INTENT = (
    "Draft the methodology section for a new EU research proposal, "
    "highlighting the deep learning approach and evaluation framework."
)


def run_all(
    *,
    index_path: Path,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    corpus: Sequence[tuple[ParsedProposal, ProposalMetadata]] | None = None,
    retrieval_probes: Sequence[str] | None = None,
    retrieval_top_k: int = 5,
    generation_request: GenerationRequest | None = None,
    collection_name: str = "benchmark_run",
    runtime_label: str = "deterministic",
) -> BenchmarkReport:
    """Run all three benchmarks against a freshly-built in-memory index.

    Wires the default deterministic backends so a fresh clone can
    call ``run_all(index_path=tmp_path)`` and get a complete report
    without any user-supplied input. Real backends (Ollama) are
    passed in via ``embedder`` and ``llm`` from the CLI when the
    operator requests them.

    ``index_path`` is the directory Chroma persists into. Callers
    that want a one-shot in-memory measurement pass a ``tmp_path``;
    callers that want a persistent index pass the configured corpus
    path. Either way the collection is created freshly so the
    indexing benchmark is honest (no warm cache from a previous run).
    """

    embedder = embedder if embedder is not None else DeterministicHashEmbedder(dimension=128)
    llm = llm if llm is not None else DeterministicLLMClient()
    corpus = corpus if corpus is not None else build_synthetic_corpus()
    probes = retrieval_probes if retrieval_probes is not None else _DEFAULT_RETRIEVAL_PROBES
    request = generation_request or GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent=_DEFAULT_GENERATION_INTENT,
        top_k_examples=retrieval_top_k,
    )

    chunker = HierarchicalChunker()
    index = ChromaIndex(
        index_path=index_path,
        embedder=embedder,
        collection_name=collection_name,
    )

    # Indexing: must run first because retrieval + generation need
    # an index with content. Using a lenient policy keeps the
    # deterministic hash embedder's modest cosine scores above the
    # threshold (mirrors the ``deterministic_workflow`` test
    # fixture's reasoning in conftest.py).
    indexing = IndexingBenchmark(chunker=chunker, index=index).measure(corpus)

    policy = RetrievalPolicy(
        relevance_threshold=0.0,
        max_rejected_fraction=1.0,
    )
    retriever = SourceStatusAwareRetriever(index, policy=policy)
    retrieval = RetrievalBenchmark(retriever).measure(probes, top_k=retrieval_top_k)

    workflow = SectionGenerationWorkflow(retriever=retriever, llm=llm)
    generation = GenerationBenchmark(workflow).measure(request=request)

    return BenchmarkReport(
        runtime=capture_runtime_fingerprint(
            runtime=runtime_label,
            llm=llm,
            embedder=embedder,
        ),
        indexing=indexing,
        retrieval=retrieval,
        generation=generation,
    )
