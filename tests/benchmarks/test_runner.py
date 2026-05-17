"""Unit tests for :mod:`eurpe.benchmarks.runner`.

These tests pin the three measurement primitives' behaviour at the
Python API level (the CLI surface is covered separately in
``test_cli.py``). Every test runs under the ``no_network`` fixture so
a regression that quietly introduces an outbound call (e.g.,
accidentally constructing :class:`OllamaLLMClient` instead of the
deterministic stub) fails loudly.

Coverage map
------------
* :class:`build_synthetic_corpus` — shape, validation, content.
* :class:`RuntimeFingerprint` / :func:`capture_runtime_fingerprint`
  — required fields populated, deterministic identifiers used.
* :class:`IndexingBenchmark.measure` — happy path returns non-zero
  counts; empty corpus is the documented zero-result path.
* :class:`RetrievalBenchmark.measure` — happy path; empty probes
  raises; per-probe latencies aggregate into min/avg/p95/max.
* :class:`GenerationBenchmark.measure` — happy path returns a
  populated :class:`GenerationMeasurement` whose section type
  matches the request.
* :func:`run_all` — produces a complete :class:`BenchmarkReport`
  with all three measurements set and a valid runtime fingerprint.
* :class:`BenchmarkReport.to_json` — round-trips through JSON
  without drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eurpe.benchmarks import (
    BenchmarkReport,
    GenerationBenchmark,
    IndexingBenchmark,
    RetrievalBenchmark,
    RuntimeFingerprint,
    build_synthetic_corpus,
    run_all,
)
from eurpe.benchmarks.runner import (
    _DEFAULT_RETRIEVAL_PROBES,
    _p95,
    capture_runtime_fingerprint,
)
from eurpe.generation.llm import DeterministicLLMClient
from eurpe.generation.models import GenerationRequest
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
)
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.schema import Programme, SectionType, SourceStatus

# ---------------------------------------------------------------------------
# build_synthetic_corpus
# ---------------------------------------------------------------------------


def test_build_synthetic_corpus_returns_expected_shape() -> None:
    """Default invocation returns a 4-proposal, well-formed corpus."""

    corpus = build_synthetic_corpus()
    assert len(corpus) == 4
    for parsed, proposal in corpus:
        # Each pair must be (ParsedProposal, ProposalMetadata) and the
        # chunker contract requires non-empty sections + a matching
        # source_status / outcome (drift validator enforces this).
        assert parsed.sections, "synthetic proposal must have sections"
        assert parsed.source_path.startswith("synthetic/")
        assert proposal.proposal_title is not None
        assert proposal.outcome == SourceStatus.FUNDED
        assert proposal.programme == Programme.HORIZON_EUROPE


def test_build_synthetic_corpus_respects_size_arguments() -> None:
    """``proposal_count`` and ``sections_per_proposal`` flow through."""

    corpus = build_synthetic_corpus(proposal_count=2, sections_per_proposal=3)
    assert len(corpus) == 2
    for parsed, _ in corpus:
        assert len(parsed.sections) == 3


@pytest.mark.parametrize(
    ("proposal_count", "sections_per_proposal"),
    [(0, 5), (1, 0), (-1, 5)],
)
def test_build_synthetic_corpus_rejects_invalid_sizes(
    proposal_count: int, sections_per_proposal: int
) -> None:
    """Zero/negative counts raise a ValueError rather than silently emit []."""

    with pytest.raises(ValueError):
        build_synthetic_corpus(
            proposal_count=proposal_count,
            sections_per_proposal=sections_per_proposal,
        )


# ---------------------------------------------------------------------------
# RuntimeFingerprint
# ---------------------------------------------------------------------------


def test_capture_runtime_fingerprint_records_required_fields(no_network: None) -> None:
    """All required fields are populated and the runtime label is preserved."""

    llm = DeterministicLLMClient()
    embedder = DeterministicHashEmbedder(dimension=64)

    rt = capture_runtime_fingerprint(runtime="deterministic", llm=llm, embedder=embedder)
    assert rt.runtime == "deterministic"
    assert rt.llm_model == llm.model
    assert "DeterministicHashEmbedder" in rt.embedder
    # python_version + platform are non-empty strings on every supported OS.
    assert rt.python_version
    assert "/" in rt.platform


def test_runtime_fingerprint_is_frozen() -> None:
    """The model_config marks it frozen so the report record is immutable."""

    rt = RuntimeFingerprint(
        runtime="deterministic",
        llm_model="deterministic-stub-v1",
        embedder="DeterministicHashEmbedder",
        python_version="3.13.0",
        platform="Darwin/arm64",
        cpu_count=8,
    )
    # Pydantic v2 frozen models raise ValidationError on attribute assignment.
    with pytest.raises(ValidationError):
        rt.runtime = "ollama"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IndexingBenchmark
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_index(tmp_path: Path) -> ChromaIndex:
    """Open a fresh Chroma collection backed by the hash embedder.

    Per-test ``tmp_path`` keeps each measurement honest — no warm
    cache leaks from a previous test's index.
    """

    return ChromaIndex(
        index_path=tmp_path,
        embedder=DeterministicHashEmbedder(dimension=128),
        collection_name="benchmark_unit_test",
    )


def test_indexing_benchmark_measures_corpus(
    deterministic_index: ChromaIndex, no_network: None
) -> None:
    """Happy path: a 2-proposal corpus produces non-zero chunk + elapsed counts."""

    corpus = build_synthetic_corpus(proposal_count=2)
    benchmark = IndexingBenchmark(chunker=HierarchicalChunker(), index=deterministic_index)
    result = benchmark.measure(corpus)

    assert result.proposal_count == 2
    assert result.chunk_count > 0, "synthetic proposals must emit at least one chunk"
    # ``per_proposal_ms_avg`` is a float derived from elapsed_ms; with
    # a non-empty corpus the value is well-defined.
    assert result.per_proposal_ms_avg >= 0.0
    # ``chunks_per_second`` is 0 only when elapsed_ms rounded to 0 ms —
    # both outcomes are valid on a fast machine.
    assert result.chunks_per_second >= 0.0


def test_indexing_benchmark_empty_corpus_returns_zeros(
    deterministic_index: ChromaIndex, no_network: None
) -> None:
    """Empty corpus is the documented zero-result path (no exception)."""

    benchmark = IndexingBenchmark(chunker=HierarchicalChunker(), index=deterministic_index)
    result = benchmark.measure([])

    assert result.proposal_count == 0
    assert result.chunk_count == 0
    assert result.elapsed_ms == 0
    assert result.per_proposal_ms_avg == 0.0
    assert result.chunks_per_second == 0.0


# ---------------------------------------------------------------------------
# RetrievalBenchmark
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_retriever(tmp_path: Path) -> SourceStatusAwareRetriever:
    """Build a retriever over a 4-proposal synthetic corpus.

    The lenient policy mirrors ``conftest.py::deterministic_workflow``:
    the deterministic hash embedder produces modest cosine scores
    that sit below the default 0.30 threshold, so the override is
    required for the retriever to return results.
    """

    embedder = DeterministicHashEmbedder(dimension=128)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="benchmark_retrieval_fixture",
    )
    chunker = HierarchicalChunker()
    from eurpe.retrieval.pipeline import index_proposal

    for parsed, proposal in build_synthetic_corpus(proposal_count=4):
        index_proposal(parsed, proposal, chunker=chunker, index=index)
    policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
    return SourceStatusAwareRetriever(index, policy=policy)


def test_retrieval_benchmark_aggregates_per_query_latency(
    seeded_retriever: SourceStatusAwareRetriever, no_network: None
) -> None:
    """min/avg/p95/max are populated and ordered correctly."""

    probes = ["methodology approach", "impact pathway", "consortium roles"]
    result = RetrievalBenchmark(seeded_retriever).measure(probes, top_k=3)

    assert result.query_count == len(probes)
    assert result.top_k == 3
    assert result.elapsed_ms_min <= result.elapsed_ms_avg <= result.elapsed_ms_max
    assert result.elapsed_ms_p95 >= result.elapsed_ms_min
    # The result_count_avg must be > 0 because the lenient policy
    # ensures the deterministic embedder returns matches for these
    # token-overlapping probes.
    assert result.result_count_avg > 0.0


def test_retrieval_benchmark_rejects_empty_probes(
    seeded_retriever: SourceStatusAwareRetriever,
) -> None:
    """Empty probe list raises ValueError — measuring nothing is a wiring bug."""

    with pytest.raises(ValueError, match="at least one probe"):
        RetrievalBenchmark(seeded_retriever).measure([])


def test_retrieval_benchmark_rejects_empty_probe_string(
    seeded_retriever: SourceStatusAwareRetriever,
) -> None:
    """An empty string inside the probe list raises with the index named."""

    with pytest.raises(ValueError, match=r"probes\[1\]"):
        RetrievalBenchmark(seeded_retriever).measure(["ok", ""])


def test_retrieval_benchmark_with_default_probes(
    seeded_retriever: SourceStatusAwareRetriever, no_network: None
) -> None:
    """The five default probes shipped with the harness all return non-empty."""

    result = RetrievalBenchmark(seeded_retriever).measure(_DEFAULT_RETRIEVAL_PROBES, top_k=5)
    assert result.query_count == len(_DEFAULT_RETRIEVAL_PROBES)
    assert result.result_count_avg > 0.0


# ---------------------------------------------------------------------------
# GenerationBenchmark
# ---------------------------------------------------------------------------


def test_generation_benchmark_measures_section_draft(
    deterministic_workflow: SectionGenerationWorkflow, no_network: None
) -> None:
    """End-to-end generation timing populates every field of the measurement."""

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe the proposed methodology for this work.",
        top_k_examples=3,
    )
    result = GenerationBenchmark(deterministic_workflow).measure(request=request)

    assert result.section_type == "methodology"
    assert result.top_k_examples == 3
    # Deterministic backend is fast — elapsed_ms may be 0 on a very
    # quick run, but it is bounded below at 0 by the validator.
    assert result.elapsed_ms >= 0
    assert result.prompt_length > 0
    assert result.draft_length > 0
    # The deterministic workflow fixture in conftest.py provides
    # citations; the count should be in [1, top_k_examples].
    assert 1 <= result.citation_count <= 3


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def test_run_all_produces_complete_report(tmp_path: Path, no_network: None) -> None:
    """End-to-end aggregator wires all three benchmarks deterministically."""

    report = run_all(index_path=tmp_path)

    # Every measurement is set — this is the AC2/3 "report includes
    # model/runtime configuration" contract.
    assert report.indexing is not None
    assert report.retrieval is not None
    assert report.generation is not None
    assert report.runtime.runtime == "deterministic"
    assert report.runtime.llm_model
    assert report.runtime.embedder.startswith("DeterministicHashEmbedder")
    # The runtime fingerprint must always be populated regardless of
    # which sub-command produced the report.
    assert report.runtime.python_version
    assert report.runtime.platform


def test_run_all_supports_custom_corpus_and_probes(tmp_path: Path, no_network: None) -> None:
    """Caller-supplied corpus + probes flow through to the measurements."""

    custom_corpus = build_synthetic_corpus(proposal_count=2, sections_per_proposal=3)
    custom_probes = ["custom probe text"]
    report = run_all(
        index_path=tmp_path,
        corpus=custom_corpus,
        retrieval_probes=custom_probes,
        retrieval_top_k=2,
    )
    assert report.indexing is not None
    assert report.indexing.proposal_count == 2
    assert report.retrieval is not None
    assert report.retrieval.query_count == 1
    assert report.retrieval.top_k == 2


def test_report_to_json_round_trip(tmp_path: Path, no_network: None) -> None:
    """``to_json`` produces a parseable string with the same fields."""

    report = run_all(index_path=tmp_path)
    payload = report.to_json()
    parsed = json.loads(payload)
    # Top-level shape: runtime + three measurement keys.
    assert "runtime" in parsed
    assert "indexing" in parsed
    assert "retrieval" in parsed
    assert "generation" in parsed
    # Round-trip through Pydantic to confirm the JSON is valid against
    # the model — this catches accidental drift in field names.
    rehydrated = BenchmarkReport.model_validate(parsed)
    assert rehydrated.runtime.runtime == report.runtime.runtime
    assert rehydrated.indexing is not None
    assert rehydrated.indexing.chunk_count == report.indexing.chunk_count


# ---------------------------------------------------------------------------
# _p95
# ---------------------------------------------------------------------------


def test_p95_with_single_value() -> None:
    """A single sample is its own p95."""

    assert _p95([42]) == 42


def test_p95_with_uniform_values() -> None:
    """Uniform input → that value is the p95."""

    assert _p95([10, 10, 10, 10]) == 10


def test_p95_with_long_tail() -> None:
    """A clear long-tail input: p95 sits between the 95th and max sample."""

    # 100 samples 1..100; p95 = linear interpolation between sorted[94] (95) and
    # sorted[95] (96) at frac 0.05 → 95.05 → rounded to 95.
    values = list(range(1, 101))
    assert _p95(values) == 95


def test_p95_rejects_empty() -> None:
    """Empty input is undefined; the function raises rather than guess."""

    with pytest.raises(ValueError, match="at least one value"):
        _p95([])
