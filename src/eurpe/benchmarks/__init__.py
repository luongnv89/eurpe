"""Performance benchmarks for EURPE v1 PRD targets.

What this package is (and is not)
---------------------------------
This is the **operator-facing benchmark harness** for Task 3.5 / issue
#19. It measures three latencies the PRD pins as v1 release targets:

* **Initial indexing** — parse + chunk + embed + upsert for a fixture
  corpus. PRD target: ``<2 hours on Mac M1 32 GB for 40 proposals``.
* **Internal retrieval** — top-k retrieval latency against the indexed
  corpus. PRD target: ``<2 s for top-k retrieval on the indexed v1
  corpus``.
* **Section generation** — end-to-end ``retrieve → prompt → LLM →
  validate → assemble`` latency for one section draft. PRD target:
  ``<2 min on M1, <30 s on DGX (5–10 page section)``.

The harness is NOT a pytest-benchmark suite. Tasks.md Acceptance
Criteria #1–3 read "Benchmark measures X" — operator-facing wording —
and the codebase has no ``pytest-benchmark`` dependency. We keep the
``eurpe analytics export`` / ``eurpe smoke`` convention: a Typer
sub-app that an operator runs to produce a report, plus standard
``pytest`` tests that verify the harness behaves correctly under the
``no_network`` fixture.

Offline-by-default
------------------
Like the rest of EURPE the benchmarks must run with zero external
network. The default code path uses
:class:`~eurpe.retrieval.embeddings.DeterministicHashEmbedder` and
:class:`~eurpe.generation.llm.DeterministicLLMClient` so the harness
produces a complete report on a CI machine with no Ollama daemon.
Operators measuring against the PRD's "<2 min on M1" target pass
``--runtime ollama`` to swap in the real backends; the report names
the actual runtime + model so a reviewer can tell which numbers came
from which path. Deterministic timings against a PRD target meant for
a real LLM would be misleading without that label — AC3 explicitly
requires "reports model/runtime configuration".

The fixture problem
-------------------
``tests/fixtures/pdfs/`` is intentionally empty (README only — real
proposal PDFs are confidential and out of repo). The indexing
benchmark therefore needs two fixture paths:

* **Synthetic** (default) — programmatically built
  :class:`~eurpe.ingestion.models.ParsedProposal` records. Used by the
  test suite and as the out-of-the-box CLI default so a fresh clone
  can run ``eurpe benchmark all`` without any user data.
* **Real corpus** (``--corpus PATH``) — an operator-supplied directory
  of PDFs. Used when measuring against the PRD's "40 proposals" target
  on a representative call. Out of scope for CI but the contract the
  PRD numbers were written against.

What this package exposes
-------------------------
* :class:`BenchmarkReport` — pydantic record of one full benchmark
  run. Carries the three measurements + the runtime fingerprint.
* :class:`IndexingBenchmark`, :class:`RetrievalBenchmark`,
  :class:`GenerationBenchmark` — measurement primitives. Each takes
  pre-built backends so the CLI can wire them once and the tests can
  stub them in isolation.
* :func:`run_all` — convenience wrapper that runs all three in order
  on a shared in-memory index.
* :func:`build_synthetic_corpus` — generator for the default fixture
  corpus.
"""

from __future__ import annotations

from eurpe.benchmarks.runner import (
    BenchmarkReport,
    GenerationBenchmark,
    GenerationMeasurement,
    IndexingBenchmark,
    IndexingMeasurement,
    RetrievalBenchmark,
    RetrievalMeasurement,
    RuntimeFingerprint,
    build_synthetic_corpus,
    run_all,
)

__all__ = [
    "BenchmarkReport",
    "GenerationBenchmark",
    "GenerationMeasurement",
    "IndexingBenchmark",
    "IndexingMeasurement",
    "RetrievalBenchmark",
    "RetrievalMeasurement",
    "RuntimeFingerprint",
    "build_synthetic_corpus",
    "run_all",
]
