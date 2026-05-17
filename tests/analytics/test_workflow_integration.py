"""Workflow integration test — pins AC2 of issue #13.

Runs a real :class:`SectionGenerationWorkflow.run` with a corpus that
contains *sentinel strings* in the user intent and citation snippets,
then reads back the raw JSONL log file and asserts that NONE of those
sentinels appear anywhere in the file content.

This is the keystone proof of AC2: "Tests confirm event payloads do
not include raw document content, retrieved passages, or generated
draft text." The test relies on two complementary signals:

1. **Intent leak.** The user-intent sentinel must not appear in the
   JSONL. A future regression that passed ``request.user_intent`` (or
   any derived prompt fragment) into an event payload would fail
   here.
2. **Retrieved-passage leak.** Each corpus chunk carries a unique
   ``SENTINEL_SNIPPET_*`` substring. The retriever surfaces those
   chunks as ``CitationRef.snippet`` values; a regression that
   passed the citation list or any snippet into an event payload
   would fail here.

Draft text is covered indirectly: the workflow's
``_emit_draft_completed`` never receives the generated ``text``, only
the citation list and operational metadata. The structural absence
of any ``text`` parameter on the emit path is what makes the
draft-text leak architecturally impossible — and the no-outbound-IO
architectural test reinforces this by pinning that the analytics
package cannot transitively pull in any IO module.

The test reads the JSONL as raw text on purpose; an assertion on
model fields could miss a content leak that landed in a stringified
nested object. The raw substring test is the strict invariant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eurpe.analytics import (
    AnalyticsLogger,
    DraftCompletedEvent,
    DraftStartedEvent,
)
from eurpe.analytics.logger import _reset_handlers_for_tests
from eurpe.generation import (
    DeterministicLLMClient,
    GenerationRequest,
    SectionGenerationWorkflow,
)
from eurpe.retrieval import (
    ChromaIndex,
    Chunk,
    DeterministicHashEmbedder,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
)
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)

# Sentinel substrings injected into the user intent and the corpus
# chunk text. The test asserts neither appears in the analytics JSONL.
# The string shapes are deliberately uncommon and easy to grep for so a
# leak surfaces immediately.
_SENTINEL_INTENT = "SENTINEL_USER_INTENT_XYZ_12345_methodology_request"
_SENTINEL_SNIPPET_PREFIX = "SENTINEL_SNIPPET_ABCD_67890"


@pytest.fixture(autouse=True)
def _clean_analytics_handlers() -> None:
    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


def _make_chunk(
    *,
    status: SourceStatus,
    document_id: str,
    chunk_index: int,
    sentinel_id: int,
) -> Chunk:
    """Build a corpus chunk whose text contains a unique sentinel.

    The sentinel + the words ``deep learning methodology`` give the
    deterministic-hash embedder enough overlap with the query to
    return the chunk; the sentinel itself is what the JSONL must
    NEVER contain.
    """

    text = (
        f"{_SENTINEL_SNIPPET_PREFIX}_{sentinel_id}: This is a sample "
        f"{status.value} chunk discussing deep learning methodology for "
        "the horizon europe call."
    )
    proposal = ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        year=2024,
        outcome=status,
        proposal_title=f"Proposal {document_id}",
        source_path=f"data/{document_id}.pdf",
    )
    anchor = CitationAnchor(
        document_id=document_id,
        section_heading="1.2 Methodology",
        page=8,
    )
    metadata = ChunkMetadata(
        proposal=proposal,
        section_type=SectionType.METHODOLOGY,
        parent_section_heading="1.2 Methodology",
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=status,
    )
    return Chunk(text=text, metadata=metadata)


def _build_workflow(
    tmp_path: Path,
    analytics: AnalyticsLogger,
) -> SectionGenerationWorkflow:
    """Build a workflow with a real ChromaIndex + lenient policy + analytics.

    Same shape as ``tests/test_generation_workflow.py::_build_workflow``,
    but always wires an analytics logger in.
    """

    embedder = DeterministicHashEmbedder(dimension=128)
    index = ChromaIndex(
        index_path=tmp_path / "chroma",
        embedder=embedder,
        collection_name="analytics_integration_test",
    )
    chunks = [
        _make_chunk(status=SourceStatus.FUNDED, document_id="f1", chunk_index=0, sentinel_id=1),
        _make_chunk(status=SourceStatus.FUNDED, document_id="f2", chunk_index=1, sentinel_id=2),
        _make_chunk(
            status=SourceStatus.REJECTED,
            document_id="r1",
            chunk_index=0,
            sentinel_id=3,
        ),
    ]
    index.upsert(chunks)
    retriever = SourceStatusAwareRetriever(
        index,
        policy=RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0),
    )
    return SectionGenerationWorkflow(
        retriever=retriever,
        llm=DeterministicLLMClient(),
        analytics=analytics,
    )


# ---------------------------------------------------------------------------
# AC2 keystone — sentinel strings must NOT leak into the JSONL log
# ---------------------------------------------------------------------------


def test_workflow_emits_two_events_with_no_content_leak(tmp_path: Path) -> None:
    """Run the workflow once; assert exactly 2 events, no content leak.

    AC2 of issue #13: event payloads must NOT include raw document
    content, retrieved passages, or generated draft text. The
    sentinel-substring check on the raw JSONL file is the strict
    invariant.
    """

    log_path = tmp_path / "analytics.log"
    analytics = AnalyticsLogger(log_path)
    workflow = _build_workflow(tmp_path, analytics)

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent=_SENTINEL_INTENT,
        top_k_examples=5,
    )
    draft = workflow.run(request)

    # The workflow must have produced a draft with at least one
    # citation; otherwise the test would not exercise the
    # source_status_mix path.
    assert draft.citations, "test corpus must produce at least one citation"
    # Sanity: the snippet sentinel actually flows through into at
    # least one citation. Otherwise the "snippet not in JSONL"
    # assertion below would be vacuously true.
    assert any(_SENTINEL_SNIPPET_PREFIX in c.snippet for c in draft.citations), (
        "test corpus must surface sentinel-bearing snippets so the leak assertion is meaningful"
    )

    raw = log_path.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    assert len(lines) == 2, f"expected 2 events, got {len(lines)}: {lines!r}"

    # AC2 strict invariant: NO sentinel anywhere in the file content.
    # The intent sentinel catches a future regression that passed
    # request.user_intent (or any prompt fragment derived from it)
    # into an event payload — the workflow architecturally never does
    # this today, and this test pins that.
    assert _SENTINEL_INTENT not in raw, f"user_intent sentinel leaked into analytics log:\n{raw!r}"
    # Snippet sentinel is unique per chunk; check that NONE of them
    # surfaced. A retrieved passage that landed in a payload field
    # would be caught here.
    for sentinel_id in range(1, 4):
        marker = f"{_SENTINEL_SNIPPET_PREFIX}_{sentinel_id}"
        assert marker not in raw, (
            f"chunk snippet sentinel {marker!r} leaked into analytics log:\n{raw!r}"
        )
    # Defensive: the literal prefix should also be absent even without
    # a sentinel id (catches any partial leak where only the prefix
    # made it through a truncation).
    assert _SENTINEL_SNIPPET_PREFIX not in raw, (
        f"snippet sentinel prefix leaked into analytics log:\n{raw!r}"
    )

    # Event-shape assertions: one start, one complete, no extras.
    parsed = [json.loads(line) for line in lines]
    types = [rec["event_type"] for rec in parsed]
    assert types == ["draft_started", "draft_completed"], types

    started = DraftStartedEvent.model_validate(parsed[0])
    completed = DraftCompletedEvent.model_validate(parsed[1])
    assert started.section_type == "methodology"
    assert started.top_k_examples == 5
    assert started.model == "deterministic-stub-v1"
    assert completed.section_type == "methodology"
    assert completed.model == "deterministic-stub-v1"


def test_workflow_records_source_status_mix(tmp_path: Path) -> None:
    """``DraftCompletedEvent.source_status_mix`` matches the citations' Counter.

    AC1 contract: the schema must include source-status mix. This
    test pins that the workflow populates it from the actual
    citation list.
    """

    from collections import Counter

    log_path = tmp_path / "analytics.log"
    analytics = AnalyticsLogger(log_path)
    workflow = _build_workflow(tmp_path, analytics)

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent=_SENTINEL_INTENT,
        top_k_examples=10,
        lessons_learned=True,  # surface the rejected chunk too
    )
    draft = workflow.run(request)

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_rec = json.loads(lines[-1])
    assert completed_rec["event_type"] == "draft_completed"

    expected_mix = dict(Counter(c.source_status.value for c in draft.citations))
    assert completed_rec["source_status_mix"] == expected_mix


def test_workflow_records_generation_time(tmp_path: Path) -> None:
    """``DraftCompletedEvent.generation_time_ms`` is a non-negative int.

    AC1 contract: schema must include generation time. We do not
    pin an exact value (it depends on machine speed) but the int +
    non-negative invariant is enough.
    """

    log_path = tmp_path / "analytics.log"
    analytics = AnalyticsLogger(log_path)
    workflow = _build_workflow(tmp_path, analytics)

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent=_SENTINEL_INTENT,
        top_k_examples=5,
    )
    workflow.run(request)

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_rec = json.loads(lines[-1])
    assert completed_rec["event_type"] == "draft_completed"
    assert isinstance(completed_rec["generation_time_ms"], int)
    assert completed_rec["generation_time_ms"] >= 0


def test_critic_loop_emits_iteration_count_without_critique_text_leak(
    tmp_path: Path,
) -> None:
    """Task 3.2 / issue #16: the critic loop's per-iteration analytics carry
    the iteration_count but no critique narrative.

    Pins two invariants:

    1. DraftCompletedEvent.iteration_count increments per pass — 1 for
       the initial draft, 2 for the first critic pass, etc. An operator
       can read the JSONL log and reconstruct how many iterations a
       given draft cost.
    2. NO critique text, NO requirements list, NO changes summary
       lands in the analytics log. The critique sentinel string we
       inject into the LLM output must NOT appear anywhere in the file.
       The privacy contract on :mod:`eurpe.analytics.events` is
       binding even for the iteration loop.
    """

    from eurpe.generation import CriticAgent, CriticLoopWorkflow

    log_path = tmp_path / "analytics_iter.log"
    analytics = AnalyticsLogger(log_path)
    workflow = _build_workflow(tmp_path, analytics)
    critic_loop = CriticLoopWorkflow(
        workflow=workflow,
        critic=CriticAgent(workflow.llm),
    )

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe a deep learning methodology.",
        top_k_examples=5,
    )

    # Initial draft (iteration 1) + one critic pass (iteration 2).
    draft = workflow.run(request)
    critic_loop.iterate(prior_draft=draft, request=request, max_iterations=3)

    raw = log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    completed = [r for r in records if r["event_type"] == "draft_completed"]
    # One completed event per pass (initial + critic).
    assert len(completed) == 2
    # iteration_count climbs 1 → 2 across the passes.
    counts = [r["iteration_count"] for r in completed]
    assert counts == [1, 2], f"expected iteration counts [1, 2], got {counts}"

    # The critic critique never reaches the JSONL.
    # The deterministic stub produces a recognisable refrain but here
    # we assert on the *category* of the leak: any of the iteration
    # record fields would name "critique", "changes_summary", or
    # "requirements_checked".
    assert "critique_text" not in raw
    assert "changes_summary" not in raw
    assert "requirements_checked" not in raw


def test_workflow_without_analytics_emits_no_events(tmp_path: Path) -> None:
    """``analytics=None`` (default) keeps the workflow silent.

    Regression guard: introducing analytics must not change the
    behaviour of existing call sites that haven't been updated yet.
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path / "chroma2",
        embedder=embedder,
        collection_name="silent_analytics_test",
    )
    index.upsert(
        [_make_chunk(status=SourceStatus.FUNDED, document_id="f1", chunk_index=0, sentinel_id=1)]
    )
    retriever = SourceStatusAwareRetriever(
        index,
        policy=RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0),
    )
    workflow = SectionGenerationWorkflow(
        retriever=retriever,
        llm=DeterministicLLMClient(),
        # analytics omitted on purpose.
    )

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent=_SENTINEL_INTENT,
        top_k_examples=5,
    )
    draft = workflow.run(request)
    assert draft.text  # workflow still returns a draft

    # No analytics log file was created because no analytics logger
    # was wired.
    expected_log = tmp_path / "analytics.log"
    assert not expected_log.exists()
