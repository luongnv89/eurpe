"""Tests for ``eurpe.generation.workflow`` — the section-generation pipeline.

The workflow is the public entry point that ties retrieval + prompt +
LLM together. Tests here cover:

* AC1: works for any :class:`SectionType` (Methodology and Impact
  Pathway both exercised explicitly).
* AC2: every returned :class:`GenerationDraft` carries citations with
  source-status labels; hallucinated ``[N]`` markers raise
  :class:`GenerationError`.
* AC3 keystone: the whole pipeline runs under the ``no_network``
  fixture (no socket connections).

The tests use a real :class:`ChromaIndex` populated with fabricated
chunks (mix of funded + rejected) and the deterministic LLM stub.
This is the same pattern ``test_retriever.py`` uses for its
integration tests.
"""

from __future__ import annotations

import pytest

from eurpe.generation import (
    DeterministicLLMClient,
    GenerationDraft,
    GenerationError,
    GenerationRequest,
    SectionGenerationWorkflow,
)
from eurpe.generation.llm import LLMClient
from eurpe.generation.workflow import _scan_citation_markers
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

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    status: SourceStatus,
    programme: Programme = Programme.HORIZON_EUROPE,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    section_type: SectionType = SectionType.METHODOLOGY,
    document_id: str = "doc",
    chunk_index: int = 0,
    text: str | None = None,
) -> Chunk:
    if text is None:
        text = (
            f"This is a sample {status.value} chunk discussing methodology "
            f"deep learning approach for the {programme.value} call."
        )
    proposal = ProposalMetadata(
        programme=programme,
        call_id=call_id,
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
        section_type=section_type,
        parent_section_heading="1.2 Methodology",
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=status,
    )
    return Chunk(text=text, metadata=metadata)


def _build_corpus() -> list[Chunk]:
    """Build a small mixed-status corpus for workflow integration tests.

    3 funded + 2 rejected chunks, all in the methodology section, all
    talking about deep learning so the deterministic-hash embedder
    finds them on a "deep learning" query.
    """

    return [
        _make_chunk(status=SourceStatus.FUNDED, document_id="f1", chunk_index=0),
        _make_chunk(status=SourceStatus.FUNDED, document_id="f2", chunk_index=1),
        _make_chunk(status=SourceStatus.FUNDED, document_id="f3", chunk_index=2),
        _make_chunk(
            status=SourceStatus.REJECTED,
            programme=Programme.HORIZON_2020,
            call_id="H2020-X-2018-1",
            document_id="r1",
            chunk_index=0,
        ),
        _make_chunk(
            status=SourceStatus.REJECTED,
            programme=Programme.HORIZON_2020,
            call_id="H2020-X-2018-1",
            document_id="r2",
            chunk_index=1,
        ),
    ]


def _build_workflow(tmp_path, llm: LLMClient | None = None) -> SectionGenerationWorkflow:  # type: ignore[no-untyped-def]
    """Build a workflow with a real ChromaIndex + lenient policy + deterministic LLM.

    The lenient policy (``relevance_threshold=0.0``,
    ``max_rejected_fraction=1.0``) is critical: the
    :class:`DeterministicHashEmbedder` produces modest cosine scores
    that sit below the default 0.30 threshold, so without the override
    the workflow returns zero citations. ``test_retrieval_cli.py``
    uses the same trick.
    """

    embedder = DeterministicHashEmbedder(dimension=128)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="generation_workflow_test",
    )
    index.upsert(_build_corpus())
    policy = RetrievalPolicy(
        relevance_threshold=0.0,
        max_rejected_fraction=1.0,
    )
    retriever = SourceStatusAwareRetriever(index, policy=policy)
    return SectionGenerationWorkflow(
        retriever=retriever,
        llm=llm or DeterministicLLMClient(),
    )


def _basic_request(
    section_type: SectionType = SectionType.METHODOLOGY,
) -> GenerationRequest:
    return GenerationRequest(
        section_type=section_type,
        user_intent="Describe our deep learning approach for methodology",
        top_k_examples=5,
    )


# ---------------------------------------------------------------------------
# AC1 + AC2: workflow produces drafts with source-status-labelled citations
# ---------------------------------------------------------------------------


def test_run_produces_draft_with_citations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    workflow = _build_workflow(tmp_path)
    draft = workflow.run(_basic_request())

    assert isinstance(draft, GenerationDraft)
    assert draft.text  # non-empty
    assert draft.citations  # non-empty (corpus has matching chunks)
    assert draft.section_type is SectionType.METHODOLOGY
    assert draft.model == "deterministic-stub-v1"


def test_run_with_impact_pathway_section_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """AC1: any SectionType works — Impact Pathway is the second-most likely first use."""

    # Seed the corpus with at least one impact-pathway chunk so the
    # section_type filter doesn't return empty.
    embedder = DeterministicHashEmbedder(dimension=128)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="generation_impact_pathway_test",
    )
    impact_chunk = _make_chunk(
        status=SourceStatus.FUNDED,
        document_id="impact_funded",
        section_type=SectionType.IMPACT_PATHWAY,
        text=(
            "Impact pathway chunk: outputs to outcomes to wider impacts "
            "in cyber-physical systems."
        ),
    )
    index.upsert([impact_chunk])
    retriever = SourceStatusAwareRetriever(
        index,
        policy=RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0),
    )
    workflow = SectionGenerationWorkflow(retriever=retriever, llm=DeterministicLLMClient())

    request = GenerationRequest(
        section_type=SectionType.IMPACT_PATHWAY,
        user_intent="Articulate the impact pathway for cyber-physical systems",
    )
    draft = workflow.run(request)

    assert draft.section_type is SectionType.IMPACT_PATHWAY
    assert draft.citations
    # The prompt must record the section type so the renderer can title the draft.
    assert "Impact Pathway" in draft.prompt_used


def test_citations_carry_source_status_labels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """AC2: every CitationRef has a source_status; at least one is FUNDED."""

    workflow = _build_workflow(tmp_path)
    draft = workflow.run(_basic_request())

    assert draft.citations
    statuses = [c.source_status for c in draft.citations]
    assert all(s is not None for s in statuses)
    assert SourceStatus.FUNDED in statuses


def test_citation_ids_are_one_indexed_and_consecutive(tmp_path) -> None:  # type: ignore[no-untyped-def]
    workflow = _build_workflow(tmp_path)
    draft = workflow.run(_basic_request())

    ids = [c.citation_id for c in draft.citations]
    assert ids == list(range(1, len(ids) + 1))


def test_draft_echoes_request_for_traceability(tmp_path) -> None:  # type: ignore[no-untyped-def]
    workflow = _build_workflow(tmp_path)
    request = _basic_request()
    draft = workflow.run(request)

    assert draft.request.section_type is request.section_type
    assert draft.request.user_intent == request.user_intent
    assert draft.request.top_k_examples == request.top_k_examples


def test_draft_records_full_prompt_for_audit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    workflow = _build_workflow(tmp_path)
    draft = workflow.run(_basic_request())

    assert draft.prompt_used
    # The user intent should appear verbatim in the prompt — it's the
    # one field the workflow forwards into the prompt builder.
    assert "deep learning approach for methodology" in draft.prompt_used


# ---------------------------------------------------------------------------
# AC3 keystone: end-to-end under no_network
# ---------------------------------------------------------------------------


def test_offline_end_to_end_under_no_network_fixture(  # type: ignore[no-untyped-def]
    tmp_path,
    no_network: None,
) -> None:
    """AC3 keystone: full workflow runs without any socket.connect calls.

    Builds a real ChromaIndex (deterministic embedder, no network),
    seeds it with chunks, runs the workflow with the
    DeterministicLLMClient, and asserts a valid draft comes out. If
    *any* code path here opens a socket, the ``no_network`` fixture
    raises ``pytest.fail.Exception`` — so a passing test is positive
    proof that AC3 holds end-to-end.
    """

    workflow = _build_workflow(tmp_path)
    draft = workflow.run(_basic_request())

    # Same shape assertions as the happy path, but the *fact that the
    # test passed* is the AC3 evidence. Keep the assertions so a
    # change that produces an empty draft still fails loudly.
    assert isinstance(draft, GenerationDraft)
    assert draft.text
    assert draft.citations
    assert draft.model == "deterministic-stub-v1"


# ---------------------------------------------------------------------------
# Lessons-learned mode
# ---------------------------------------------------------------------------


def test_run_with_lessons_learned_propagates_to_retriever(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``lessons_learned=True`` on the request is forwarded to the retriever.

    Verified observably: under lessons-learned mode, rejected
    citations are present (the rejected-fraction cap is bypassed).
    """

    workflow = _build_workflow(tmp_path)
    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our deep learning approach for methodology",
        top_k_examples=10,
        lessons_learned=True,
    )
    draft = workflow.run(request)

    statuses = [c.source_status for c in draft.citations]
    # The corpus has 2 rejected chunks; under lessons_learned mode at
    # least one should appear in the citations.
    assert SourceStatus.REJECTED in statuses, (
        f"Expected at least one rejected citation under lessons_learned=True; "
        f"got {statuses}"
    )


# ---------------------------------------------------------------------------
# Edge: empty index
# ---------------------------------------------------------------------------


def test_run_with_no_results_still_returns_draft_with_warning(  # type: ignore[no-untyped-def]
    tmp_path,
    caplog,
) -> None:
    """An empty index → workflow still returns a draft (no exception).

    The deterministic stub emits an explicit "No retrieved evidence"
    sentence in this case so the user can tell what happened. The
    workflow logs a warning about missing ``[N]`` markers (the stub
    omits them when there are no citations to reflect).
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="empty_index_test",
    )
    # Don't upsert anything.
    retriever = SourceStatusAwareRetriever(
        index,
        policy=RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0),
    )
    workflow = SectionGenerationWorkflow(retriever=retriever, llm=DeterministicLLMClient())

    with caplog.at_level("WARNING", logger="eurpe.generation.workflow"):
        draft = workflow.run(_basic_request())

    assert draft.citations == []
    assert draft.text  # non-empty fallback text from the stub
    # Warning was logged because there are no markers in the draft.
    assert any("no [N] citation markers" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Citation-marker validation (AC2 hallucination guard)
# ---------------------------------------------------------------------------


class _HallucinatingLLM:
    """LLM stub that emits a draft with an out-of-range citation marker.

    Used to verify that the workflow's :meth:`_validate_citations`
    raises on hallucinated markers and names which marker is bad.
    """

    @property
    def model(self) -> str:
        return "hallucinating-stub"

    def generate(
        self,
        prompt: str,  # noqa: ARG002
        *,
        max_tokens: int = 1024,  # noqa: ARG002
        temperature: float = 0.2,  # noqa: ARG002
    ) -> str:
        # Hallucinate citation [99] which can't possibly be valid for
        # any reasonable top_k.
        return "Drafted text that references [1] and the bogus [99]."


def test_run_validates_citation_markers_and_names_offender(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Hallucinated [99] → :class:`GenerationError` mentioning the marker.

    The error message MUST include the offending marker (per the
    advisor's note) so an operator can debug without re-running with
    a debugger attached.
    """

    workflow = _build_workflow(tmp_path, llm=_HallucinatingLLM())
    with pytest.raises(GenerationError, match=r"\[99\]"):
        workflow.run(_basic_request())


def test_validate_citations_no_markers_logs_warning(caplog) -> None:  # type: ignore[no-untyped-def]
    """Zero ``[N]`` markers in the draft → warning, not error."""

    from eurpe.generation.models import CitationRef

    citations = [
        CitationRef(
            citation_id=1,
            source_status=SourceStatus.FUNDED,
            programme=Programme.HORIZON_EUROPE,
            call_id="X",
            chunk_id="d::a::000000",
            snippet="...",
        )
    ]
    with caplog.at_level("WARNING", logger="eurpe.generation.workflow"):
        # Static method — invoked directly with a citation list and
        # marker-free text. Asserts no exception is raised.
        SectionGenerationWorkflow._validate_citations(
            text="Plain prose without any markers.",
            citations=citations,
        )
    assert any("no [N] citation markers" in rec.message for rec in caplog.records)


def test_scan_citation_markers_finds_all_in_order() -> None:
    """Helper returns markers in document order, including duplicates."""

    text = "First [1], then [2], then [1] again, and [3]."
    assert _scan_citation_markers(text) == [1, 2, 1, 3]


def test_scan_citation_markers_returns_empty_on_no_match() -> None:
    assert _scan_citation_markers("no citations here at all") == []


def test_validate_citations_accepts_in_range_markers() -> None:
    """All-valid markers → no exception, no warning required."""

    from eurpe.generation.models import CitationRef

    citations = [
        CitationRef(
            citation_id=i,
            source_status=SourceStatus.FUNDED,
            programme=Programme.HORIZON_EUROPE,
            call_id="X",
            chunk_id=f"d::a::00000{i}",
            snippet="...",
        )
        for i in range(1, 4)
    ]
    SectionGenerationWorkflow._validate_citations(
        text="Used [1], [2], and [3].",
        citations=citations,
    )


# ---------------------------------------------------------------------------
# SectionType.OTHER
# ---------------------------------------------------------------------------


def test_section_type_other_works(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """SectionType.OTHER still produces a draft (uses the OTHER guidance)."""

    # Seed the corpus with an OTHER-typed chunk so the section_type filter
    # has something to match.
    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="other_section_test",
    )
    other_chunk = _make_chunk(
        status=SourceStatus.FUNDED,
        document_id="o1",
        section_type=SectionType.OTHER,
        text="Other-section content with general guidance.",
    )
    index.upsert([other_chunk])
    retriever = SourceStatusAwareRetriever(
        index,
        policy=RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0),
    )
    workflow = SectionGenerationWorkflow(retriever=retriever, llm=DeterministicLLMClient())

    request = GenerationRequest(
        section_type=SectionType.OTHER,
        user_intent="Provide other content with general guidance",
    )
    draft = workflow.run(request)

    assert draft.section_type is SectionType.OTHER
    assert draft.citations


# ---------------------------------------------------------------------------
# Read-only properties
# ---------------------------------------------------------------------------


def test_workflow_exposes_read_only_llm_and_retriever(tmp_path) -> None:  # type: ignore[no-untyped-def]
    workflow = _build_workflow(tmp_path)
    assert workflow.llm.model == "deterministic-stub-v1"
    assert isinstance(workflow.retriever, SourceStatusAwareRetriever)
