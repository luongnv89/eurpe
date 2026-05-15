"""Tests for ``eurpe.retrieval.retriever`` — source-status-aware policy.

Two tiers of tests live here:

* **Policy-mechanics tests** drive the retriever against a stub index
  (:class:`_StubIndex`) that returns hardcoded ``(Chunk, score)`` pairs.
  This isolates the policy from the embedder so a test can engineer
  exact score relationships (ties, near-ties, threshold crossings) that
  are hard to coax out of the deterministic-hash embedder.
* **Integration tests** drive the retriever against a real
  :class:`ChromaIndex` populated with the fixture chunks. These prove
  the AC1 invariant that source-status labels survive the round trip
  through Chroma, exercise filter forwarding, and confirm the offline
  contract under :func:`tests.conftest.no_network`.

Issue #5 acceptance criteria mapped to tests:

* **AC1** ("source-status labels attached to every result") — every test
  reads ``result.source_status``; the integration round-trip test
  asserts that the label survives the flatten/inflate path.
* **AC2** ("rejected examples must satisfy the same topical relevance
  threshold ... unless a lessons-learned flag is enabled") — covered by
  the keystone test
  :func:`test_lessons_learned_relaxes_rejected_threshold_same_chunk_crosses`.
  That test demonstrates the *same chunk* being excluded under default
  policy and accepted under lessons-learned mode.
* **AC3** ("funded-only, rejected-only, mixed-status, and no-match
  retrieval scenarios") — one test per scenario, plus the threshold
  edge cases.
"""

from __future__ import annotations

import pytest

from eurpe.retrieval import (
    ChromaIndex,
    Chunk,
    DeterministicHashEmbedder,
    RetrievalPolicy,
    RetrievalResult,
    SourceStatusAwareRetriever,
)
from eurpe.retrieval.retriever import (
    POLICY_REASON_ESR,
    POLICY_REASON_FUNDED,
    POLICY_REASON_LESSONS_LEARNED,
    POLICY_REASON_REJECTED,
    POLICY_REASON_SECTION_TYPE_FALLBACK_SUFFIX,
    POLICY_REASON_UNKNOWN,
)
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)
from tests._chunk_helpers import build_fixture_chunks, query_text_for

# ---------------------------------------------------------------------------
# Stub index — used by policy-mechanics tests
# ---------------------------------------------------------------------------


class _StubIndex:
    """Minimal stand-in for :class:`ChromaIndex` for policy unit tests.

    Returns the configured ``(chunk, score)`` rows up to ``top_k``,
    optionally filtering by a single ``where`` key/value pair (the only
    shape the retriever uses for the source_status passthrough). Records
    the ``query`` calls so tests can assert on the over-fetch behaviour
    and on the ``where`` clause that the retriever built.
    """

    def __init__(self, rows: list[tuple[Chunk, float]]) -> None:
        # Stored pre-sorted by descending score so the stub mirrors what
        # the real ChromaIndex returns (index orders by ascending distance
        # which is descending similarity).
        self._rows = sorted(rows, key=lambda r: -r[1])
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        where: dict[str, object] | None = None,
    ) -> list[tuple[Chunk, float]]:
        self.calls.append({"query_text": query_text, "top_k": top_k, "where": where})

        rows = self._rows
        # The retriever only uses single-key wheres for source_status /
        # programme / section_type, or an ``$and`` of those. The stub
        # implements just enough filtering to make those filter-propagation
        # tests meaningful; anything more elaborate is out of scope.
        if where is not None:
            rows = [r for r in rows if _row_matches_where(r[0], where)]

        return rows[:top_k]


def _row_matches_where(chunk: Chunk, where: dict[str, object]) -> bool:
    """Tiny ``where`` evaluator — supports flat ``{key: value}`` and ``{$and: [...]}``."""

    if "$and" in where:
        clauses = where["$and"]
        if not isinstance(clauses, list):  # pragma: no cover - defensive
            return False
        return all(_row_matches_where(chunk, c) for c in clauses)
    for key, expected in where.items():
        actual = _flat_metadata_lookup(chunk, key)
        if actual != expected:
            return False
    return True


def _flat_metadata_lookup(chunk: Chunk, key: str) -> object:
    """Mirror the dotted-key flatten convention used by ``index._metadata_to_chroma``."""

    meta = chunk.metadata
    if key == "source_status":
        return meta.source_status.value
    if key == "section_type":
        return meta.section_type.value
    if key == "proposal.programme":
        return meta.proposal.programme.value
    if key == "proposal.call_id":
        return meta.proposal.call_id
    raise KeyError(f"_flat_metadata_lookup: unknown key {key!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Chunk fabrication helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    status: SourceStatus,
    programme: Programme = Programme.HORIZON_EUROPE,
    call_id: str = "HORIZON-CL5-2024-D3-02",
    section_type: SectionType = SectionType.METHODOLOGY,
    document_id: str = "doc",
    chunk_index: int = 0,
    text: str = "stub chunk text",
) -> Chunk:
    """Build a Chunk with the requested status and just enough metadata."""

    proposal = ProposalMetadata(
        programme=programme,
        call_id=call_id,
        year=2024,
        outcome=status,
        source_path=f"data/{document_id}.pdf",
    )
    anchor = CitationAnchor(document_id=document_id)
    metadata = ChunkMetadata(
        proposal=proposal,
        section_type=section_type,
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=status,
    )
    return Chunk(text=text, metadata=metadata)


def _funded(idx: int = 0, doc: str = "funded_doc") -> Chunk:
    return _make_chunk(status=SourceStatus.FUNDED, document_id=doc, chunk_index=idx)


def _rejected(idx: int = 0, doc: str = "rejected_doc") -> Chunk:
    return _make_chunk(
        status=SourceStatus.REJECTED,
        programme=Programme.HORIZON_2020,
        call_id="H2020-ICT-2018-2",
        section_type=SectionType.IMPACT,
        document_id=doc,
        chunk_index=idx,
    )


def _esr(idx: int = 0, doc: str = "esr_doc") -> Chunk:
    return _make_chunk(
        status=SourceStatus.ESR_NOTE,
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL3-2023-CS-01",
        section_type=SectionType.EXCELLENCE,
        document_id=doc,
        chunk_index=idx,
    )


def _unknown(idx: int = 0, doc: str = "unknown_doc") -> Chunk:
    return _make_chunk(
        status=SourceStatus.UNKNOWN,
        programme=Programme.OTHER,
        call_id="NATIONAL-FR-2022-CYBER",
        section_type=SectionType.OTHER,
        document_id=doc,
        chunk_index=idx,
    )


# ---------------------------------------------------------------------------
# Policy-mechanics tests (stubbed index)
# ---------------------------------------------------------------------------


def test_default_returns_funded_chunks_with_source_status_label() -> None:
    """AC1: every result carries a non-None source_status, and at least one is FUNDED."""

    rows = [
        (_funded(), 0.9),
        (_rejected(), 0.7),
        (_esr(), 0.6),
    ]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("query", top_k=3)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.source_status is not None  # StrEnum members are truthy strings
    assert any(r.source_status is SourceStatus.FUNDED for r in results)


def test_default_funded_first_at_similar_scores() -> None:
    """When funded and rejected have equal scores, funded outranks rejected.

    Disable the rejected-fraction cap so this test isolates the ordering
    behaviour — the cap (default 40%, which at top_k=2 floors to 0) would
    otherwise skip the rejected chunk for an unrelated reason.
    """

    rows = [
        (_funded(idx=0, doc="f1"), 0.80),
        (_rejected(idx=0, doc="r1"), 0.80),  # exactly equal score
    ]
    policy = RetrievalPolicy(max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=2)

    assert results[0].source_status is SourceStatus.FUNDED
    assert results[1].source_status is SourceStatus.REJECTED


def test_default_rejected_below_threshold_is_dropped() -> None:
    """A rejected chunk with score below relevance_threshold is excluded."""

    rows = [
        (_funded(), 0.90),
        (_rejected(), 0.20),  # below the 0.30 default threshold
    ]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("query", top_k=5)

    assert len(results) == 1
    assert results[0].source_status is SourceStatus.FUNDED
    assert all(r.source_status is not SourceStatus.REJECTED for r in results)


def test_default_max_rejected_fraction_caps_rejected() -> None:
    """``max_rejected_fraction=0.25`` + top_k=4 → at most 1 REJECTED result."""

    # All scores comfortably above the threshold.
    rows = [
        (_funded(idx=0, doc="f1"), 0.91),
        (_funded(idx=1, doc="f2"), 0.90),
        (_funded(idx=2, doc="f3"), 0.89),
        (_rejected(idx=0, doc="r1"), 0.88),
        (_rejected(idx=1, doc="r2"), 0.87),
        (_rejected(idx=2, doc="r3"), 0.86),
        (_rejected(idx=3, doc="r4"), 0.85),
    ]
    policy = RetrievalPolicy(max_rejected_fraction=0.25)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=4)

    rejected_count = sum(1 for r in results if r.source_status is SourceStatus.REJECTED)
    # floor(4 * 0.25) = 1
    assert rejected_count <= 1
    assert len(results) == 4  # the cap should skip rejected, not shorten the list


def test_policy_reason_set_correctly_for_each_status() -> None:
    """Each status maps to its canonical policy_reason string."""

    rows = [
        (_funded(), 0.95),
        (_rejected(), 0.85),
        (_esr(), 0.75),
        (_unknown(), 0.65),
    ]
    # Disable funded_first so all four come through in score order — easier
    # to assert against without juggling status priority.
    policy = RetrievalPolicy(funded_first=False, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=4)

    by_status = {r.source_status: r.policy_reason for r in results}
    assert by_status[SourceStatus.FUNDED] == POLICY_REASON_FUNDED
    assert by_status[SourceStatus.REJECTED] == POLICY_REASON_REJECTED
    assert by_status[SourceStatus.ESR_NOTE] == POLICY_REASON_ESR
    assert by_status[SourceStatus.UNKNOWN] == POLICY_REASON_UNKNOWN


# --- Lessons-learned mode (AC2 keystone) ----------------------------------


def test_lessons_learned_relaxes_rejected_threshold_same_chunk_crosses() -> None:
    """AC2 keystone: the same rejected chunk is excluded by default and
    accepted under lessons-learned mode with a negative offset.

    The advisor's specific check: don't merely show that lessons_learned
    enables the mode — show that a chunk *which would be excluded* now
    appears as a result. That is the only honest demonstration of
    threshold relaxation.
    """

    borderline_rejected = _rejected(idx=0, doc="r_borderline")
    rows = [
        (_funded(), 0.90),
        (borderline_rejected, 0.25),  # below default threshold (0.30)
    ]
    stub = _StubIndex(rows)

    # Default policy — the borderline rejected MUST be excluded.
    default_results = SourceStatusAwareRetriever(stub, policy=RetrievalPolicy()).retrieve(
        "query", top_k=5
    )
    rejected_default = [r for r in default_results if r.source_status is SourceStatus.REJECTED]
    assert rejected_default == [], (
        "Borderline rejected chunk should be excluded under default policy "
        "(score 0.25 < threshold 0.30)"
    )

    # Lessons-learned mode with a -0.10 offset → effective rejected
    # threshold is 0.20, which lets the 0.25-score chunk through.
    lessons_policy = RetrievalPolicy(
        lessons_learned_mode=True,
        rejected_threshold_offset=-0.10,
    )
    lessons_results = SourceStatusAwareRetriever(stub, policy=lessons_policy).retrieve(
        "query", top_k=5
    )
    rejected_lessons = [r for r in lessons_results if r.source_status is SourceStatus.REJECTED]
    assert len(rejected_lessons) == 1, (
        "Borderline rejected chunk should now be included under lessons-learned "
        f"mode with offset -0.10. Got: {[(r.source_status, r.score) for r in lessons_results]}"
    )
    assert rejected_lessons[0].chunk.chunk_id == borderline_rejected.chunk_id
    assert rejected_lessons[0].policy_reason == POLICY_REASON_LESSONS_LEARNED


def test_lessons_learned_skips_rejected_fraction_cap() -> None:
    """Under lessons-learned mode, max_rejected_fraction is ignored.

    Configure max_rejected_fraction=0.0 (which would normally exclude all
    rejected) and verify rejected results still appear when
    lessons_learned_mode=True. This proves the cap is bypassed.
    """

    rows = [
        (_funded(idx=0, doc="f1"), 0.95),
        (_rejected(idx=0, doc="r1"), 0.90),
        (_rejected(idx=1, doc="r2"), 0.85),
    ]
    policy = RetrievalPolicy(
        lessons_learned_mode=True,
        max_rejected_fraction=0.0,  # would exclude all rejected without lessons-learned
    )
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=3)

    rejected_count = sum(1 for r in results if r.source_status is SourceStatus.REJECTED)
    assert rejected_count >= 1, (
        "Lessons-learned mode should skip the rejected-fraction cap; "
        f"got {rejected_count} rejected results"
    )


def test_per_call_lessons_learned_override() -> None:
    """``retrieve(lessons_learned=True)`` enables the mode for one call.

    Per the advisor: assert the *observable* effect (cap-skip and
    policy_reason flip), not the threshold relaxation — that's covered by
    the dedicated test above.
    """

    rows = [
        (_funded(idx=0, doc="f1"), 0.95),
        (_rejected(idx=0, doc="r1"), 0.90),
        (_rejected(idx=1, doc="r2"), 0.85),
    ]
    # Global policy: lessons_learned OFF, max_rejected_fraction=0.0 →
    # rejected normally excluded.
    policy = RetrievalPolicy(
        lessons_learned_mode=False,
        max_rejected_fraction=0.0,
    )
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)

    # Without override: cap excludes all rejected.
    default = retriever.retrieve("query", top_k=3)
    assert all(r.source_status is not SourceStatus.REJECTED for r in default)

    # With per-call override: cap is skipped; rejected appear with the
    # lessons_learned reason string.
    overridden = retriever.retrieve("query", top_k=3, lessons_learned=True)
    rejected = [r for r in overridden if r.source_status is SourceStatus.REJECTED]
    assert rejected, "per-call lessons_learned=True should re-admit rejected results"
    assert all(r.policy_reason == POLICY_REASON_LESSONS_LEARNED for r in rejected)


# --- Filters / hard scoping ------------------------------------------------


def test_filter_by_programme_propagates_to_index() -> None:
    """``programme=`` becomes a Chroma ``where`` clause on ``proposal.programme``."""

    rows = [
        (_funded(), 0.9),  # horizon_europe
        (_rejected(), 0.8),  # horizon_2020
    ]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(stub)
    results = retriever.retrieve("query", top_k=5, programme=Programme.HORIZON_EUROPE)

    assert stub.calls[-1]["where"] == {"proposal.programme": "horizon_europe"}
    for r in results:
        assert r.chunk.metadata.proposal.programme is Programme.HORIZON_EUROPE


def test_filter_by_section_type_propagates_to_index() -> None:
    """``section_type=`` becomes a Chroma ``where`` clause on ``section_type``."""

    rows = [
        (_funded(), 0.9),  # methodology
        (_rejected(), 0.8),  # impact
    ]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(stub)
    results = retriever.retrieve("query", top_k=5, section_type=SectionType.METHODOLOGY)

    assert stub.calls[-1]["where"] == {"section_type": "methodology"}
    for r in results:
        assert r.chunk.metadata.section_type is SectionType.METHODOLOGY


def test_filter_by_source_status_propagates_to_index() -> None:
    """``source_status=`` is forwarded as a hard server-side filter."""

    rows = [
        (_funded(), 0.9),
        (_rejected(), 0.8),
    ]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(stub)
    results = retriever.retrieve("query", top_k=5, source_status=SourceStatus.FUNDED)

    assert stub.calls[-1]["where"] == {"source_status": "funded"}
    assert all(r.source_status is SourceStatus.FUNDED for r in results)


def test_combined_filters_use_and_clause() -> None:
    """Multiple filters are composed under ``$and``."""

    rows = [(_funded(), 0.9)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(stub)
    retriever.retrieve(
        "query",
        top_k=5,
        programme=Programme.HORIZON_EUROPE,
        section_type=SectionType.METHODOLOGY,
    )

    where = stub.calls[-1]["where"]
    assert isinstance(where, dict)
    assert "$and" in where
    clauses = where["$and"]
    assert {"proposal.programme": "horizon_europe"} in clauses
    assert {"section_type": "methodology"} in clauses


def test_no_match_returns_empty() -> None:
    """When every candidate is below threshold, the result list is empty."""

    rows = [
        (_funded(), 0.10),  # below 0.30 default
        (_rejected(), 0.05),
    ]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("nothing matches", top_k=5)

    assert results == []


def test_no_match_with_high_threshold_returns_empty() -> None:
    """An extreme ``relevance_threshold`` still produces an empty list, not an error."""

    rows = [
        (_funded(), 0.5),
        (_rejected(), 0.4),
    ]
    policy = RetrievalPolicy(relevance_threshold=0.99)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=5)

    assert results == []


# --- Section_type fallback (Issue #46) ------------------------------------


def test_section_type_filter_empty_pool_falls_back_to_unfiltered() -> None:
    """Issue #46: ``section_type=methodology`` finding 0 candidates retries unfiltered.

    Reproduces the GEIGER gap from the issue: the chunker labelled every
    chunk with ``section_type=other`` (or non-methodology), so the
    ``index query`` CLI — which does not pass a section_type filter —
    returns hits, while ``generate section --type methodology`` empties
    the candidate pool server-side. With the fallback enabled, the same
    retriever call still returns the hits and tags them so an audit can
    trace that the filter was relaxed.
    """

    # Both rows are _rejected (impact) and _funded (methodology), but
    # we set the funded one's section_type to OTHER so a methodology
    # filter rejects everything on the first call.
    impact_funded = _make_chunk(
        status=SourceStatus.FUNDED,
        section_type=SectionType.OTHER,
        document_id="other_section",
    )
    rows = [(impact_funded, 0.5)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(
        stub,
        policy=RetrievalPolicy(relevance_threshold=0.0),
    )

    # Sanity: with the strict filter on (opt-out), the result list is empty.
    strict_results = retriever.retrieve(
        "query",
        top_k=5,
        section_type=SectionType.METHODOLOGY,
        enable_section_type_fallback=False,
    )
    assert strict_results == []

    # With the fallback enabled (the default), we get the row back.
    fallback_results = retriever.retrieve(
        "query",
        top_k=5,
        section_type=SectionType.METHODOLOGY,
    )
    assert len(fallback_results) == 1
    assert fallback_results[0].chunk is impact_funded
    # Policy reason carries the fallback suffix so audits can trace it.
    assert fallback_results[0].policy_reason.endswith(POLICY_REASON_SECTION_TYPE_FALLBACK_SUFFIX)
    # The underlying status reason still leads the string.
    assert fallback_results[0].policy_reason.startswith(POLICY_REASON_FUNDED)
    # Two index calls were made: the first with the section_type clause,
    # the second without.
    assert len(stub.calls) >= 2
    assert stub.calls[-2]["where"] == {"section_type": "methodology"}
    assert stub.calls[-1]["where"] is None


def test_section_type_fallback_does_not_fire_when_pool_nonempty() -> None:
    """Successful section_type-filtered call does NOT trigger a fallback retry."""

    rows = [(_funded(), 0.9)]  # _funded() defaults to methodology
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(stub)
    results = retriever.retrieve("query", top_k=5, section_type=SectionType.METHODOLOGY)

    # Exactly one index call — no fallback.
    assert len(stub.calls) == 1
    assert stub.calls[0]["where"] == {"section_type": "methodology"}
    # policy_reason carries no fallback suffix.
    assert "_section_type_fallback" not in results[0].policy_reason


def test_section_type_fallback_skipped_when_source_status_pinned() -> None:
    """A pinned ``source_status`` is a deliberate audience pick — never relax it."""

    # No funded methodology rows, but a rejected impact row exists.
    rows = [(_rejected(), 0.5)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(
        stub,
        policy=RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0),
    )

    results = retriever.retrieve(
        "query",
        top_k=5,
        section_type=SectionType.METHODOLOGY,
        source_status=SourceStatus.FUNDED,
    )

    # Empty — fallback would have surfaced a rejected chunk we MUST NOT
    # return when the caller pinned source_status=funded.
    assert results == []
    # Exactly one index call — fallback skipped.
    assert len(stub.calls) == 1


def test_section_type_fallback_disabled_in_policy() -> None:
    """``RetrievalPolicy(enable_section_type_fallback=False)`` preserves strict mode."""

    impact_chunk = _make_chunk(
        status=SourceStatus.FUNDED,
        section_type=SectionType.OTHER,
    )
    rows = [(impact_chunk, 0.5)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(
        stub,
        policy=RetrievalPolicy(
            relevance_threshold=0.0,
            enable_section_type_fallback=False,
        ),
    )

    results = retriever.retrieve("query", top_k=5, section_type=SectionType.METHODOLOGY)

    assert results == []
    assert len(stub.calls) == 1


def test_section_type_fallback_per_call_kwarg_overrides_policy() -> None:
    """Per-call ``enable_section_type_fallback=True`` wins over policy=False.

    Mirrors the policy-disabled fixture but flips the per-call kwarg to
    True, proving the override resolution at retriever.py works in both
    directions (the policy-default-True / per-call-False direction is
    covered by ``test_section_type_filter_empty_pool_falls_back_to_unfiltered``).
    """

    impact_chunk = _make_chunk(
        status=SourceStatus.FUNDED,
        section_type=SectionType.OTHER,
    )
    rows = [(impact_chunk, 0.5)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(
        stub,
        policy=RetrievalPolicy(
            relevance_threshold=0.0,
            enable_section_type_fallback=False,
        ),
    )

    results = retriever.retrieve(
        "query",
        top_k=5,
        section_type=SectionType.METHODOLOGY,
        enable_section_type_fallback=True,
    )

    assert len(results) == 1
    assert results[0].chunk is impact_chunk
    assert results[0].policy_reason.endswith(POLICY_REASON_SECTION_TYPE_FALLBACK_SUFFIX)
    assert len(stub.calls) == 2


def test_section_type_fallback_preserves_programme_filter() -> None:
    """Fallback drops section_type but keeps programme intact.

    The user's programme pin (HORIZON_EUROPE) is a deliberate topical
    scope decision — even when we relax section_type, we must NOT
    cross-pollinate with chunks from other programmes.
    """

    he_other = _make_chunk(
        status=SourceStatus.FUNDED,
        programme=Programme.HORIZON_EUROPE,
        section_type=SectionType.OTHER,
        document_id="he_other",
    )
    rows = [(he_other, 0.5)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(
        stub,
        policy=RetrievalPolicy(relevance_threshold=0.0),
    )

    results = retriever.retrieve(
        "query",
        top_k=5,
        section_type=SectionType.METHODOLOGY,
        programme=Programme.HORIZON_EUROPE,
    )

    assert len(results) == 1
    # First call had both filters (under $and); fallback call kept programme.
    first_where = stub.calls[0]["where"]
    assert isinstance(first_where, dict)
    assert "$and" in first_where
    second_where = stub.calls[-1]["where"]
    assert second_where == {"proposal.programme": "horizon_europe"}


# --- Corpus-shape scenarios -----------------------------------------------


def test_funded_only_corpus_returns_only_funded() -> None:
    rows = [(_funded(idx=i, doc=f"f{i}"), 0.95 - i * 0.01) for i in range(5)]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("query", top_k=3)

    assert len(results) == 3
    assert all(r.source_status is SourceStatus.FUNDED for r in results)


def test_rejected_only_corpus_capped_by_default() -> None:
    """With only-rejected corpus and the 40% default cap, ≤4 rejected at top_k=10."""

    rows = [(_rejected(idx=i, doc=f"r{i}"), 0.95 - i * 0.01) for i in range(10)]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("query", top_k=10)

    rejected_count = sum(1 for r in results if r.source_status is SourceStatus.REJECTED)
    # floor(10 * 0.4) = 4
    assert rejected_count <= 4


def test_rejected_only_corpus_under_lessons_learned_returns_more() -> None:
    """Lessons-learned mode bypasses the cap; more rejected results come back."""

    rows = [(_rejected(idx=i, doc=f"r{i}"), 0.95 - i * 0.01) for i in range(10)]
    policy = RetrievalPolicy(lessons_learned_mode=True)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=10)

    rejected_count = sum(1 for r in results if r.source_status is SourceStatus.REJECTED)
    assert rejected_count > 4  # more than the default cap would have allowed


def test_mixed_corpus_returns_blend_with_funded_priority() -> None:
    """A mixed corpus returns a blend; at near-tied scores funded leads."""

    rows = [
        (_funded(idx=0, doc="f1"), 0.90),
        (_funded(idx=1, doc="f2"), 0.85),
        (_rejected(idx=0, doc="r1"), 0.85),  # tied with f2
        (_esr(idx=0, doc="e1"), 0.80),
        (_unknown(idx=0, doc="u1"), 0.75),
    ]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("query", top_k=5)

    statuses = [r.source_status for r in results]
    # At least one of each kept-by-default status appears.
    assert SourceStatus.FUNDED in statuses
    assert SourceStatus.REJECTED in statuses
    assert SourceStatus.ESR_NOTE in statuses
    assert SourceStatus.UNKNOWN in statuses
    # Funded leads at near-tied scores: the score-0.85 funded chunk
    # outranks the score-0.85 rejected chunk.
    funded_85_rank = next(
        r.rank
        for r in results
        if r.source_status is SourceStatus.FUNDED and r.score == pytest.approx(0.85)
    )
    rejected_85_rank = next(
        r.rank
        for r in results
        if r.source_status is SourceStatus.REJECTED and r.score == pytest.approx(0.85)
    )
    assert funded_85_rank < rejected_85_rank


def test_no_esr_flag_excludes_esr() -> None:
    """``include_esr=False`` removes ESR notes even when otherwise eligible."""

    rows = [
        (_funded(), 0.9),
        (_esr(), 0.85),
    ]
    policy = RetrievalPolicy(include_esr=False)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=5)

    assert all(r.source_status is not SourceStatus.ESR_NOTE for r in results)
    assert any(r.source_status is SourceStatus.FUNDED for r in results)


def test_results_carry_chunk_metadata_source_status_label_intact() -> None:
    """``result.source_status`` and ``result.chunk.metadata.source_status`` agree."""

    rows = [
        (_funded(), 0.9),
        (_rejected(), 0.85),
        (_esr(), 0.8),
        (_unknown(), 0.75),
    ]
    retriever = SourceStatusAwareRetriever(
        _StubIndex(rows),
        policy=RetrievalPolicy(funded_first=False, max_rejected_fraction=1.0),
    )
    results = retriever.retrieve("query", top_k=4)

    for r in results:
        assert r.source_status == r.chunk.metadata.source_status


def test_top_k_respected() -> None:
    rows = [(_funded(idx=i, doc=f"f{i}"), 0.95 - i * 0.01) for i in range(10)]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    results = retriever.retrieve("query", top_k=3)

    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]


def test_top_k_zero_raises() -> None:
    retriever = SourceStatusAwareRetriever(_StubIndex([(_funded(), 0.9)]))
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=0)


def test_deterministic_ordering() -> None:
    """Same params → identical ``(chunk_id, rank)`` sequence across two calls."""

    rows = [
        (_funded(idx=0, doc="f1"), 0.90),
        (_funded(idx=1, doc="f2"), 0.90),  # same score
        (_funded(idx=2, doc="f3"), 0.85),
        (_rejected(idx=0, doc="r1"), 0.80),
    ]
    retriever = SourceStatusAwareRetriever(_StubIndex(rows))
    first = retriever.retrieve("query", top_k=4)
    second = retriever.retrieve("query", top_k=4)

    first_seq = [(r.chunk.chunk_id, r.rank) for r in first]
    second_seq = [(r.chunk.chunk_id, r.rank) for r in second]
    assert first_seq == second_seq


def test_funded_first_disabled_uses_pure_score_order() -> None:
    """With ``funded_first=False``, results are sorted strictly by descending score."""

    rows = [
        (_funded(idx=0, doc="f1"), 0.70),
        (_rejected(idx=0, doc="r1"), 0.85),  # rejected > funded by score
        (_esr(idx=0, doc="e1"), 0.95),  # ESR > both
    ]
    policy = RetrievalPolicy(funded_first=False, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=3)

    # Scores must be strictly non-increasing.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # And the ESR chunk (highest score) is rank 1, not the funded.
    assert results[0].source_status is SourceStatus.ESR_NOTE


def test_funded_first_disabled_status_priority_breaks_score_ties() -> None:
    """Even with ``funded_first=False``, status priority breaks *exact* score ties.

    Regression guard: an earlier implementation concatenated the status
    priority into the chunk_id string, which let lexicographic ordering
    of cross-document chunk_ids override the priority on tied scores. The
    advisor's catch — this test pins the correct behaviour.

    Setup: an ESR chunk and a FUNDED chunk both at score 0.85, with
    chunk_ids whose lexicographic ordering puts the ESR chunk first.
    Without the fix, ESR would rank above FUNDED on the tie. With the
    fix, FUNDED (priority 0) outranks ESR (priority 2).
    """

    # Document IDs chosen so the ESR chunk's chunk_id sorts BEFORE the
    # funded chunk's chunk_id alphabetically — this is what would have
    # made the buggy implementation order ESR first.
    rows = [
        (_esr(idx=0, doc="aaa_esr"), 0.85),
        (_funded(idx=0, doc="zzz_funded"), 0.85),  # exact tie
    ]
    policy = RetrievalPolicy(funded_first=False, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(_StubIndex(rows), policy=policy)
    results = retriever.retrieve("query", top_k=2)

    assert results[0].source_status is SourceStatus.FUNDED, (
        "On tied scores, status priority (FUNDED=0) MUST outrank ESR (=2) "
        "regardless of chunk_id lexicographic order."
    )
    assert results[1].source_status is SourceStatus.ESR_NOTE


def test_over_fetch_requests_more_than_top_k_from_index() -> None:
    """The retriever asks the index for ``top_k * 4`` candidates (capped)."""

    rows = [(_funded(idx=i, doc=f"f{i}"), 0.9 - i * 0.001) for i in range(20)]
    stub = _StubIndex(rows)
    retriever = SourceStatusAwareRetriever(stub)
    retriever.retrieve("query", top_k=5)

    assert stub.calls[-1]["top_k"] == 20  # 5 * 4

    # And the cap kicks in for very large top_k.
    retriever.retrieve("query", top_k=50)
    assert stub.calls[-1]["top_k"] == 100  # min(100, 50*4)


# ---------------------------------------------------------------------------
# Integration tests (real ChromaIndex + fixture chunks)
# ---------------------------------------------------------------------------


def _build_real_retriever(tmp_path):  # type: ignore[no-untyped-def]
    """Spin up a real :class:`ChromaIndex` populated with the four fixture chunks.

    Used by integration tests that need to exercise the round trip
    through Chroma — flatten/inflate of metadata, persistence of the
    source-status label, and the no-network contract.
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="retriever_integration",
    )
    index.upsert(build_fixture_chunks())
    return index


def test_integration_label_survives_chroma_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """AC1 round-trip: source-status survives the flatten/inflate path through Chroma."""

    index = _build_real_retriever(tmp_path)
    # Use a low threshold so the deterministic-hash embedder's modest
    # scores still pass — we're testing the round trip, not the policy.
    policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(index, policy=policy)

    results = retriever.retrieve(query_text_for("funded_horizon_europe.yaml"), top_k=5)
    assert results, "expected at least one result"
    for r in results:
        assert r.chunk.metadata.source_status is r.chunk.metadata.proposal.outcome
        assert r.source_status is r.chunk.metadata.source_status


def test_integration_marker_query_lands_funded_at_rank_1(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The funded fixture's marker token lands its chunk at rank 1."""

    index = _build_real_retriever(tmp_path)
    policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(index, policy=policy)

    results = retriever.retrieve(query_text_for("funded_horizon_europe.yaml"), top_k=4)
    assert results[0].source_status is SourceStatus.FUNDED
    assert results[0].rank == 1
    assert results[0].policy_reason == POLICY_REASON_FUNDED


def test_integration_filter_propagates_to_real_chroma(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``programme=HORIZON_EUROPE`` filters at Chroma level — no h2020 in results."""

    index = _build_real_retriever(tmp_path)
    policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(index, policy=policy)

    results = retriever.retrieve(
        "any text",
        top_k=10,
        programme=Programme.HORIZON_EUROPE,
    )
    assert results
    for r in results:
        assert r.chunk.metadata.proposal.programme is Programme.HORIZON_EUROPE


def test_integration_offline_no_network(tmp_path, no_network: None) -> None:  # type: ignore[no-untyped-def]
    """The whole retrieve path runs without any network access.

    Layered on top of the existing index offline-flow test in test_index.py
    to prove the policy layer doesn't accidentally add a network call.
    """

    index = _build_real_retriever(tmp_path)
    policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(index, policy=policy)

    results = retriever.retrieve(query_text_for("funded_horizon_europe.yaml"), top_k=2)
    assert results
    assert results[0].source_status is SourceStatus.FUNDED


def test_integration_section_type_fallback_against_chroma(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Issue #46 integration check on a real Chroma index.

    Reproduces the GEIGER pattern in miniature: the seeded chunks
    include exactly one ``methodology``-labelled fixture, none labelled
    ``consortium``. A query for the funded fixture's marker (which only
    matches that one methodology chunk) with ``section_type=consortium``
    would, without the fallback, return 0 results — exactly the gap the
    issue describes between ``index query`` (no filter, hits returned)
    and ``generate section --type consortium`` (filter, hits dropped).
    With the fallback enabled, the funded chunk comes back tagged so an
    audit can see the relaxation.
    """

    index = _build_real_retriever(tmp_path)
    policy = RetrievalPolicy(relevance_threshold=0.0, max_rejected_fraction=1.0)
    retriever = SourceStatusAwareRetriever(index, policy=policy)
    marker_query = query_text_for("funded_horizon_europe.yaml")

    # Strict mode (opt-out): no consortium-typed chunk exists → empty.
    strict_results = retriever.retrieve(
        marker_query,
        top_k=5,
        section_type=SectionType.CONSORTIUM,
        enable_section_type_fallback=False,
    )
    assert strict_results == []

    # Default mode (fallback on): the funded chunk comes back, tagged.
    fallback_results = retriever.retrieve(
        marker_query,
        top_k=5,
        section_type=SectionType.CONSORTIUM,
    )
    assert fallback_results, "fallback should have returned the marker chunk"
    # Identity check: the funded fixture's unique marker token lands at
    # rank 1, not any of the other three fixtures that happened to score.
    assert "funded_horizon_europe-marker" in fallback_results[0].chunk.text
    assert fallback_results[0].source_status is SourceStatus.FUNDED
    assert fallback_results[0].policy_reason.endswith(POLICY_REASON_SECTION_TYPE_FALLBACK_SUFFIX)
