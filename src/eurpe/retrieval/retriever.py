"""Source-status-aware retriever — the policy layer on top of :class:`ChromaIndex`.

EURPE's retrieval guarantees are policy decisions, not embedder decisions:

* Funded examples are the *primary* positive pattern source. They appear
  first when scores tie or are very close.
* Rejected examples must clear the **same topical relevance threshold** as
  funded ones — they cannot sneak through at a lower bar (PRD § 142, § 187).
  They are visibly labelled (the chunk's :attr:`SourceStatus` is preserved
  end-to-end) and capped at a fraction of top-k so they don't flood results.
* ESR (External Subject Reviewer) notes are advisory commentary only;
  ranked below rejected on near-ties, and excludable entirely with a flag.
* Unknown-status chunks are present but treated as low-confidence; ranked
  last on near-ties.

The "lessons learned" mode is the explicit opt-in for cautionary contrast:
when enabled, the rejected threshold is relaxed by an offset (typically
negative, e.g., -0.10) and the rejected-fraction cap is skipped. This is
how AC2 ("rejected examples must satisfy the same topical relevance
threshold ... unless a lessons-learned flag is enabled") is realised.

The retriever does NOT call the embedder itself; it delegates to
:class:`ChromaIndex.query` which embeds the query text under the configured
:class:`Embedder`. Over-fetching by a factor of 4 (clamped at 100) gives
the policy enough candidates to filter without starving the result set.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from eurpe.retrieval.index import ChromaIndex
from eurpe.retrieval.models import Chunk
from eurpe.schema import Programme, SectionType, SourceStatus

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

#: Multiplier applied to ``top_k`` when over-fetching candidates from Chroma.
#: Policy filtering (threshold, fraction cap, ESR exclusion) can drop many
#: candidates, so over-fetching keeps the pool large enough to still return
#: ``top_k`` results in the common case. Clamped at :data:`_MAX_FETCH_CANDIDATES`
#: so a caller asking for ``top_k=50`` doesn't trigger an unbounded query.
_OVER_FETCH_MULTIPLIER = 4

#: Hard ceiling on the candidate pool size requested from Chroma.
_MAX_FETCH_CANDIDATES = 100

#: Source-status priority used as a sort tiebreaker. Lower wins.
#: FUNDED < REJECTED < ESR_NOTE < UNKNOWN — i.e., funded examples beat
#: rejected on near-ties, etc. Ordering is documented in the PRD.
_STATUS_PRIORITY: dict[SourceStatus, int] = {
    SourceStatus.FUNDED: 0,
    SourceStatus.REJECTED: 1,
    SourceStatus.ESR_NOTE: 2,
    SourceStatus.UNKNOWN: 3,
}

#: Human-readable rationale strings written into :attr:`RetrievalResult.policy_reason`.
#: Centralised so callers and tests can compare against the canonical value.
POLICY_REASON_FUNDED = "funded_primary"
POLICY_REASON_REJECTED = "rejected_threshold_met"
POLICY_REASON_LESSONS_LEARNED = "lessons_learned_mode"
POLICY_REASON_ESR = "esr_advisory"
POLICY_REASON_UNKNOWN = "unknown_low_confidence"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """A single retrieval hit after the source-status policy has been applied.

    The wrapped :class:`Chunk` carries its own :attr:`SourceStatus` (via
    ``chunk.metadata.source_status``) — exposed here as the
    :attr:`source_status` convenience property — so AC1 ("Retriever returns
    top-k chunks with source-status labels attached to every result") is
    satisfied structurally: there is no path that strips the label.

    ``score`` is the cosine similarity reported by Chroma in ``1 - distance``
    form. With L2-normalised vectors and Chroma's cosine space the value
    lives in ``[-1, 1]``; the deterministic-hash embedder produces
    non-negative vectors so its scores are in ``[0, 1]``. The wider bound
    matches what the index can actually return, so a future swap to a real
    embedder that produces signed components doesn't trip the validator.

    ``policy_reason`` records *why* this chunk was kept — useful in CLI
    output and in audit logs. Values are the ``POLICY_REASON_*`` constants
    above. ``rank`` is 1-based for human display.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk = Field(description="The retrieved chunk with full metadata.")
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Cosine similarity from the index, in [-1, 1]. Higher is better."
        ),
    )
    rank: int = Field(
        ge=1,
        description="1-based final rank after the source-status policy is applied.",
    )
    policy_reason: str = Field(
        default=POLICY_REASON_FUNDED,
        description=(
            "Human-readable rationale: funded_primary, rejected_threshold_met, "
            "lessons_learned_mode, esr_advisory, or unknown_low_confidence."
        ),
    )

    @property
    def source_status(self) -> SourceStatus:
        """Convenience accessor — the chunk's source status label."""
        return self.chunk.metadata.source_status


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


class RetrievalPolicy(BaseModel):
    """Configuration knobs for source-status-aware retrieval.

    All knobs default to the PRD's stated invariants: funded-first, uniform
    relevance threshold across all statuses, ESR included but ranked low,
    rejected results capped at 40% of top-k, lessons-learned mode off.

    The defaults are tuned for the most common call site — drafting a new
    section where the user wants funded patterns as the primary signal —
    so most callers can construct a retriever with no policy argument at
    all and get the expected behaviour.
    """

    relevance_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for a chunk to be considered topically "
            "relevant. Applied uniformly to ALL source statuses by default — "
            "this is the PRD invariant that AC2 enforces."
        ),
    )
    lessons_learned_mode: bool = Field(
        default=False,
        description=(
            "When True, rejected examples are surfaced more aggressively as "
            "cautionary evidence: their threshold is relaxed by "
            "rejected_threshold_offset, and the rejected-fraction cap is "
            "skipped. Use when generating contrastive 'what to avoid' content."
        ),
    )
    rejected_threshold_offset: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description=(
            "Added to relevance_threshold for REJECTED chunks when "
            "lessons_learned_mode is True. A NEGATIVE value relaxes the bar "
            "(typical: -0.10). Ignored when lessons_learned_mode is False."
        ),
    )
    include_esr: bool = Field(
        default=True,
        description=(
            "If False, ESR notes are excluded from results entirely. Useful "
            "for the final-draft pipeline when the user has decided to skip "
            "subjective reviewer commentary."
        ),
    )
    funded_first: bool = Field(
        default=True,
        description=(
            "When True (default), the secondary sort key is status priority "
            "(FUNDED < REJECTED < ESR_NOTE < UNKNOWN). When False, results "
            "are ordered strictly by score; status is only a deterministic "
            "tiebreaker. Set False to benchmark the policy itself."
        ),
    )
    max_rejected_fraction: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum fraction of the returned top-k that may be rejected "
            "chunks. Even when many rejected chunks pass the threshold, the "
            "user mostly wants funded patterns. Set to 1.0 to disable. "
            "Ignored when lessons_learned_mode is True."
        ),
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Index protocol
# ---------------------------------------------------------------------------


class _IndexQueryProtocol:
    """Structural type for what the retriever needs from a backing index.

    Spelled out so tests can substitute a stub. Not exported; the
    :class:`SourceStatusAwareRetriever` constructor accepts any object
    that has a compatible ``query`` method (duck-typed).
    """

    def query(  # pragma: no cover - documentation only
        self,
        query_text: str,
        *,
        top_k: int = 10,
        where: dict[str, object] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return ``(chunk, similarity)`` pairs ordered by descending similarity."""
        ...


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class SourceStatusAwareRetriever:
    """Wraps a :class:`ChromaIndex` (or any compatible index) with policy filtering.

    Acceptance criteria covered by this class:

    * **AC1.** Every :class:`RetrievalResult` carries the chunk's
      :class:`SourceStatus` via :attr:`RetrievalResult.source_status`. The
      retriever never strips or rewrites the label — :class:`Chunk` is passed
      through verbatim.
    * **AC2.** :attr:`RetrievalPolicy.relevance_threshold` is applied
      uniformly to every status by default. Only when
      :attr:`RetrievalPolicy.lessons_learned_mode` is True (or the per-call
      override is set) does the rejected threshold relax by
      :attr:`RetrievalPolicy.rejected_threshold_offset`.
    * **AC3.** Funded-only, rejected-only, mixed-status, and no-match
      scenarios are exercised in :mod:`tests.test_retriever`.

    The retriever is intentionally stateless apart from the immutable
    policy and the index reference — multiple callers can share one
    instance from different threads. The index itself is the only mutable
    component, and Chroma's PersistentClient is documented as thread-safe
    for read operations.
    """

    def __init__(
        self,
        index: ChromaIndex | _IndexQueryProtocol,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        # The type annotation is the intersection of "real ChromaIndex" and
        # "anything with a compatible query()". We don't use a Protocol here
        # because ChromaIndex isn't decorated with @runtime_checkable, and
        # we want type checkers to accept both real and stub indexes
        # without forcing every caller into a Protocol type.
        self._index = index
        self._policy = policy or RetrievalPolicy()

    @property
    def policy(self) -> RetrievalPolicy:
        """Read-only access to the configured policy."""
        return self._policy

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        programme: Programme | None = None,
        section_type: SectionType | None = None,
        source_status: SourceStatus | None = None,
        lessons_learned: bool | None = None,
    ) -> list[RetrievalResult]:
        """Return up to ``top_k`` policy-filtered results for ``query``.

        Steps:

        1. Build a Chroma ``where`` clause for the **hard** filters
           (programme, section_type, optional source_status). Source status
           is intentionally NOT auto-filtered server-side: the policy needs
           to see all statuses so the rejected-fraction cap and the
           funded-first ordering are computable. The caller can still pass
           ``source_status=`` explicitly to scope to one status.
        2. Over-fetch candidates from Chroma (``top_k * 4``, capped at 100).
        3. Apply the per-status threshold. Default: same threshold for
           every status. Lessons-learned mode (global or per-call): the
           REJECTED threshold relaxes by ``rejected_threshold_offset``.
        4. Drop ESR notes if ``include_esr=False``.
        5. Sort: by ``(status_priority, -score, chunk_id)`` when
           ``funded_first=True``; by ``(-score, status_priority, chunk_id)``
           otherwise. The trailing ``chunk_id`` makes ordering deterministic
           across runs even when scores tie.
        6. Cap rejected results at ``floor(top_k * max_rejected_fraction)``
           — skip extra rejected candidates while continuing to accept
           others (the advisor's "skip-not-truncate" rule). Skipped when
           lessons_learned is in effect.
        7. Take the first ``top_k``, assign ranks, attach policy_reason.

        ``lessons_learned`` (per-call override) takes precedence over
        :attr:`RetrievalPolicy.lessons_learned_mode` when not None. Use
        ``None`` to inherit the policy default.
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        lessons_active = (
            self._policy.lessons_learned_mode if lessons_learned is None else lessons_learned
        )

        # Step 1: hard filters → Chroma where clause.
        where = self._build_where_clause(
            programme=programme,
            section_type=section_type,
            source_status=source_status,
        )

        # Step 2: over-fetch candidates so policy has room to work.
        candidate_k = min(_MAX_FETCH_CANDIDATES, top_k * _OVER_FETCH_MULTIPLIER)
        raw = self._index.query(query, top_k=candidate_k, where=where)

        # Step 3 + 4: threshold filter + ESR exclusion.
        filtered = self._apply_threshold(raw, lessons_active=lessons_active)

        # Step 5: sort with status priority tiebreak.
        sorted_candidates = self._sort_candidates(filtered)

        # Step 6: cap rejected fraction (skip-not-truncate).
        capped = self._apply_rejected_cap(
            sorted_candidates,
            top_k=top_k,
            lessons_active=lessons_active,
        )

        # Step 7: take top_k, assign ranks + reasons.
        return self._build_results(capped[:top_k], lessons_active=lessons_active)

    # ------------------------------------------------------------------
    # Internal helpers — kept private but small enough to test indirectly
    # via the public ``retrieve()`` path.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where_clause(
        *,
        programme: Programme | None,
        section_type: SectionType | None,
        source_status: SourceStatus | None,
    ) -> dict[str, object] | None:
        """Build the Chroma ``where`` filter for hard (server-side) filters.

        Chroma 1.x expects a single ``{"key": "value"}`` for one filter and
        ``{"$and": [...]}`` for multiple. Mirrors the convention used in
        :func:`eurpe.retrieval.cli._build_query_where`.
        """

        filters: list[dict[str, str]] = []
        if programme is not None:
            filters.append({"proposal.programme": programme.value})
        if section_type is not None:
            filters.append({"section_type": section_type.value})
        if source_status is not None:
            filters.append({"source_status": source_status.value})
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def _threshold_for(self, status: SourceStatus, *, lessons_active: bool) -> float:
        """Return the effective threshold for a given source status.

        Default: every status uses ``policy.relevance_threshold``. Under
        lessons-learned mode, the REJECTED threshold is shifted by
        ``rejected_threshold_offset`` (typically negative). Threshold is
        clamped to ``[0, 1]`` so a misconfigured offset can't produce a
        negative bar that lets every chunk through.
        """

        base = self._policy.relevance_threshold
        if lessons_active and status is SourceStatus.REJECTED:
            shifted = base + self._policy.rejected_threshold_offset
            return max(0.0, min(1.0, shifted))
        return base

    def _apply_threshold(
        self,
        candidates: list[tuple[Chunk, float]],
        *,
        lessons_active: bool,
    ) -> list[tuple[Chunk, float]]:
        """Drop candidates below their per-status threshold and (optionally) ESR."""

        kept: list[tuple[Chunk, float]] = []
        for chunk, score in candidates:
            status = chunk.metadata.source_status
            if not self._policy.include_esr and status is SourceStatus.ESR_NOTE:
                continue
            if score < self._threshold_for(status, lessons_active=lessons_active):
                continue
            kept.append((chunk, score))
        return kept

    def _sort_candidates(
        self,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Sort by the configured priority. ``chunk_id`` is the deterministic tiebreak."""

        # Each branch builds a 3-tuple sort key; element types are uniform
        # *within* each branch so element-by-element comparison is well
        # defined. The branches differ in which key (status priority vs
        # negated score) leads, which is the whole point of funded_first.
        funded_first = self._policy.funded_first

        def _key(item: tuple[Chunk, float]) -> tuple[float, float, str]:
            chunk, score = item
            priority = float(_STATUS_PRIORITY[chunk.metadata.source_status])
            neg_score = -score
            chunk_id = chunk.chunk_id
            if funded_first:
                # Funded-first: status priority leads (FUNDED=0 wins ties),
                # then descending score, then chunk_id deterministically.
                return (priority, neg_score, chunk_id)
            # Pure score order: descending score leads, status priority only
            # breaks exact-score ties, chunk_id breaks remaining ties.
            # Critical: priority must be its own tuple element — concatenating
            # it into chunk_id would let lexicographic ordering of unrelated
            # documents override the priority on real ties.
            return (neg_score, priority, chunk_id)

        return sorted(candidates, key=_key)

    def _apply_rejected_cap(
        self,
        candidates: list[tuple[Chunk, float]],
        *,
        top_k: int,
        lessons_active: bool,
    ) -> list[tuple[Chunk, float]]:
        """Skip extra rejected candidates beyond the cap (skip-not-truncate).

        Walk the sorted candidates: accept non-rejected freely; once the
        rejected count reaches the cap, skip further rejected candidates
        but keep accepting others. Lessons-learned mode opts out — when on,
        we deliberately want more rejected signal.
        """

        if lessons_active or self._policy.max_rejected_fraction >= 1.0:
            return candidates

        # ``floor`` semantics — at top_k=10 with cap=0.4 we allow 4 rejected.
        # int() on a non-negative float is equivalent to floor.
        max_rejected = int(top_k * self._policy.max_rejected_fraction)

        kept: list[tuple[Chunk, float]] = []
        rejected_seen = 0
        for chunk, score in candidates:
            if chunk.metadata.source_status is SourceStatus.REJECTED:
                if rejected_seen >= max_rejected:
                    # Skip this rejected — but don't stop; later non-rejected
                    # candidates may still be accepted.
                    continue
                rejected_seen += 1
            kept.append((chunk, score))
        return kept

    @staticmethod
    def _policy_reason_for(status: SourceStatus, *, lessons_active: bool) -> str:
        """Map a status (and the lessons-learned flag) to its policy reason string."""

        if status is SourceStatus.FUNDED:
            return POLICY_REASON_FUNDED
        if status is SourceStatus.REJECTED:
            return POLICY_REASON_LESSONS_LEARNED if lessons_active else POLICY_REASON_REJECTED
        if status is SourceStatus.ESR_NOTE:
            return POLICY_REASON_ESR
        # SourceStatus.UNKNOWN — and any future statuses default to "low confidence".
        return POLICY_REASON_UNKNOWN

    def _build_results(
        self,
        candidates: list[tuple[Chunk, float]],
        *,
        lessons_active: bool,
    ) -> list[RetrievalResult]:
        """Turn the final candidate list into :class:`RetrievalResult` records."""

        out: list[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(candidates, start=1):
            status = chunk.metadata.source_status
            out.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    policy_reason=self._policy_reason_for(status, lessons_active=lessons_active),
                )
            )
        return out
