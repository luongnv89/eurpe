"""Service facade for retrieval queries.

The retrieval service wraps :class:`SourceStatusAwareRetriever` behind
a single ``query(request)`` method so the React UI (Task 4.4 hybrid
search), the section-drafting workflow, and the CLI can all speak the
same wire shape. Before this issue each call site reconstructed the
per-call keyword arguments inline — manageable for three callers,
brittle as soon as a fourth shows up.

What this service does NOT own
------------------------------
* Index/embedder construction — those belong to higher-layer factories
  (the CLI builds them from :class:`EurpeConfig`). Injecting an
  already-constructed retriever keeps the service unit-testable
  without spinning up Chroma.
* Hybrid search (BM25 + dense) — Task 4.4 will land that as a swap of
  the underlying retriever. The service signature does not need to
  change because the filter set is already complete.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eurpe.retrieval.retriever import RetrievalResult, SourceStatusAwareRetriever
from eurpe.schema import Programme, SectionType, SourceStatus


class RetrievalQuery(BaseModel):
    """Input to :meth:`RetrievalService.query`.

    Mirrors the kwargs of :meth:`SourceStatusAwareRetriever.retrieve`
    so the service is a transparent wrapper today and a swap-point for
    Task 4.4 hybrid search tomorrow. The closed enum types
    (:class:`Programme`, :class:`SectionType`, :class:`SourceStatus`)
    are validated by Pydantic on construction — bad client input gets
    a 422 from FastAPI instead of a ValueError deep in retriever.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, description="The user's natural-language query.")
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum number of results.")
    programme: Programme | None = Field(
        default=None,
        description="Restrict to a single :class:`Programme` (hard filter).",
    )
    section_type: SectionType | None = Field(
        default=None,
        description="Restrict to a single :class:`SectionType` (hard filter, with fallback).",
    )
    source_status: SourceStatus | None = Field(
        default=None,
        description=(
            "Pin a specific :class:`SourceStatus`. When set, the section_type "
            "fallback is suppressed (the pin is a deliberate choice the "
            "service never relaxes)."
        ),
    )
    lessons_learned: bool | None = Field(
        default=None,
        description=(
            "Per-call override for :attr:`RetrievalPolicy.lessons_learned_mode`. "
            "``None`` inherits the policy default."
        ),
    )
    enable_section_type_fallback: bool | None = Field(
        default=None,
        description=(
            "Per-call override for the section_type fallback. ``None`` "
            "inherits the policy default."
        ),
    )


class RetrievalResponse(BaseModel):
    """Output of :meth:`RetrievalService.query`.

    Wrapping the list in a model leaves room for forward-compatible
    fields (``total_candidates``, ``policy_reason_counts``, etc.)
    without changing call-site signatures. ``result_count`` is exposed
    so HTTP callers can populate ``X-Total-Count`` headers without
    walking the list.

    The model_validator enforces ``result_count == len(results)`` so a
    bad caller cannot construct a response whose count lies about its
    payload — useful as a defence in depth when this travels over the
    wire and clients trust the integer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: list[RetrievalResult] = Field(default_factory=list)
    result_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _enforce_result_count_matches_results(self) -> Self:
        if self.result_count != len(self.results):
            raise ValueError(
                f"result_count ({self.result_count}) must match len(results) "
                f"({len(self.results)})."
            )
        return self


class RetrievalService:
    """Thin wrapper around :class:`SourceStatusAwareRetriever`.

    Stateless aside from the injected retriever. The retriever holds
    the source-status policy; this service holds nothing extra. Tests
    can build a service with a stub retriever and exercise the
    request/response models without touching Chroma.
    """

    def __init__(self, retriever: SourceStatusAwareRetriever) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> SourceStatusAwareRetriever:
        """Read-only access — used by tests and analytics surfaces."""

        return self._retriever

    def query(self, request: RetrievalQuery) -> RetrievalResponse:
        """Execute ``request`` and return the policy-filtered results.

        Errors:

        * :class:`ValueError` — propagated from the underlying
          retriever on impossible inputs (e.g., negative top_k slipped
          past Pydantic — Pydantic catches the obvious cases).
        """

        results = self._retriever.retrieve(
            request.query,
            top_k=request.top_k,
            programme=request.programme,
            section_type=request.section_type,
            source_status=request.source_status,
            lessons_learned=request.lessons_learned,
            enable_section_type_fallback=request.enable_section_type_fallback,
        )
        return RetrievalResponse(results=results, result_count=len(results))
