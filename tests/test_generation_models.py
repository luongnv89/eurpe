"""Tests for ``eurpe.generation.models``.

The three Pydantic records (``CitationRef``, ``GenerationRequest``,
``GenerationDraft``) are the public data shapes downstream code relies
on, so the invariants exercised here — extra="forbid", frozen
citations, required fields — are part of the package's contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eurpe.generation.models import (
    CitationRef,
    GenerationDraft,
    GenerationRequest,
)
from eurpe.schema import Programme, SectionType, SourceStatus

# ---------------------------------------------------------------------------
# CitationRef
# ---------------------------------------------------------------------------


def _sample_citation(**overrides: object) -> CitationRef:
    """Build a CitationRef with sensible defaults overridden."""

    base: dict[str, object] = dict(
        citation_id=1,
        source_status=SourceStatus.FUNDED,
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        proposal_title="Edge AI",
        section_heading="1.2 Methodology",
        page=12,
        chunk_id="doc::aaaa::000001",
        snippet="A short illustrative snippet.",
    )
    base.update(overrides)
    return CitationRef(**base)  # type: ignore[arg-type]


def test_citation_ref_basic_construction() -> None:
    c = _sample_citation()
    assert c.citation_id == 1
    assert c.source_status is SourceStatus.FUNDED
    assert c.programme is Programme.HORIZON_EUROPE


def test_citation_ref_is_frozen() -> None:
    """``frozen=True`` blocks attribute mutation after construction."""

    c = _sample_citation()
    with pytest.raises(ValidationError):
        c.citation_id = 99  # type: ignore[misc]


def test_citation_ref_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        CitationRef(  # type: ignore[call-arg]
            citation_id=1,
            source_status=SourceStatus.FUNDED,
            programme=Programme.HORIZON_EUROPE,
            call_id="X",
            chunk_id="x",
            snippet="x",
            unexpected_field="boom",
        )


def test_citation_ref_id_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _sample_citation(citation_id=0)


def test_citation_ref_snippet_capped_at_500_chars() -> None:
    too_long = "x" * 501
    with pytest.raises(ValidationError):
        _sample_citation(snippet=too_long)
    # Boundary: exactly 500 chars is OK.
    ok = _sample_citation(snippet="x" * 500)
    assert len(ok.snippet) == 500


def test_citation_ref_optional_fields_default_to_none() -> None:
    c = CitationRef(
        citation_id=2,
        source_status=SourceStatus.REJECTED,
        programme=Programme.HORIZON_2020,
        call_id="H2020-X",
        chunk_id="d::a::000000",
        snippet="...",
    )
    assert c.proposal_title is None
    assert c.section_heading is None
    assert c.page is None


# ---------------------------------------------------------------------------
# GenerationRequest
# ---------------------------------------------------------------------------


def test_generation_request_minimal_fields() -> None:
    req = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our DL approach",
    )
    assert req.section_type is SectionType.METHODOLOGY
    assert req.user_intent == "Describe our DL approach"
    # Defaults
    assert req.call_context == ""
    assert req.target_programme is None
    assert req.top_k_examples == 5
    assert req.lessons_learned is False


def test_generation_request_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(  # type: ignore[call-arg]
            section_type=SectionType.METHODOLOGY,
            user_intent="x",
            unexpected="boom",
        )


def test_generation_request_user_intent_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="",
        )


def test_generation_request_top_k_bounds_enforced() -> None:
    """``top_k_examples`` lives in [1, 20]; outside the range fails."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="x",
            top_k_examples=0,
        )
    with pytest.raises(ValidationError):
        GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="x",
            top_k_examples=21,
        )
    # Boundaries are accepted.
    GenerationRequest(section_type=SectionType.METHODOLOGY, user_intent="x", top_k_examples=1)
    GenerationRequest(section_type=SectionType.METHODOLOGY, user_intent="x", top_k_examples=20)


def test_generation_request_accepts_target_programme() -> None:
    req = GenerationRequest(
        section_type=SectionType.IMPACT_PATHWAY,
        user_intent="x",
        target_programme=Programme.HORIZON_EUROPE,
    )
    assert req.target_programme is Programme.HORIZON_EUROPE


# ---------------------------------------------------------------------------
# GenerationDraft
# ---------------------------------------------------------------------------


def _sample_draft(citations: list[CitationRef] | None = None) -> GenerationDraft:
    return GenerationDraft(
        section_type=SectionType.METHODOLOGY,
        text="Some draft text [1].",
        citations=citations if citations is not None else [_sample_citation()],
        prompt_used="prompt body",
        model="deterministic-stub-v1",
        request=GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="x",
        ),
    )


def test_generation_draft_basic_construction() -> None:
    d = _sample_draft()
    assert d.text == "Some draft text [1]."
    assert d.citation_count() == 1
    assert d.has_unlabeled_citations() is False


def test_generation_draft_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        GenerationDraft(  # type: ignore[call-arg]
            section_type=SectionType.METHODOLOGY,
            text="x",
            citations=[],
            prompt_used="p",
            model="m",
            request=GenerationRequest(section_type=SectionType.METHODOLOGY, user_intent="x"),
            unexpected="boom",
        )


def test_generation_draft_text_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        GenerationDraft(
            section_type=SectionType.METHODOLOGY,
            text="",
            citations=[],
            prompt_used="p",
            model="m",
            request=GenerationRequest(section_type=SectionType.METHODOLOGY, user_intent="x"),
        )


def test_generation_draft_citation_count_matches_list() -> None:
    cs = [_sample_citation(citation_id=i) for i in (1, 2, 3)]
    d = _sample_draft(citations=cs)
    assert d.citation_count() == 3


def test_generation_draft_has_unlabeled_citations_defensive_check() -> None:
    """Pydantic enforces ``source_status`` as required, so this returns False."""

    d = _sample_draft()
    # The defensive check returns False because Pydantic prevents
    # constructing a CitationRef without a source_status in the first
    # place. The method exists for downstream renderers to have a
    # single canonical check.
    assert d.has_unlabeled_citations() is False


def test_generation_draft_request_is_echoed() -> None:
    d = _sample_draft()
    assert d.request.section_type is SectionType.METHODOLOGY
    assert d.request.user_intent == "x"
