"""Tests for ``eurpe.intake.models``.

Pin the :class:`TopicContext` contract so the generation layer can rely
on it: ``extra="forbid"``, ``is_empty()`` semantics, JSON round-trip,
and the :class:`TopicSource` value set.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eurpe.intake import TopicContext, TopicSource
from eurpe.schema import Programme, SectionType


def test_topic_context_constructs_with_all_fields() -> None:
    """Every documented field round-trips through the constructor."""

    ctx = TopicContext(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL3-2024-CS-01",
        topic_id="952672",
        topic_title="Resilient digital infrastructure",
        expected_outcomes=["outcome 1", "outcome 2"],
        scope="Proposals should address...",
        destination="Cluster 3",
        section_guidance={SectionType.METHODOLOGY: "validate on pilots"},
        raw_text="full normalised body",
        source=TopicSource.PASTED_TEXT,
    )
    assert ctx.programme is Programme.HORIZON_EUROPE
    assert ctx.call_id == "HORIZON-CL3-2024-CS-01"
    assert ctx.topic_id == "952672"
    assert ctx.topic_title == "Resilient digital infrastructure"
    assert ctx.expected_outcomes == ["outcome 1", "outcome 2"]
    assert ctx.scope == "Proposals should address..."
    assert ctx.destination == "Cluster 3"
    assert ctx.section_guidance == {SectionType.METHODOLOGY: "validate on pilots"}
    assert ctx.raw_text == "full normalised body"
    assert ctx.source is TopicSource.PASTED_TEXT
    assert ctx.source_path is None


def test_topic_context_extra_fields_forbidden() -> None:
    """A stray key surfaces as a validation error.

    Matches the convention used across ``eurpe.schema`` / ``ingestion``
    / ``generation``: typos in field names must fail loudly.
    """

    with pytest.raises(ValidationError):
        TopicContext(
            unknown_field="oops",  # type: ignore[call-arg]
            source=TopicSource.PASTED_TEXT,
        )


def test_topic_context_source_required() -> None:
    """``source`` has no default — every record must declare its origin."""

    with pytest.raises(ValidationError):
        TopicContext()  # type: ignore[call-arg]


def test_topic_context_empty_default_is_empty() -> None:
    """Blank instance — only ``source`` set — is :meth:`is_empty`."""

    ctx = TopicContext(source=TopicSource.PASTED_TEXT)
    assert ctx.is_empty() is True


def test_topic_context_is_not_empty_when_topic_id_set() -> None:
    ctx = TopicContext(topic_id="952672", source=TopicSource.PASTED_TEXT)
    assert ctx.is_empty() is False


def test_topic_context_is_not_empty_when_topic_title_set() -> None:
    ctx = TopicContext(topic_title="t", source=TopicSource.PASTED_TEXT)
    assert ctx.is_empty() is False


def test_topic_context_is_not_empty_when_outcomes_populated() -> None:
    ctx = TopicContext(expected_outcomes=["one"], source=TopicSource.PASTED_TEXT)
    assert ctx.is_empty() is False


def test_topic_context_json_roundtrip() -> None:
    """Pydantic v2 round-trip via JSON preserves every field, including the
    ``SectionType``-keyed ``section_guidance`` dict.
    """

    ctx = TopicContext(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL3-2024-CS-01",
        topic_id="952672",
        topic_title="Resilient digital infrastructure",
        expected_outcomes=["outcome 1"],
        scope="...",
        destination="Cluster 3",
        section_guidance={SectionType.METHODOLOGY: "guidance text"},
        raw_text="raw",
        source=TopicSource.PDF_EXCERPT,
        source_path="/tmp/topic.pdf",
    )
    payload = ctx.model_dump_json()
    revived = TopicContext.model_validate_json(payload)
    assert revived == ctx
    assert revived.section_guidance[SectionType.METHODOLOGY] == "guidance text"


def test_topic_source_enum_values() -> None:
    """The two source-mode tokens are stable and lowercase-snake."""

    assert TopicSource.PASTED_TEXT.value == "pasted_text"
    assert TopicSource.PDF_EXCERPT.value == "pdf_excerpt"
    assert {s.value for s in TopicSource} == {"pasted_text", "pdf_excerpt"}
