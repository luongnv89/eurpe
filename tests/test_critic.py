"""Tests for ``eurpe.generation.critic`` — the source-grounded critic agent.

The critic is the inner half of the Task 3.2 / issue #16 critic loop.
These tests pin three contracts:

* :func:`build_requirements_checked` is deterministic — same inputs
  yield the same list, sentinel-first, dedup-preserved. AC #3 ("which
  call/profile requirements were checked") depends on this list being
  reconstructible without an LLM round-trip.
* :class:`CriticAgent.critique` returns ``(critique_text,
  requirements_checked)`` with both halves non-empty. The
  deterministic LLM stub exercises the offline contract.
* The critic prompt names every requirement so a manual / fuzz tester
  can verify the LLM saw what the request demanded.
"""

from __future__ import annotations

import pytest

from eurpe.generation import (
    CriticAgent,
    DeterministicLLMClient,
    GenerationDraft,
    GenerationRequest,
)
from eurpe.generation.critic import (
    _DEFAULT_REQUIREMENT_SENTINEL,
    build_critic_prompt,
    build_requirements_checked,
)
from eurpe.generation.profiles import DraftingProfile
from eurpe.intake.models import TopicContext, TopicSource
from eurpe.schema import Programme, SectionType


def _make_basic_draft(section: SectionType = SectionType.METHODOLOGY) -> GenerationDraft:
    """Construct a minimal GenerationDraft for critic tests.

    The critic only reads ``section_type``, ``text``, and ``citations`` from
    the draft, so the other fields are filled with placeholder values that
    satisfy the Pydantic constraints.
    """

    request = GenerationRequest(
        section_type=section,
        user_intent="Describe a deep-learning approach to anomaly detection.",
    )
    return GenerationDraft(
        section_type=section,
        text=(
            "## Methodology\n"
            "We will apply graph neural networks [1] augmented with "
            "self-supervised pretraining to detect anomalies in critical "
            "infrastructure logs."
        ),
        citations=[],
        prompt_used="(test placeholder)",
        model="deterministic-stub-v1",
        request=request,
    )


# ---------------------------------------------------------------------------
# build_requirements_checked
# ---------------------------------------------------------------------------


class TestBuildRequirementsChecked:
    def test_sentinel_only_when_no_profile_and_no_topic(self) -> None:
        """Even with no profile and no topic context, the list is non-empty."""

        result = build_requirements_checked(
            section_type=SectionType.METHODOLOGY,
            profile=None,
            topic_context=None,
        )
        assert result == [_DEFAULT_REQUIREMENT_SENTINEL]

    def test_profile_outputs_follow_sentinel(self) -> None:
        """Profile expected_outputs append in declared order after the sentinel."""

        profile = DraftingProfile(
            programme=Programme.HORIZON_EUROPE,
            name="Horizon Europe Standard",
            expected_outputs={
                SectionType.METHODOLOGY: [
                    "validation strategy",
                    "risk register",
                ],
            },
        )
        result = build_requirements_checked(
            section_type=SectionType.METHODOLOGY,
            profile=profile,
            topic_context=None,
        )
        assert result == [
            _DEFAULT_REQUIREMENT_SENTINEL,
            "validation strategy",
            "risk register",
        ]

    def test_topic_section_guidance_keys_added_with_prefix(self) -> None:
        """TopicContext section_guidance keys add ``topic:<value>`` entries."""

        topic = TopicContext(
            programme=Programme.HORIZON_EUROPE,
            section_guidance={
                SectionType.METHODOLOGY: "Use FAIR principles.",
                SectionType.IMPACT: "Quantify societal impact.",
            },
            source=TopicSource.PASTED_TEXT,
            raw_text="",
        )
        result = build_requirements_checked(
            section_type=SectionType.METHODOLOGY,
            profile=None,
            topic_context=topic,
        )
        assert _DEFAULT_REQUIREMENT_SENTINEL in result
        assert "topic:methodology" in result
        assert "topic:impact" in result

    def test_duplicates_deduplicated_first_seen_position(self) -> None:
        """A label that appears in both profile and topic only appears once."""

        profile = DraftingProfile(
            programme=Programme.HORIZON_EUROPE,
            name="P",
            expected_outputs={
                SectionType.METHODOLOGY: [
                    "validation strategy",
                    "validation strategy",  # explicit duplicate
                    "topic:methodology",  # collides with topic key form
                ],
            },
        )
        topic = TopicContext(
            programme=Programme.HORIZON_EUROPE,
            section_guidance={SectionType.METHODOLOGY: "use FAIR"},
            source=TopicSource.PASTED_TEXT,
            raw_text="",
        )
        result = build_requirements_checked(
            section_type=SectionType.METHODOLOGY,
            profile=profile,
            topic_context=topic,
        )
        # Each label appears once; order is first-seen.
        assert result.count("validation strategy") == 1
        assert result.count("topic:methodology") == 1
        assert result[0] == _DEFAULT_REQUIREMENT_SENTINEL


# ---------------------------------------------------------------------------
# build_critic_prompt
# ---------------------------------------------------------------------------


class TestBuildCriticPrompt:
    def test_prompt_names_every_requirement(self) -> None:
        """Each requirement appears verbatim in the rendered prompt."""

        draft = _make_basic_draft()
        requirements = [
            _DEFAULT_REQUIREMENT_SENTINEL,
            "validation strategy",
            "risk register",
        ]
        prompt = build_critic_prompt(
            draft=draft,
            requirements=requirements,
            iteration_index=2,
            max_iterations=3,
        )
        for req in requirements:
            assert req in prompt, f"requirement {req!r} missing from critic prompt"

    def test_prompt_mentions_iteration_index_and_cap(self) -> None:
        """The critic prompt names the iteration position so the LLM can scale advice."""

        prompt = build_critic_prompt(
            draft=_make_basic_draft(),
            requirements=[_DEFAULT_REQUIREMENT_SENTINEL],
            iteration_index=2,
            max_iterations=5,
        )
        # iteration_index - 1 = pass number (the first critic pass is "pass 1 of 4")
        assert "iteration 1" in prompt
        assert "4" in prompt

    def test_prompt_includes_prior_draft_text(self) -> None:
        """The critic must see the prior draft text to evaluate it."""

        draft = _make_basic_draft()
        prompt = build_critic_prompt(
            draft=draft,
            requirements=[_DEFAULT_REQUIREMENT_SENTINEL],
            iteration_index=2,
            max_iterations=3,
        )
        assert "graph neural networks" in prompt


# ---------------------------------------------------------------------------
# CriticAgent.critique
# ---------------------------------------------------------------------------


class TestCriticAgent:
    def test_critique_returns_non_empty_text_and_requirements_under_offline_stub(
        self, no_network: None
    ) -> None:
        """The deterministic stub runs offline and produces a usable critique."""

        del no_network  # signal: the fixture activates the network block
        agent = CriticAgent(DeterministicLLMClient())
        text, reqs = agent.critique(
            prior_draft=_make_basic_draft(),
            profile=None,
            topic_context=None,
            iteration_index=2,
            max_iterations=3,
        )
        assert text.strip(), "deterministic critic returned empty text"
        assert reqs == [_DEFAULT_REQUIREMENT_SENTINEL]

    def test_critique_truncates_long_output_at_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critique longer than 4000 chars is truncated at the cap (not rejected)."""

        class _LongLLM:
            model = "long-test-llm"

            def generate(self, prompt: str, **_: object) -> str:  # noqa: ARG002
                return "x" * 10_000

        agent = CriticAgent(_LongLLM())  # type: ignore[arg-type]
        text, _ = agent.critique(
            prior_draft=_make_basic_draft(),
            profile=None,
            topic_context=None,
            iteration_index=2,
            max_iterations=3,
        )
        assert len(text) <= 4000

    def test_critique_substitutes_default_when_llm_returns_empty(self) -> None:
        """Empty LLM response is replaced with a manual-review notice."""

        class _EmptyLLM:
            model = "empty-test-llm"

            def generate(self, prompt: str, **_: object) -> str:  # noqa: ARG002
                return ""

        agent = CriticAgent(_EmptyLLM())  # type: ignore[arg-type]
        text, _ = agent.critique(
            prior_draft=_make_basic_draft(),
            profile=None,
            topic_context=None,
            iteration_index=2,
            max_iterations=3,
        )
        assert "manually" in text.lower() or "empty" in text.lower()
