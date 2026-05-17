"""Tests for ``eurpe.generation.critic_loop`` — the per-iteration critic workflow.

Pins the three acceptance criteria of issue #16 at the workflow level:

* **AC #1** ("user can set critic iterations between 1 and 5"):
  :func:`_clamp_max_iterations` accepts [1, 5] and clamps out-of-range
  values; :meth:`CriticLoopWorkflow.iterate` rejects ``max_iterations <
  2`` (one is no loop).
* **AC #2** ("user can stop the loop after any completed iteration"):
  the loop is *per-iteration* — each call to :meth:`iterate` advances
  by one and returns the refined draft + ``stopped`` flag. The client
  stops by not calling again. A test drives 1, 2, and 3 iterations
  through the same prior chain to prove the loop is stoppable at any
  position.
* **AC #3** ("each iteration records what changed and which call/
  profile requirements were checked"): every
  :class:`IterationRecord` carries non-empty ``changes_summary`` and
  ``requirements_checked``; the cumulative history travels with the
  draft.

All tests run against the deterministic LLM stub + ``no_network``
fixture so the offline contract is exercised end-to-end.
"""

from __future__ import annotations

import pytest

from eurpe.generation import (
    CriticAgent,
    CriticLoopWorkflow,
    GenerationDraft,
    GenerationError,
    GenerationRequest,
    IterationRecord,
    SectionGenerationWorkflow,
)
from eurpe.generation.critic import _DEFAULT_REQUIREMENT_SENTINEL
from eurpe.generation.critic_loop import (
    DEFAULT_MAX_ITERATIONS,
    MAX_ITERATIONS_CEILING,
    _build_changes_summary,
    _clamp_max_iterations,
)
from eurpe.generation.profiles import DraftingProfile
from eurpe.intake.models import TopicContext, TopicSource
from eurpe.schema import Programme, SectionType


@pytest.fixture
def critic_loop(
    deterministic_workflow: SectionGenerationWorkflow,
) -> CriticLoopWorkflow:
    """Critic loop wired to the deterministic workflow fixture.

    Shares the workflow's LLM client for both drafting and critique —
    matches the default :class:`GenerationService` wiring.
    """

    critic = CriticAgent(deterministic_workflow.llm)
    return CriticLoopWorkflow(workflow=deterministic_workflow, critic=critic)


def _basic_request() -> GenerationRequest:
    """Request that exercises the deterministic workflow fixture's corpus."""

    return GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe a deep learning methodology with validation.",
        top_k_examples=3,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestClampMaxIterations:
    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_accepts_values_in_range(self, value: int) -> None:
        assert _clamp_max_iterations(value) == value

    def test_clamps_values_above_ceiling(self) -> None:
        assert _clamp_max_iterations(10) == MAX_ITERATIONS_CEILING

    def test_rejects_zero_or_negative(self) -> None:
        with pytest.raises(ValueError):
            _clamp_max_iterations(0)
        with pytest.raises(ValueError):
            _clamp_max_iterations(-1)


class TestBuildChangesSummary:
    def test_summary_includes_text_and_citation_deltas(self) -> None:
        prior = GenerationDraft(
            section_type=SectionType.METHODOLOGY,
            text="short prior",
            citations=[],
            prompt_used="(t)",
            model="m",
            request=_basic_request(),
        )
        refined = GenerationDraft(
            section_type=SectionType.METHODOLOGY,
            text="a much longer refined draft",
            citations=[],
            prompt_used="(t)",
            model="m",
            request=_basic_request(),
        )
        summary = _build_changes_summary(prior, refined)
        # Summary mentions both the char and citation deltas.
        assert "chars" in summary
        assert "citations" in summary
        assert "+" in summary  # text grew

    def test_summary_reports_no_structural_change_when_identical(self) -> None:
        draft = GenerationDraft(
            section_type=SectionType.METHODOLOGY,
            text="identical draft",
            citations=[],
            prompt_used="(t)",
            model="m",
            request=_basic_request(),
        )
        summary = _build_changes_summary(draft, draft)
        assert "no structural change" in summary.lower()


# ---------------------------------------------------------------------------
# iterate() — public surface
# ---------------------------------------------------------------------------


class TestIterateBoundary:
    def test_rejects_max_iterations_one(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
    ) -> None:
        """max_iterations=1 leaves no room for a critic pass."""

        prior = deterministic_workflow.run(_basic_request())
        with pytest.raises(GenerationError, match="max_iterations<2"):
            critic_loop.iterate(
                prior_draft=prior,
                request=_basic_request(),
                max_iterations=1,
            )

    def test_clamps_value_above_ceiling(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
    ) -> None:
        """A request above the ceiling clamps to MAX_ITERATIONS_CEILING."""

        prior = deterministic_workflow.run(_basic_request())
        result = critic_loop.iterate(
            prior_draft=prior,
            request=_basic_request(),
            max_iterations=10,  # over the ceiling
        )
        assert result.max_iterations == MAX_ITERATIONS_CEILING
        assert result.iteration_index == 2

    def test_rejects_when_prior_already_at_cap(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
    ) -> None:
        """A prior draft already at the cap cannot be refined further."""

        prior = deterministic_workflow.run(_basic_request())
        # Manually inflate the iteration history to the cap (2 records ⇒
        # prior.total_iterations() == 3) so the next attempt at cap=3
        # exceeds the limit.
        prior_with_iters = prior.model_copy(
            update={
                "iterations": [
                    IterationRecord(
                        iteration_index=2,
                        changes_summary="prior",
                        requirements_checked=[_DEFAULT_REQUIREMENT_SENTINEL],
                        critique_text="prior critique",
                    ),
                    IterationRecord(
                        iteration_index=3,
                        changes_summary="prior",
                        requirements_checked=[_DEFAULT_REQUIREMENT_SENTINEL],
                        critique_text="prior critique",
                    ),
                ]
            }
        )
        with pytest.raises(GenerationError, match="already at max_iterations"):
            critic_loop.iterate(
                prior_draft=prior_with_iters,
                request=_basic_request(),
                max_iterations=3,
            )


# ---------------------------------------------------------------------------
# AC #2 / AC #3: per-iteration record + stop control
# ---------------------------------------------------------------------------


class TestIterateRecordsHistory:
    def test_single_pass_appends_one_iteration_record(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
        no_network: None,
    ) -> None:
        """One iterate() call produces iteration 2 and a non-empty record."""

        del no_network
        prior = deterministic_workflow.run(_basic_request())
        result = critic_loop.iterate(
            prior_draft=prior,
            request=_basic_request(),
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )
        assert result.iteration_index == 2
        assert result.max_iterations == DEFAULT_MAX_ITERATIONS
        assert result.stopped is False  # 2 < 3
        assert len(result.draft.iterations) == 1
        record = result.draft.iterations[0]
        assert record.iteration_index == 2
        # AC #3: each iteration records what changed AND requirements checked.
        assert record.changes_summary.strip()
        assert record.requirements_checked  # non-empty
        assert record.critique_text.strip()

    def test_three_iterations_build_cumulative_history(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
        no_network: None,
    ) -> None:
        """Drive max_iterations=3 end-to-end; history grows by one per call."""

        del no_network
        draft = deterministic_workflow.run(_basic_request())
        for expected_index in (2, 3):
            result = critic_loop.iterate(
                prior_draft=draft,
                request=_basic_request(),
                max_iterations=3,
            )
            assert result.iteration_index == expected_index
            draft = result.draft
        # After 2 critic passes, total_iterations should be 3.
        assert draft.total_iterations() == 3
        # The cumulative history has exactly 2 records (iter 2 and 3).
        assert [r.iteration_index for r in draft.iterations] == [2, 3]
        # Stop flag is set on the last permitted iteration.
        assert result.stopped is True

    def test_client_can_stop_after_iteration_two_with_max_three(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
        no_network: None,
    ) -> None:
        """AC #2: client stops by not calling iterate() again.

        After one critic pass with max_iterations=3, the draft has
        iteration_index=2 and stopped=False; the client may legally
        keep that draft as the final return value.
        """

        del no_network
        prior = deterministic_workflow.run(_basic_request())
        result = critic_loop.iterate(
            prior_draft=prior,
            request=_basic_request(),
            max_iterations=3,
        )
        # The client picks the second iteration as final; no further
        # call is required and the workflow holds no state about the
        # decision.
        assert result.iteration_index == 2
        assert result.draft.total_iterations() == 2
        # No additional state on the workflow (it is stateless).
        assert critic_loop.workflow is deterministic_workflow


class TestRequirementsConsistentAcrossIterations:
    def test_requirements_match_deterministic_construction(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
        no_network: None,
    ) -> None:
        """Every iteration's requirements_checked matches the deterministic builder."""

        del no_network
        prior = deterministic_workflow.run(_basic_request())
        result = critic_loop.iterate(
            prior_draft=prior,
            request=_basic_request(),
            max_iterations=3,
        )
        record = result.draft.iterations[0]
        # No profile, no topic → only the sentinel.
        assert record.requirements_checked == [_DEFAULT_REQUIREMENT_SENTINEL]

    def test_requirements_include_profile_outputs_when_profile_supplied(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
        no_network: None,
    ) -> None:
        """A profile with expected_outputs flows the labels into requirements_checked."""

        del no_network
        profile = DraftingProfile(
            programme=Programme.HORIZON_EUROPE,
            name="Test Profile",
            expected_outputs={
                SectionType.METHODOLOGY: ["validation strategy", "risk register"],
            },
        )
        prior = deterministic_workflow.run(_basic_request(), profile=profile)
        result = critic_loop.iterate(
            prior_draft=prior,
            request=_basic_request(),
            max_iterations=3,
            profile=profile,
        )
        record = result.draft.iterations[0]
        assert "validation strategy" in record.requirements_checked
        assert "risk register" in record.requirements_checked
        assert _DEFAULT_REQUIREMENT_SENTINEL in record.requirements_checked

    def test_requirements_include_topic_section_guidance_keys(
        self,
        critic_loop: CriticLoopWorkflow,
        deterministic_workflow: SectionGenerationWorkflow,
        no_network: None,
    ) -> None:
        """TopicContext keys appear as ``topic:<section>`` entries in the record."""

        del no_network
        topic = TopicContext(
            programme=Programme.HORIZON_EUROPE,
            section_guidance={SectionType.METHODOLOGY: "Use FAIR principles."},
            source=TopicSource.PASTED_TEXT,
            raw_text="",
        )
        request = GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="Describe a methodology.",
            top_k_examples=3,
            topic_context=topic,
        )
        prior = deterministic_workflow.run(request)
        result = critic_loop.iterate(
            prior_draft=prior,
            request=request,
            max_iterations=3,
        )
        record = result.draft.iterations[0]
        assert "topic:methodology" in record.requirements_checked


# ---------------------------------------------------------------------------
# Backward-compat / single-pass invariants
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_workflow_run_still_returns_draft_with_empty_iterations(
        self, deterministic_workflow: SectionGenerationWorkflow
    ) -> None:
        """The Sprint 1 single-pass entry point is unchanged."""

        draft = deterministic_workflow.run(_basic_request())
        assert draft.iterations == []
        assert draft.total_iterations() == 1

    def test_workflow_run_accepts_iteration_count_kwarg(
        self, deterministic_workflow: SectionGenerationWorkflow
    ) -> None:
        """workflow.run gains an ``iteration_count`` knob that defaults to 1."""

        # The kwarg is forward-looking: when the critic loop calls
        # workflow.run(..., iteration_count=N), the produced draft is
        # still a single-pass draft with empty iterations (the loop
        # appends the record), but the analytics emission carries the
        # correct count.
        draft = deterministic_workflow.run(_basic_request(), iteration_count=3)
        assert draft.iterations == []
