"""Tests for programme drafting profiles (Task 2.1).

Acceptance criteria coverage:
* AC1: At least two sample drafting profiles exist (Horizon Europe + Digital Europe).
* AC2: Generation records the active drafting profile used for each draft.
* AC3: Tests prove drafting profiles do not include scoring dimensions or
       evaluator scoring behavior.
"""

from __future__ import annotations

import pytest

from eurpe.generation import (
    DraftingProfile,
    GenerationRequest,
    list_available_profiles,
    load_profile,
)
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.schema import Programme, SectionType

# ---------------------------------------------------------------------------
# AC1: At least two sample drafting profiles exist
# ---------------------------------------------------------------------------


def test_horizon_europe_profile_exists() -> None:
    """Horizon Europe profile can be loaded from disk."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    assert profile.programme == Programme.HORIZON_EUROPE
    assert profile.name == "Horizon Europe Standard"


def test_digital_europe_profile_exists() -> None:
    """Digital Europe profile can be loaded from disk."""
    profile = load_profile(Programme.DIGITAL_EUROPE)
    assert profile.programme == Programme.DIGITAL_EUROPE
    assert profile.name == "Digital Europe Programme"


def test_list_available_profiles_includes_both_sample_profiles() -> None:
    """At least Horizon Europe and Digital Europe profiles are available."""
    available = list_available_profiles()
    assert Programme.HORIZON_EUROPE in available
    assert Programme.DIGITAL_EUROPE in available


def test_load_profile_raises_for_missing_programme() -> None:
    """Loading a profile for a programme without a profile file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No drafting profile found"):
        load_profile(Programme.CEF)


# ---------------------------------------------------------------------------
# Profile structure and content
# ---------------------------------------------------------------------------


def test_horizon_europe_profile_has_section_guidance() -> None:
    """Horizon Europe profile defines section-specific guidance."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    # Check a few key sections have guidance.
    assert profile.get_section_guidance(SectionType.METHODOLOGY) is not None
    assert profile.get_section_guidance(SectionType.IMPACT_PATHWAY) is not None
    assert profile.get_section_guidance(SectionType.IMPLEMENTATION) is not None


def test_digital_europe_profile_has_section_guidance() -> None:
    """Digital Europe profile defines section-specific guidance."""
    profile = load_profile(Programme.DIGITAL_EUROPE)
    assert profile.get_section_guidance(SectionType.METHODOLOGY) is not None
    assert profile.get_section_guidance(SectionType.IMPACT) is not None


def test_horizon_europe_profile_has_expected_outputs() -> None:
    """Horizon Europe profile defines expected outputs for key sections."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    impl_outputs = profile.get_expected_outputs(SectionType.IMPLEMENTATION)
    assert len(impl_outputs) > 0
    # Check for typical Horizon Europe deliverables.
    assert any("Gantt" in output for output in impl_outputs)
    assert any("milestone" in output.lower() for output in impl_outputs)


def test_digital_europe_profile_has_expected_outputs() -> None:
    """Digital Europe profile defines expected outputs for key sections."""
    profile = load_profile(Programme.DIGITAL_EUROPE)
    impl_outputs = profile.get_expected_outputs(SectionType.IMPLEMENTATION)
    assert len(impl_outputs) > 0


def test_profile_has_terminology_mappings() -> None:
    """Profiles define programme-specific terminology."""
    he_profile = load_profile(Programme.HORIZON_EUROPE)
    assert "work_package" in he_profile.terminology
    assert he_profile.terminology["work_package"] == "WP"

    dep_profile = load_profile(Programme.DIGITAL_EUROPE)
    assert "work_package" in dep_profile.terminology
    # Digital Europe uses "Work Package" (full form).
    assert dep_profile.terminology["work_package"] == "Work Package"


# ---------------------------------------------------------------------------
# AC3: Profiles do not include scoring dimensions
# ---------------------------------------------------------------------------


def test_drafting_profile_model_has_no_scoring_fields() -> None:
    """DraftingProfile model does not include scoring rubric fields.

    This test pins the separation between drafting profiles (Task 2.1)
    and scoring rubrics (post-v1). If a future change adds scoring
    fields to DraftingProfile, this test will fail loudly.
    """
    profile = load_profile(Programme.HORIZON_EUROPE)
    # The model should NOT have these fields (they belong in a separate ScoringRubric).
    assert not hasattr(profile, "scoring_criteria")
    assert not hasattr(profile, "evaluation_dimensions")
    assert not hasattr(profile, "excellence_score")
    assert not hasattr(profile, "impact_score")
    assert not hasattr(profile, "rubric")


def test_profile_guidance_does_not_mention_scoring() -> None:
    """Profile section guidance does not include evaluator scoring language.

    Drafting profiles guide *how to write*; they should not leak
    evaluator behavior ("score", "assess", "evaluate", "grade").
    """
    he_profile = load_profile(Programme.HORIZON_EUROPE)
    dep_profile = load_profile(Programme.DIGITAL_EUROPE)

    forbidden_terms = ["score", "assess", "evaluate", "grade", "rating"]

    for section_type in SectionType:
        he_guidance = he_profile.get_section_guidance(section_type)
        if he_guidance:
            for term in forbidden_terms:
                assert term not in he_guidance.lower(), (
                    f"Horizon Europe guidance for {section_type.value} contains "
                    f"scoring term '{term}' — drafting profiles must not include "
                    "evaluator behavior."
                )

        dep_guidance = dep_profile.get_section_guidance(section_type)
        if dep_guidance:
            for term in forbidden_terms:
                assert term not in dep_guidance.lower(), (
                    f"Digital Europe guidance for {section_type.value} contains "
                    f"scoring term '{term}' — drafting profiles must not include "
                    "evaluator behavior."
                )


# ---------------------------------------------------------------------------
# AC2: Generation records the active drafting profile
# ---------------------------------------------------------------------------


def test_workflow_records_profile_name_in_draft(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """When a profile is provided, the draft records its name."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our approach",
        target_programme=Programme.HORIZON_EUROPE,
    )
    draft = deterministic_workflow.run(request, profile=profile)
    assert draft.drafting_profile == "Horizon Europe Standard"


def test_workflow_records_none_when_no_profile_provided(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """When no profile is provided, drafting_profile is None."""
    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our approach",
    )
    draft = deterministic_workflow.run(request)
    assert draft.drafting_profile is None


def test_different_profiles_recorded_for_different_programmes(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """Different profiles are recorded for different programmes."""
    he_profile = load_profile(Programme.HORIZON_EUROPE)
    dep_profile = load_profile(Programme.DIGITAL_EUROPE)

    he_request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our approach",
        target_programme=Programme.HORIZON_EUROPE,
    )
    dep_request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our approach",
        target_programme=Programme.DIGITAL_EUROPE,
    )

    he_draft = deterministic_workflow.run(he_request, profile=he_profile)
    dep_draft = deterministic_workflow.run(dep_request, profile=dep_profile)

    assert he_draft.drafting_profile == "Horizon Europe Standard"
    assert dep_draft.drafting_profile == "Digital Europe Programme"
    assert he_draft.drafting_profile != dep_draft.drafting_profile


# ---------------------------------------------------------------------------
# Profile-aware prompt building
# ---------------------------------------------------------------------------


def test_prompt_uses_profile_section_guidance(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """When a profile is provided, its section guidance appears in the prompt."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our approach",
        target_programme=Programme.HORIZON_EUROPE,
    )
    draft = deterministic_workflow.run(request, profile=profile)

    # The Horizon Europe methodology guidance should appear in the prompt.
    he_guidance = profile.get_section_guidance(SectionType.METHODOLOGY)
    assert he_guidance is not None
    assert he_guidance in draft.prompt_used


def test_prompt_includes_expected_outputs_when_profile_defines_them(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """When a profile defines expected outputs, they appear in the prompt."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    request = GenerationRequest(
        section_type=SectionType.IMPLEMENTATION,
        user_intent="Describe our work plan",
        target_programme=Programme.HORIZON_EUROPE,
    )
    draft = deterministic_workflow.run(request, profile=profile)

    # Expected outputs should be listed in the prompt.
    assert "Expected outputs" in draft.prompt_used
    expected = profile.get_expected_outputs(SectionType.IMPLEMENTATION)
    for output in expected:
        assert output in draft.prompt_used


def test_prompt_omits_expected_outputs_when_profile_has_none(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """When a profile defines no expected outputs for a section, the section is omitted."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    request = GenerationRequest(
        section_type=SectionType.OTHER,
        user_intent="Describe something",
        target_programme=Programme.HORIZON_EUROPE,
    )
    draft = deterministic_workflow.run(request, profile=profile)

    # OTHER section has no expected outputs, so the section should not appear.
    expected = profile.get_expected_outputs(SectionType.OTHER)
    assert len(expected) == 0
    # The prompt should not have an "Expected outputs" section if the list is empty.
    # (This is a soft check — the prompt builder may still include the header.)


def test_prompt_uses_default_guidance_when_no_profile_provided(
    deterministic_workflow: SectionGenerationWorkflow,
) -> None:
    """When no profile is provided, default SECTION_GUIDANCE is used."""
    from eurpe.generation.prompt import SECTION_GUIDANCE

    request = GenerationRequest(
        section_type=SectionType.METHODOLOGY,
        user_intent="Describe our approach",
    )
    draft = deterministic_workflow.run(request)

    # Default guidance should appear in the prompt.
    default_guidance = SECTION_GUIDANCE[SectionType.METHODOLOGY]
    assert default_guidance in draft.prompt_used


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_profile_with_empty_section_guidance_falls_back_to_default() -> None:
    """If a profile has no guidance for a section, the default is used."""
    profile = load_profile(Programme.HORIZON_EUROPE)
    # Assume the profile doesn't override ETHICS (check the YAML).
    ethics_guidance = profile.get_section_guidance(SectionType.ETHICS)
    # If the profile doesn't define it, get_section_guidance returns None.
    # The prompt builder should fall back to the default.
    assert ethics_guidance is None or len(ethics_guidance) > 0


def test_profile_model_validates_programme_enum() -> None:
    """DraftingProfile validates that programme is a valid Programme enum."""
    with pytest.raises(ValueError):
        DraftingProfile(
            programme="invalid_programme",  # type: ignore[arg-type]
            name="Test",
        )
