"""Example: Using programme drafting profiles.

This example demonstrates how to load and use drafting profiles to generate
programme-specific proposal sections.
"""

from eurpe.generation import (
    GenerationRequest,
    list_available_profiles,
    load_profile,
)
from eurpe.schema import Programme, SectionType

# List available profiles
print("Available drafting profiles:")
for programme in list_available_profiles():
    print(f"  - {programme.value}")

# Load a specific profile
he_profile = load_profile(Programme.HORIZON_EUROPE)
print(f"\nLoaded profile: {he_profile.name}")
print(f"Programme: {he_profile.programme.value}")

# Inspect profile contents
print("\nTerminology mappings:")
for key, value in he_profile.terminology.items():
    print(f"  {key}: {value}")

# Check section guidance
methodology_guidance = he_profile.get_section_guidance(SectionType.METHODOLOGY)
print("\nMethodology guidance (first 100 chars):")
print(f"  {methodology_guidance[:100]}..." if methodology_guidance else "  (none)")

# Check expected outputs
impl_outputs = he_profile.get_expected_outputs(SectionType.IMPLEMENTATION)
print("\nImplementation expected outputs:")
for output in impl_outputs[:3]:  # Show first 3
    print(f"  - {output}")

# Using a profile with the workflow
# (This is a conceptual example - in practice you'd have a configured workflow)
print("\n--- Using profile with workflow ---")

# Create a request
request = GenerationRequest(
    section_type=SectionType.METHODOLOGY,
    user_intent="Describe our AI-based approach for cybersecurity",
    target_programme=Programme.HORIZON_EUROPE,
)

# In a real scenario, you'd pass the profile to workflow.run():
# workflow = SectionGenerationWorkflow(retriever=..., llm=...)
# draft = workflow.run(request, profile=he_profile)
# print(f"Draft generated with profile: {draft.drafting_profile}")

print(f"\nRequest created for {request.section_type.value} section")
print(f"Profile to use: {he_profile.name}")
print("\nThe workflow will:")
print("  1. Use programme-specific section guidance")
print("  2. Include expected outputs in the prompt")
print("  3. Record the profile name in the draft for audit")

# Compare with Digital Europe profile
print("\n--- Comparing profiles ---")
dep_profile = load_profile(Programme.DIGITAL_EUROPE)
print(f"\nHorizon Europe work_package term: {he_profile.terminology['work_package']}")
print(f"Digital Europe work_package term: {dep_profile.terminology['work_package']}")
print("\nThis shows how profiles adapt terminology to programme conventions.")
