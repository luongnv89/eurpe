"""Programme drafting profiles for section generation.

Drafting profiles provide programme-specific terminology, section guidance,
expected-output fields, and source-labeling rules. These profiles are
**separate from scoring rubrics** — they guide the drafting process but do
not include evaluator scoring dimensions or criteria.

Why profiles are separate from scoring rubrics
----------------------------------------------
The PRD distinguishes "drafting profiles" (Sprint 2, Task 2.1) from "scoring
rubrics" (post-v1). Drafting profiles tell the generator *how to write* for
a given programme (terminology, structure, citation style). Scoring rubrics
tell an evaluator *how to assess* a draft (excellence criteria, impact
dimensions, scoring scales). Conflating the two would leak evaluator behavior
into the drafting prompt, which violates the separation of concerns the PRD
establishes.

Profile structure
-----------------
Each :class:`DraftingProfile` carries:

* **terminology**: Programme-specific vocabulary (e.g., "Work Package" vs
  "Task", "TRL" vs "Maturity Level").
* **section_guidance**: Overrides for :data:`~eurpe.generation.prompt.SECTION_GUIDANCE`
  that reflect programme-specific section structures.
* **expected_outputs**: Fields the programme expects in each section (e.g.,
  "Gantt chart", "risk register", "dissemination plan").
* **source_label_style**: How to frame citations (e.g., "Funded example" vs
  "Successful proposal").

The profile is recorded in :class:`~eurpe.generation.GenerationDraft` so an
auditor can trace which profile shaped a given draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from eurpe.schema import Programme, SectionType

# Default profiles directory: src/eurpe/generation/profiles/
_PROFILES_DIR = Path(__file__).parent / "profiles"


class DraftingProfile(BaseModel):
    """Programme-specific drafting guidance for section generation.

    This model is intentionally **not** a scoring rubric. It does not
    include evaluator criteria, scoring scales, or assessment dimensions.
    Those belong in a separate ``ScoringRubric`` model (post-v1).
    """

    programme: Programme = Field(
        description="EU programme this profile applies to.",
    )
    name: str = Field(
        min_length=1,
        description="Human-readable profile name (e.g., 'Horizon Europe Standard').",
    )
    terminology: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Programme-specific vocabulary. Keys are generic terms, values "
            "are programme-preferred terms. Example: {'work_package': 'WP', "
            "'technology_readiness': 'TRL'}."
        ),
    )
    section_guidance: dict[SectionType, str] = Field(
        default_factory=dict,
        description=(
            "Section-specific guidance text. Overrides the default "
            "SECTION_GUIDANCE for sections where this programme has "
            "different structural expectations."
        ),
    )
    expected_outputs: dict[SectionType, list[str]] = Field(
        default_factory=dict,
        description=(
            "Expected deliverables per section. Example: "
            "{SectionType.WORK_PLAN: ['Gantt chart', 'Milestone table', "
            "'Risk register']}."
        ),
    )
    source_label_style: str = Field(
        default="standard",
        description=(
            "Citation framing style. 'standard' uses 'FUNDED'/'REJECTED'; "
            "future profiles may use 'successful'/'unsuccessful' or other "
            "programme-specific labels."
        ),
    )

    def get_section_guidance(self, section_type: SectionType) -> str | None:
        """Return programme-specific guidance for ``section_type``, or None if not overridden."""
        return self.section_guidance.get(section_type)

    def get_expected_outputs(self, section_type: SectionType) -> list[str]:
        """Return expected outputs for ``section_type``, or empty list if none defined."""
        return self.expected_outputs.get(section_type, [])


def load_profile(programme: Programme, profiles_dir: Path = _PROFILES_DIR) -> DraftingProfile:
    """Load the drafting profile for ``programme`` from disk.

    Profiles are stored as YAML files in ``profiles_dir`` with filenames
    matching the programme enum value (e.g., ``horizon_europe.yaml``).

    Raises:
        FileNotFoundError: If no profile exists for the given programme.
        ValueError: If the profile file is malformed or validation fails.
    """
    profile_path = profiles_dir / f"{programme.value}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"No drafting profile found for {programme.value} at {profile_path}. "
            "Available profiles must be created in src/eurpe/generation/profiles/."
        )

    with profile_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Profile file {profile_path} must contain a YAML mapping, "
            f"got {type(raw).__name__}."
        )

    # Pydantic will validate the structure and coerce enum strings.
    return DraftingProfile.model_validate(raw)


def list_available_profiles(profiles_dir: Path = _PROFILES_DIR) -> list[Programme]:
    """Return a list of programmes for which drafting profiles exist on disk."""
    if not profiles_dir.exists():
        return []

    available: list[Programme] = []
    for profile_file in profiles_dir.glob("*.yaml"):
        stem = profile_file.stem
        try:
            programme = Programme(stem)
            available.append(programme)
        except ValueError:
            # Ignore files that don't match a known programme enum.
            continue

    return sorted(available, key=lambda p: p.value)
