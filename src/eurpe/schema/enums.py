"""Enumerations for EURPE proposal and chunk metadata.

These enums are the *closed* vocabularies that flow from ingestion through
retrieval, generation, and exported citations. The single most important
property of the system is that every retrieved chunk carries an unambiguous
:class:`SourceStatus` so that funded examples (positive patterns) can never be
silently confused with rejected examples (cautionary) or ESR notes (advisory
only). See ``prd.md`` § "Source-status labelling" for the rationale.

All enums are ``str``-valued so they serialize naturally to YAML / JSON and
remain human-readable in fixtures and citation footers.
"""

from __future__ import annotations

from enum import StrEnum


class SourceStatus(StrEnum):
    """The provenance label that travels with every chunk and citation.

    The retriever, generator, and exporter MUST preserve this value end-to-end
    so a coordinator can always tell whether a quoted passage came from a
    funded proposal, a rejected one, an external reviewer note, or an
    unclassified source.
    """

    FUNDED = "funded"
    REJECTED = "rejected"
    ESR_NOTE = "esr_note"  # External Subject Reviewer note — advisory only.
    UNKNOWN = "unknown"


class Programme(StrEnum):
    """EU funding programme that issued the call a proposal targets.

    ``OTHER`` is a deliberate escape hatch for programmes that have not yet
    been first-classed (e.g., national co-funded calls, smaller instruments).
    First-classing a new programme is a one-line addition here.
    """

    HORIZON_EUROPE = "horizon_europe"
    HORIZON_2020 = "horizon_2020"
    DIGITAL_EUROPE = "digital_europe"
    CEF = "cef"  # Connecting Europe Facility.
    OTHER = "other"


class SectionType(StrEnum):
    """Structural section of an EU proposal Part B document.

    The top-level Horizon Europe / Horizon 2020 sections are Excellence,
    Impact, and Implementation; the remaining members are common subsections
    that appear across calls. ``OTHER`` covers prefatory text, annexes, and
    sections that do not match any first-class category.
    """

    EXCELLENCE = "excellence"
    IMPACT = "impact"
    IMPLEMENTATION = "implementation"
    METHODOLOGY = "methodology"
    IMPACT_PATHWAY = "impact_pathway"
    WORK_PLAN = "work_plan"
    CONSORTIUM = "consortium"
    BUDGET = "budget"
    ETHICS = "ethics"
    DISSEMINATION = "dissemination"
    OTHER = "other"
