"""MVP pilot validation package (Task 3.7 / issue #21).

Public entry points
-------------------
* :func:`run_pilot` — drive one pilot run end-to-end and return a
  :class:`PilotReport`.
* :func:`render_pilot_report_markdown` — render a
  :class:`PilotReport` as a deterministic Markdown summary suitable
  for committing under ``release-notes/pilots/<release-tag>.md``.
* :func:`attach_satisfaction` — post-edit a smoke-mode pilot report
  to add a coordinator's satisfaction rating, returning a new
  :class:`PilotReport` with the verdict re-computed.
* :func:`load_pilot_report` — parse a saved pilot report JSON file.

Why a package (not a single module)
-----------------------------------
The pilot composes three existing reports (citation audit, benchmark
report, smoke result) into one cohesive deliverable. Splitting the
package into :mod:`eurpe.pilot.models`, :mod:`eurpe.pilot.runner`,
and :mod:`eurpe.pilot.cli` mirrors the convention used elsewhere in
the codebase (see :mod:`eurpe.benchmarks`,
:mod:`eurpe.generation`): models hold the data classes, runner holds
the orchestration, cli holds the Typer wiring. The package exports
re-export the symbols a normal caller cares about so
``from eurpe.pilot import run_pilot`` works without an explicit
module path.
"""

from __future__ import annotations

from eurpe.pilot.models import (
    CitationIssue,
    GoNoGoVerdict,
    PilotMode,
    PilotReport,
    PilotSectionResult,
    SatisfactionRating,
    SmokeResult,
)
from eurpe.pilot.runner import (
    DEFAULT_SECTION_TYPES,
    PilotConfig,
    PilotRunError,
    attach_satisfaction,
    load_pilot_report,
    render_pilot_report_markdown,
    run_pilot,
)

__all__ = [
    "DEFAULT_SECTION_TYPES",
    "CitationIssue",
    "GoNoGoVerdict",
    "PilotConfig",
    "PilotMode",
    "PilotReport",
    "PilotRunError",
    "PilotSectionResult",
    "SatisfactionRating",
    "SmokeResult",
    "attach_satisfaction",
    "load_pilot_report",
    "render_pilot_report_markdown",
    "run_pilot",
]
