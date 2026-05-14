"""Shared helpers for EURPE unit + E2E tests.

This package centralises three concerns that previously lived inlined in
individual test modules:

* :mod:`tests._helpers.offline` — the canonical offline ``config.yaml``
  writer used by every CLI test (Ollama pinned to an unreachable port so
  the factory falls back to the deterministic stub).
* :mod:`tests._helpers.metadata` — proposal-metadata YAML synthesis,
  including a ``load_or_synthesise_metadata`` helper that prefers a
  sibling ``<stem>.yml`` / ``<stem>.yaml`` when present.
* :mod:`tests._helpers.pipeline` — a ``run_full_pipeline`` orchestrator
  used by the E2E suite to drive the five CLI steps in order and return
  a structured artefact dict.

Keeping these in a single sub-package means a future refactor of the
offline contract (e.g., switching to a different unreachable port or
adding a knob) only needs to touch one place.
"""

from __future__ import annotations
