"""FastAPI route modules for the EURPE local API.

Routers live one-per-feature so the surface stays browseable and tests
can import a single router without dragging the whole app in. The
top-level ``eurpe.api.main`` registers each router under its own prefix.
"""

from __future__ import annotations

__all__ = ["ingest"]
