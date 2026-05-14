"""End-to-end test suite for EURPE.

Each test in this package drives the full ingest -> retrieve -> generate
-> export pipeline against a real PDF (either a confidential proposal
from ``proposals/`` on a developer machine, or a synthetic fixture PDF
generated on demand by :mod:`tests.e2e.conftest` so CI runs without
any committed PDFs).

These tests are slow and require ``docling`` plus ``reportlab`` (the
latter only for the synthetic fixture). They carry the ``e2e`` pytest
marker so the fast unit suite — ``pytest -m 'not e2e'`` — skips them.
"""

from __future__ import annotations
