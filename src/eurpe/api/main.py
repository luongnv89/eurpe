"""FastAPI entry point for the local EURPE HTTP API.

The application is intentionally minimal at this stage — it only exposes a
``/health`` endpoint. Real ingestion, retrieval, and generation endpoints will
be wired in later issues.

Run locally with::

    uvicorn eurpe.api.main:app --host 127.0.0.1 --port 8765

Binding to ``127.0.0.1`` keeps the service inside the local machine; do not
expose this to a public interface.
"""

from __future__ import annotations

from fastapi import FastAPI

from eurpe import __version__

app = FastAPI(
    title="EURPE Local API",
    version=__version__,
    description="Local-only HTTP API for the EURPE proposal-drafting assistant.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a basic health payload (no external calls)."""
    return {"status": "ok", "version": __version__}
