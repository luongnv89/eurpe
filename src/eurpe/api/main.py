"""FastAPI entry point for the local EURPE HTTP API.

The application exposes the local-only HTTP surface consumed by the React
frontend. Endpoints currently registered:

* ``GET /health`` — basic liveness check (no external calls).
* ``POST /api/ingestion/parse`` — upload a proposal PDF, run Docling, get
  back a parse token + draft metadata.
* ``POST /api/ingestion/confirm`` — submit operator-confirmed metadata
  for a parse token, persist a sidecar, and index the chunks.
* ``GET /api/ingestion/enums`` — closed enum vocabularies for the UI
  Selects (programme, source_status).

Run locally with::

    uvicorn eurpe.api.main:app --host 127.0.0.1 --port 8765

Binding to ``127.0.0.1`` keeps the service inside the local machine; do
not expose this to a public interface. The startup hook below logs the
intended invariant so an operator running ``uvicorn`` with a different
``--host`` sees the warning before they get past the splash output.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from eurpe import __version__
from eurpe.api.routes import ingest as ingest_routes

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EURPE Local API",
    version=__version__,
    description="Local-only HTTP API for the EURPE proposal-drafting assistant.",
)

# Register feature routers. Each router carries its own ``prefix`` so the
# top-level app stays free of route-by-route URL knowledge.
app.include_router(ingest_routes.router)


@app.on_event("startup")
def _log_local_only_invariant() -> None:
    """Remind operators the API is local-only on startup.

    Uvicorn prints the bind address itself so we don't try to second-guess
    it here — but the offline-mode + 127.0.0.1 expectation is worth a log
    line so a misconfigured ``--host 0.0.0.0`` still leaves a breadcrumb.
    """

    logger.info(
        "EURPE API starting (version=%s). Expected bind address is 127.0.0.1; "
        "exposing this service to a public interface violates the offline contract.",
        __version__,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return a basic health payload (no external calls)."""
    return {"status": "ok", "version": __version__}
