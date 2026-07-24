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
* ``GET /api/generation/enums`` — closed enum vocabularies for the
  drafting workspace Selects (section_type, programme).
* ``GET /api/generation/profiles`` — list of available drafting
  profiles bundled with this build.
* ``POST /api/generation/section`` — draft one proposal section using
  indexed past-proposal evidence; returns the rendered text plus
  citations with provenance.

Run locally with::

    uvicorn eurpe.api.main:app --host 127.0.0.1 --port 8765

Binding to ``127.0.0.1`` keeps the service inside the local machine; do
not expose this to a public interface. The lifespan context below logs
the intended invariant so an operator running ``uvicorn`` with a different
``--host`` sees the warning before they get past the splash output.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from eurpe import __version__
from eurpe.api.dependencies import close_llm_clients
from eurpe.api.routes import cloud_test as cloud_test_routes
from eurpe.api.routes import config as config_routes
from eurpe.api.routes import generate as generate_routes
from eurpe.api.routes import ingest as ingest_routes
from eurpe.api.routes import runtime as runtime_routes
from eurpe.generation import GenerationError, LLMUnavailableError
from eurpe.security import SecurityError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log the local-only invariant on startup; close pooled clients on shutdown.

    Uvicorn prints the bind address itself so we don't try to second-guess
    it here — but the offline-mode + 127.0.0.1 expectation is worth a log
    line so a misconfigured ``--host 0.0.0.0`` still leaves a breadcrumb.
    """

    logger.info(
        "EURPE API starting (version=%s). Expected bind address is 127.0.0.1; "
        "exposing this service to a public interface violates the offline contract.",
        __version__,
    )
    yield
    # The cached LLM clients' pooled httpx.Client connections would
    # otherwise only close at process exit (see close_llm_clients).
    close_llm_clients()


app = FastAPI(
    title="EURPE Local API",
    version=__version__,
    description="Local-only HTTP API for the EURPE proposal-drafting assistant.",
    lifespan=_lifespan,
)


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable_handler(
    _request: Request,
    exc: LLMUnavailableError,
) -> JSONResponse:
    """Map LLM setup/runtime unavailability even when raised by dependencies."""

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"LLM unavailable: {exc}"},
    )


@app.exception_handler(SecurityError)
async def _security_error_handler(
    _request: Request,
    exc: SecurityError,
) -> JSONResponse:
    """Map network-policy denials raised before a route handler starts."""

    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": f"network policy denied generation: {exc}"},
    )


@app.exception_handler(GenerationError)
async def _generation_error_handler(
    _request: Request,
    exc: GenerationError,
) -> JSONResponse:
    """Map generation failures that escape route-local handlers."""

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"generation failed: {exc}"},
    )


# Register feature routers. Each router carries its own ``prefix`` so the
# top-level app stays free of route-by-route URL knowledge.
app.include_router(ingest_routes.router)
app.include_router(generate_routes.router)
app.include_router(runtime_routes.router)
app.include_router(cloud_test_routes.router)
app.include_router(config_routes.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a basic health payload (no external calls)."""
    return {"status": "ok", "version": __version__}
