"""HTTP routes for local runtime health and model listing (issue #79).

Three endpoints back the Settings page runtime detection:

* ``GET /api/runtime/status`` — health status of the *currently
  configured* runtime (from ``config.yaml``). Returns availability,
  endpoint, and the list of installed models.
* ``GET /api/runtime/all`` — status of *all* supported runtimes so the
  UI can let the user switch and see availability at a glance.
* ``GET /api/runtime/instructions/{runtime}`` — step-by-step
  installation instructions for a specific runtime.

Issue #81: two additional endpoints for testing local models and
embeddings:

* ``POST /api/runtime/test-model`` — send a minimal generation request
  to verify a local LLM model is loaded and responding.
* ``POST /api/runtime/test-embedding`` — send a minimal embedding
  request to verify an embedding model can produce vectors.

Error mapping
-------------
* Unknown runtime key → ``400 Bad Request``.
* All probes are local-only (``localhost``); no outbound Internet
  traffic is generated, consistent with the offline-first invariant.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from eurpe.api.runtime_probe import (
    RUNTIME_REGISTRY,
    check_local_embedding,
    check_local_model_generation,
    get_install_instructions,
    probe_runtime,
)
from eurpe.api.schemas import (
    AllRuntimesResponse,
    LocalEmbeddingTestRequest,
    LocalModelTestRequest,
    LocalModelTestResponse,
    RuntimeInstructionsResponse,
    RuntimeStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.get("/status", response_model=RuntimeStatusResponse)
def get_runtime_status() -> RuntimeStatusResponse:
    """Return health status of the currently configured runtime.

    Reads ``config.yaml`` to determine which runtime is active, probes
    its default endpoint, and returns availability plus the model list.
    """
    from eurpe.config import load_config

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load configuration: {exc}",
        ) from exc

    runtime_key = config.models.runtime
    base_url = config.models.ollama_base_url if runtime_key == "ollama" else None

    result = probe_runtime(runtime_key, base_url=base_url)
    info = RUNTIME_REGISTRY.get(runtime_key, {})

    return RuntimeStatusResponse(
        runtime=runtime_key,
        display_name=info.get("display_name", runtime_key),
        endpoint=result.endpoint,
        available=result.available,
        models=result.models,
        error=result.error,
    )


@router.get("/all", response_model=AllRuntimesResponse)
def get_all_runtimes() -> AllRuntimesResponse:
    """Return status of all supported local runtimes.

    Probes every known runtime's default endpoint so the Settings page
    can show a comparison view.
    """
    from eurpe.config import load_config

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load configuration: {exc}",
        ) from exc

    active_runtime = config.models.runtime
    statuses: list[RuntimeStatusResponse] = []

    for runtime_key, info in RUNTIME_REGISTRY.items():
        # Use the configured URL for the active runtime, default for others
        base_url = None
        if runtime_key == active_runtime and runtime_key == "ollama":
            base_url = config.models.ollama_base_url

        result = probe_runtime(runtime_key, base_url=base_url)
        statuses.append(
            RuntimeStatusResponse(
                runtime=runtime_key,
                display_name=info["display_name"],
                endpoint=result.endpoint,
                available=result.available,
                models=result.models,
                error=result.error,
            )
        )

    return AllRuntimesResponse(runtimes=statuses, active_runtime=active_runtime)


@router.get("/instructions/{runtime}", response_model=RuntimeInstructionsResponse)
def get_runtime_instructions(runtime: str) -> RuntimeInstructionsResponse:
    """Return installation instructions for a specific runtime.

    The ``runtime`` path parameter must be one of the known runtime keys
    (``ollama``, ``mlx``, ``vllm``).
    """
    if runtime not in RUNTIME_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown runtime: {runtime}. Must be one of {sorted(RUNTIME_REGISTRY.keys())}",
        )

    instructions = get_install_instructions(runtime)
    return RuntimeInstructionsResponse(
        instructions=instructions,
    )


# ---------------------------------------------------------------------------
# Local model and embedding test (issue #81)
# ---------------------------------------------------------------------------


@router.post("/test-model", response_model=LocalModelTestResponse)
def test_model(body: LocalModelTestRequest) -> LocalModelTestResponse:
    """Test a local LLM model with a minimal generation request.

    Sends a trivial prompt to verify the selected model is loaded and
    can produce output. Returns success/failure with error details on
    failure.

    Error mapping
    -------------
    * Unknown runtime → ``400 Bad Request``.
    * Runtime unreachable → ``200`` with ``success=false``.
    * Model not loaded / generation error → ``200`` with ``success=false``.
    """
    if body.runtime not in RUNTIME_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported runtime: {body.runtime!r}. "
                f"Must be one of {sorted(RUNTIME_REGISTRY.keys())}."
            ),
        )

    result = check_local_model_generation(
        runtime=body.runtime,
        model=body.model,
        base_url=body.base_url,
    )
    return LocalModelTestResponse(
        success=result["success"],
        message=result["message"],
        error_detail=result.get("error_detail"),
    )


@router.post("/test-embedding", response_model=LocalModelTestResponse)
def test_embedding(body: LocalEmbeddingTestRequest) -> LocalModelTestResponse:
    """Test a local embedding model with a minimal embedding request.

    Sends a trivial text to verify the embedding model is loaded and
    can produce vectors. Returns success/failure with error details on
    failure.

    Error mapping
    -------------
    * Unknown runtime → ``400 Bad Request``.
    * Runtime unreachable → ``200`` with ``success=false``.
    * Model not loaded / embedding error → ``200`` with ``success=false``.
    """
    if body.runtime not in RUNTIME_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported runtime: {body.runtime!r}. "
                f"Must be one of {sorted(RUNTIME_REGISTRY.keys())}."
            ),
        )

    result = check_local_embedding(
        runtime=body.runtime,
        model=body.model,
        base_url=body.base_url,
    )
    return LocalModelTestResponse(
        success=result["success"],
        message=result["message"],
        error_detail=result.get("error_detail"),
    )
