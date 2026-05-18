"""HTTP route for testing cloud provider connections (issue #80).

One endpoint:

* ``POST /api/cloud/test`` — send a minimal request to a cloud LLM
  provider (OpenAI, Anthropic, Gemini, OpenRouter, Groq) to verify
  that the supplied API key is valid and the model is accessible.

The test uses ``max_tokens=1`` with a trivial prompt so token
consumption is negligible. The endpoint is offline-safe: it only
fires when the operator explicitly clicks "Test Connection" in the
Settings UI.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from eurpe.api.schemas import CloudProviderTestRequest, CloudProviderTestResponse
from eurpe.generation.cloud_providers import (
    SUPPORTED_PROVIDERS,
    check_cloud_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cloud", tags=["cloud"])


@router.post("/test", response_model=CloudProviderTestResponse)
def test_cloud_connection(body: CloudProviderTestRequest) -> CloudProviderTestResponse:
    """Test a cloud provider connection with the supplied credentials.

    Sends a minimal request (``max_tokens=1``) to verify the API key
    is valid and the model is accessible. Returns success/failure with
    provider-specific error details on failure.

    Error mapping
    -------------
    * Unknown provider → ``400 Bad Request``.
    * Provider returns 401/403 (invalid key) → ``200`` with
      ``success=false`` and the error detail (the operator can fix this).
    * Provider returns 404 (unknown model) → ``200`` with
      ``success=false`` and the error detail.
    * Network timeout / DNS failure → ``200`` with ``success=false``.
    * Unexpected server error → ``500 Internal Server Error``.
    """
    if body.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported provider: {body.provider!r}. "
                f"Must be one of {sorted(SUPPORTED_PROVIDERS)}."
            ),
        )

    try:
        result = check_cloud_provider(body.provider, body.model, body.api_key)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error testing cloud provider %s", body.provider)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error testing provider: {exc}",
        ) from exc

    return CloudProviderTestResponse(
        success=result.success,
        message=result.message,
        model_confirmed=result.model_confirmed,
        error_detail=result.error_detail,
    )
