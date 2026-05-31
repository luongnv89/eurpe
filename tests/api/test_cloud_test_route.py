"""TestClient coverage for the cloud provider test route (issue #80).

Tests verify the API boundary: valid requests return 200 with success/failure
payloads, unsupported providers return 400, and the endpoint delegates to
the cloud_providers module correctly.

The endpoint is self-contained (no FastAPI dependency injection), so no
config fixture is needed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eurpe.api.main import app
from eurpe.generation.cloud_providers import ConnectionTestResult


class TestCloudTestRoute:
    """Verify POST /api/cloud/test endpoint behavior."""

    def test_unsupported_provider_returns_400(self) -> None:
        with TestClient(app) as client:
            resp = client.post(
                "/api/cloud/test",
                json={
                    "provider": "bedrock",
                    "model": "anthropic.claude-v2",
                    "api_key": "sk-xxx",  # pragma: allowlist secret
                },  # pragma: allowlist secret
            )
            assert resp.status_code == 400
            assert "Unsupported provider" in resp.json()["detail"]

    def test_missing_fields_returns_422(self) -> None:
        with TestClient(app) as client:
            resp = client.post(
                "/api/cloud/test",
                json={"provider": "openai"},
            )
            assert resp.status_code == 422

    def test_success_response_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.routes.cloud_test.check_cloud_provider",
            lambda _p, _m, _k: ConnectionTestResult(
                success=True,
                message="Connection successful",
                model_confirmed="gpt-4o",
            ),
        )
        with TestClient(app) as client:
            resp = client.post(
                "/api/cloud/test",
                json={
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-test",  # pragma: allowlist secret
                },  # pragma: allowlist secret
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["message"] == "Connection successful"
            assert data["model_confirmed"] == "gpt-4o"
            assert data["error_detail"] is None

    def test_failure_response_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.routes.cloud_test.check_cloud_provider",
            lambda _p, _m, _k: ConnectionTestResult(
                success=False,
                message="Provider returned HTTP 401",
                error_detail="Invalid API key",
            ),
        )
        with TestClient(app) as client:
            resp = client.post(
                "/api/cloud/test",
                json={
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-bad",  # pragma: allowlist secret
                },  # pragma: allowlist secret
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "401" in data["message"]
            assert data["error_detail"] == "Invalid API key"

    def test_all_supported_providers_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each supported provider key passes validation (returns 200, not 400)."""
        monkeypatch.setattr(
            "eurpe.api.routes.cloud_test.check_cloud_provider",
            lambda _p, _m, _k: ConnectionTestResult(success=True, message="OK"),
        )
        with TestClient(app) as client:
            for provider in ["openai", "anthropic", "gemini", "openrouter", "groq"]:
                resp = client.post(
                    "/api/cloud/test",
                    json={
                        "provider": provider,
                        "model": "test-model",
                        "api_key": "sk-test",  # pragma: allowlist secret
                    },
                )
                assert resp.status_code == 200, f"{provider} should return 200"
