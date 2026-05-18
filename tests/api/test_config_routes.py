"""TestClient coverage for the FastAPI config/settings routes (issue #74).

Tests the GET/PUT /api/config endpoints using a tmp-dir offline config.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from eurpe.api import dependencies as deps
from eurpe.api.main import app
from tests._helpers.offline import write_offline_config


@pytest.fixture
def configured_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Yield a TestClient wired to a tmp-dir offline config."""

    monkeypatch.setattr(
        "eurpe.retrieval.embeddings._ollama_reachable",
        lambda *_args, **_kwargs: False,
    )
    cfg_path = write_offline_config(tmp_path)
    deps.set_config_path(cfg_path)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        deps.reset_dependency_caches()


class TestGetConfig:
    """GET /api/config — return effective configuration."""

    def test_returns_config_with_all_fields(self, configured_app: TestClient) -> None:
        resp = configured_app.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "corpus_path" in data
        assert "index_path" in data
        assert "runtime_dir" in data
        assert "offline_mode" in data
        assert "log_level" in data
        assert "models" in data
        assert "network_allowlist" in data

    def test_models_section_populated(self, configured_app: TestClient) -> None:
        resp = configured_app.get("/api/config")
        models = resp.json()["models"]
        assert "runtime" in models
        assert "llm_model" in models
        assert "embedding_model" in models
        assert "ollama_base_url" in models

    def test_network_allowlist_is_list(self, configured_app: TestClient) -> None:
        resp = configured_app.get("/api/config")
        assert isinstance(resp.json()["network_allowlist"], list)


class TestUpdateConfig:
    """PUT /api/config — merge partial config updates."""

    def test_update_log_level(self, configured_app: TestClient) -> None:
        resp = configured_app.put("/api/config", json={"log_level": "DEBUG"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["config"]["log_level"] == "DEBUG"

    def test_update_offline_mode(self, configured_app: TestClient) -> None:
        resp = configured_app.put("/api/config", json={"offline_mode": False})
        assert resp.status_code == 200
        assert resp.json()["config"]["offline_mode"] is False

    def test_update_models_runtime(self, configured_app: TestClient) -> None:
        resp = configured_app.put(
            "/api/config",
            json={"models": {"runtime": "openai", "llm_model": "gpt-4o-mini"}},
        )
        assert resp.status_code == 200
        models = resp.json()["config"]["models"]
        assert models["runtime"] == "openai"
        assert models["llm_model"] == "gpt-4o-mini"

    def test_update_network_allowlist(self, configured_app: TestClient) -> None:
        resp = configured_app.put(
            "/api/config",
            json={
                "network_allowlist": [
                    {
                        "host": "api.openai.com",
                        "port": 443,
                        "reason": "OpenAI generation",
                    }
                ]
            },
        )
        assert resp.status_code == 200
        allowlist = resp.json()["config"]["network_allowlist"]
        assert len(allowlist) == 1
        assert allowlist[0]["host"] == "api.openai.com"

    def test_update_persists_to_yaml(self, configured_app: TestClient, tmp_path: Path) -> None:
        configured_app.put("/api/config", json={"log_level": "WARNING"})
        config_path = deps._CONFIG_PATH
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["log_level"] == "WARNING"

    def test_update_reflected_on_next_get(self, configured_app: TestClient) -> None:
        configured_app.put("/api/config", json={"log_level": "ERROR"})
        resp = configured_app.get("/api/config")
        assert resp.json()["log_level"] == "ERROR"

    def test_update_paths(self, configured_app: TestClient) -> None:
        resp = configured_app.put(
            "/api/config",
            json={
                "corpus_path": "/tmp/new-corpus",
                "index_path": "/tmp/new-index",
                "runtime_dir": "/tmp/new-runtime",
            },
        )
        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["corpus_path"] == "/tmp/new-corpus"
        assert cfg["index_path"] == "/tmp/new-index"
        assert cfg["runtime_dir"] == "/tmp/new-runtime"

    def test_update_rejects_empty_body(self, configured_app: TestClient) -> None:
        resp = configured_app.put("/api/config", json={})
        assert resp.status_code == 400

    def test_update_rejects_invalid_runtime(self, configured_app: TestClient) -> None:
        resp = configured_app.put(
            "/api/config",
            json={"models": {"runtime": "invalid-runtime"}},
        )
        assert resp.status_code == 400

    def test_update_rejects_invalid_log_level(self, configured_app: TestClient) -> None:
        resp = configured_app.put("/api/config", json={"log_level": "VERBOSE"})
        assert resp.status_code == 400

    def test_update_rejects_extra_fields(self, configured_app: TestClient) -> None:
        resp = configured_app.put("/api/config", json={"unknown_field": "value"})
        assert resp.status_code == 422

    def test_update_multiple_fields_at_once(self, configured_app: TestClient) -> None:
        resp = configured_app.put(
            "/api/config",
            json={
                "log_level": "DEBUG",
                "offline_mode": False,
                "models": {"runtime": "gemini", "llm_model": "gemini-1.5-flash"},
            },
        )
        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["log_level"] == "DEBUG"
        assert cfg["offline_mode"] is False
        assert cfg["models"]["runtime"] == "gemini"
