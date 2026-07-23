"""Tests for the local runtime health probe module (issue #79).

These tests verify the probe logic without requiring real runtimes.
The TCP probes are monkeypatched to simulate reachable/unreachable
states, and the HTTP-level probe functions are tested via
``respx``-style mocking of the inner helper.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from eurpe.api.runtime_probe import (
    RUNTIME_REGISTRY,
    RuntimeStatus,
    check_local_embedding,
    check_local_model_generation,
    get_install_instructions,
    probe_runtime,
)


class TestProbeRuntime:
    """Verify probe_runtime dispatches correctly and handles failures."""

    def test_unknown_runtime(self) -> None:
        result = probe_runtime("nonexistent")
        assert not result.available
        assert "Unknown runtime" in (result.error or "")

    def test_ollama_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: False,
        )
        result = probe_runtime("ollama")
        assert not result.available
        assert result.name == "Ollama"
        assert result.endpoint == "http://localhost:11434"
        assert result.models == []
        assert "Cannot reach Ollama" in (result.error or "")

    def test_ollama_reachable_with_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )
        # Replace the entire _probe_ollama function with a stub
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._probe_ollama",
            lambda _url: RuntimeStatus(
                name="Ollama",
                endpoint="http://localhost:11434",
                available=True,
                models=["llama3.1:8b", "nomic-embed-text"],
            ),
        )
        result = probe_runtime("ollama")
        assert result.available
        assert result.models == ["llama3.1:8b", "nomic-embed-text"]

    def test_ollama_reachable_no_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._probe_ollama",
            lambda _url: RuntimeStatus(
                name="Ollama",
                endpoint="http://localhost:11434",
                available=True,
                models=[],
            ),
        )
        result = probe_runtime("ollama")
        assert result.available
        assert result.models == []

    def test_ollama_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._probe_ollama",
            lambda _url: RuntimeStatus(
                name="Ollama",
                endpoint="http://localhost:11434",
                available=False,
                error="Ollama returned HTTP 500",
            ),
        )
        result = probe_runtime("ollama")
        assert not result.available
        assert "HTTP 500" in (result.error or "")

    def test_vllm_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: False,
        )
        result = probe_runtime("vllm")
        assert not result.available
        assert result.name == "vLLM"
        assert "Cannot reach vLLM" in (result.error or "")

    def test_mlx_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: False,
        )
        result = probe_runtime("mlx")
        assert not result.available
        assert result.name == "MLX (Apple Silicon)"
        assert "Cannot reach MLX" in (result.error or "")

    def test_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: False,
        )
        result = probe_runtime("ollama", base_url="http://127.0.0.1:9999")
        assert result.endpoint == "http://127.0.0.1:9999"


class TestGetInstallInstructions:
    """Verify installation instructions are returned for known runtimes."""

    def test_ollama_instructions(self) -> None:
        inst = get_install_instructions("ollama")
        assert inst["title"] == "Ollama"
        assert "ollama.com/download" in inst["steps"]
        assert "ollama pull" in inst["steps"]
        assert inst["docs_url"] == "https://ollama.com/download"

    def test_vllm_instructions(self) -> None:
        inst = get_install_instructions("vllm")
        assert inst["title"] == "vLLM"
        assert "pip install vllm" in inst["steps"]
        assert inst["docs_url"] == "https://docs.vllm.ai/en/latest/"

    def test_mlx_instructions(self) -> None:
        inst = get_install_instructions("mlx")
        assert inst["title"] == "MLX (Apple Silicon)"
        assert "pip install mlx mlx-lm" in inst["steps"]
        assert inst["docs_url"] == "https://ml-explore.github.io/mlx/"

    def test_unknown_runtime(self) -> None:
        inst = get_install_instructions("nonexistent")
        assert inst["title"] == "nonexistent"
        assert "No installation instructions" in inst["steps"]


class TestRuntimeRegistry:
    """Verify the registry contains all expected runtimes."""

    def test_known_runtimes(self) -> None:
        assert "ollama" in RUNTIME_REGISTRY
        assert "mlx" in RUNTIME_REGISTRY
        assert "vllm" in RUNTIME_REGISTRY

    def test_registry_entries_have_required_fields(self) -> None:
        for key, info in RUNTIME_REGISTRY.items():
            assert "display_name" in info, f"{key} missing display_name"
            assert "default_url" in info, f"{key} missing default_url"
            assert "docs_url" in info, f"{key} missing docs_url"


class TestLocalModelGeneration:
    """Verify check_local_model_generation handles various scenarios."""

    def test_unknown_runtime(self) -> None:
        result = check_local_model_generation("nonexistent", "some-model")
        assert not result["success"]
        assert "Unknown runtime" in result["message"]

    def test_runtime_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: False,
        )
        result = check_local_model_generation("ollama", "llama3.1:8b")
        assert not result["success"]
        assert "not reachable" in result["message"]

    def test_ollama_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )

        class FakeResponse:
            status_code = 200

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, json):
                return FakeResponse()

        monkeypatch.setattr("httpx.Client", lambda **kwargs: FakeClient())
        result = check_local_model_generation("ollama", "llama3.1:8b")
        assert result["success"]
        assert "loaded and responding" in result["message"]

    def test_vllm_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )

        class FakeResponse:
            status_code = 200

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, json):
                return FakeResponse()

        monkeypatch.setattr("httpx.Client", lambda **kwargs: FakeClient())
        result = check_local_model_generation("vllm", "meta-llama/Llama-3.1-8B-Instruct")
        assert result["success"]


class TestLocalEmbedding:
    """Verify check_local_embedding handles various scenarios."""

    def test_unknown_runtime(self) -> None:
        result = check_local_embedding("nonexistent", "nomic-embed-text")
        assert not result["success"]
        assert "Unknown runtime" in result["message"]

    def test_runtime_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: False,
        )
        result = check_local_embedding("ollama", "nomic-embed-text")
        assert not result["success"]
        assert "not reachable" in result["message"]

    def test_ollama_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"embedding": [0.1] * 768}

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, json):
                return FakeResponse()

        monkeypatch.setattr("httpx.Client", lambda **kwargs: FakeClient())
        result = check_local_embedding("ollama", "nomic-embed-text")
        assert result["success"]
        assert "768-dimension" in result["message"]

    def test_ollama_malformed_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eurpe.api.runtime_probe._tcp_probe",
            lambda *_args, **_kwargs: True,
        )

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"no_embedding": True}

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, json):
                return FakeResponse()

        monkeypatch.setattr("httpx.Client", lambda **kwargs: FakeClient())
        result = check_local_embedding("ollama", "nomic-embed-text")
        assert not result["success"]
        assert "unexpected response" in result["message"]


class TestGetAllRuntimesConcurrency:
    """``GET /api/runtime/all`` probes every runtime through a ThreadPoolExecutor.

    Regression coverage for the perf fix: probing used to be sequential,
    so total latency was the SUM of every runtime's probe time. These
    tests make each probe artificially slow and assert the wall-clock
    stays near the MAX (concurrent), not the SUM (serial).
    """

    def test_probes_run_concurrently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from eurpe.api.routes.runtime import get_all_runtimes

        probe_delay = 0.15
        call_count = 0

        def fake_probe_runtime(runtime_key: str, *, base_url: str | None = None) -> RuntimeStatus:
            nonlocal call_count
            call_count += 1
            time.sleep(probe_delay)
            info = RUNTIME_REGISTRY[runtime_key]
            return RuntimeStatus(
                name=info["display_name"],
                endpoint=info["default_url"],
                available=False,
                error="unreachable in test",
            )

        monkeypatch.setattr(
            "eurpe.config.load_config",
            lambda: SimpleNamespace(models=SimpleNamespace(runtime="ollama", ollama_base_url=None)),
        )
        monkeypatch.setattr(
            "eurpe.api.routes.runtime.probe_runtime",
            fake_probe_runtime,
        )

        started = time.monotonic()
        result = get_all_runtimes()
        elapsed = time.monotonic() - started

        assert call_count == len(RUNTIME_REGISTRY)
        assert len(result.runtimes) == len(RUNTIME_REGISTRY)
        # Serial probing would take call_count * probe_delay; concurrent
        # probing should stay close to a single probe_delay.
        assert elapsed < probe_delay * len(RUNTIME_REGISTRY)
        assert result.active_runtime == "ollama"
