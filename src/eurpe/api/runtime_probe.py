"""Local runtime health probes for the Settings UI (issue #79).

Each supported local runtime has a default endpoint and a probe
function that returns a :class:`RuntimeStatus` payload. The probes
are lightweight TCP/HTTP checks — no model generation or heavy I/O.

Why this lives in its own module
---------------------------------
The existing ``eurpe.generation.llm`` module already knows how to
talk to Ollama (``OllamaLLMClient``) and has a ``_ollama_llm_reachable``
TCP probe. This module generalises that pattern to all supported
runtimes (Ollama, MLX, vLLM) and adds model-listing capabilities
so the Settings page can show what is installed.

Privacy
-------
All probes target ``localhost`` only. No outbound Internet traffic
is generated, consistent with the offline-first invariant.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeStatus:
    """Result of probing a single local runtime."""

    name: str
    """Human-readable runtime name (e.g. 'Ollama')."""

    endpoint: str
    """Default URL the probe targeted."""

    available: bool
    """True when the runtime responded to the health probe."""

    models: list[str] = field(default_factory=list)
    """Model identifiers reported by the runtime. Empty when unavailable."""

    error: str | None = None
    """Human-readable error when ``available`` is False."""


# ---------------------------------------------------------------------------
# Runtime registry
# ---------------------------------------------------------------------------

#: Known local runtimes with their default endpoints and display names.
#: The keys match the ``ModelsConfig.runtime`` enum values from config.py.
RUNTIME_REGISTRY: dict[str, dict[str, str]] = {
    "ollama": {
        "display_name": "Ollama",
        "default_url": "http://localhost:11434",
        "docs_url": "https://ollama.com/download",
    },
    "mlx": {
        "display_name": "MLX (Apple Silicon)",
        "default_url": "http://localhost:8080",
        "docs_url": "https://ml-explore.github.io/mlx/",
    },
    "vllm": {
        "display_name": "vLLM",
        "default_url": "http://localhost:8000",
        "docs_url": "https://docs.vllm.ai/en/latest/",
    },
}


# ---------------------------------------------------------------------------
# Probe implementations
# ---------------------------------------------------------------------------


def _tcp_probe(host: str, port: int, timeout: float = 2.0) -> bool:
    """Best-effort TCP connect to ``host:port``. Returns True on success."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_ollama(base_url: str) -> RuntimeStatus:
    """Probe Ollama's ``/api/tags`` endpoint for model listing."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    if not _tcp_probe(host, port):
        return RuntimeStatus(
            name="Ollama",
            endpoint=base_url,
            available=False,
            error=f"Cannot reach Ollama at {base_url}. Start the daemon with `ollama serve`.",
        )

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/tags")
        if resp.status_code == 200:
            payload = resp.json()
            models = [m["name"] for m in payload.get("models", [])]
            return RuntimeStatus(
                name="Ollama",
                endpoint=base_url,
                available=True,
                models=models,
            )
        return RuntimeStatus(
            name="Ollama",
            endpoint=base_url,
            available=False,
            error=f"Ollama returned HTTP {resp.status_code}",
        )
    except (httpx.HTTPError, ValueError) as exc:
        return RuntimeStatus(
            name="Ollama",
            endpoint=base_url,
            available=False,
            error=f"Ollama probe failed: {exc}",
        )


def _probe_vllm(base_url: str) -> RuntimeStatus:
    """Probe vLLM's ``/v1/models`` endpoint (OpenAI-compatible API)."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    if not _tcp_probe(host, port):
        return RuntimeStatus(
            name="vLLM",
            endpoint=base_url,
            available=False,
            error=f"Cannot reach vLLM at {base_url}. Start the server with `vllm serve <model>`.",
        )

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/models")
        if resp.status_code == 200:
            payload = resp.json()
            models = [m["id"] for m in payload.get("data", [])]
            return RuntimeStatus(
                name="vLLM",
                endpoint=base_url,
                available=True,
                models=models,
            )
        return RuntimeStatus(
            name="vLLM",
            endpoint=base_url,
            available=False,
            error=f"vLLM returned HTTP {resp.status_code}",
        )
    except (httpx.HTTPError, ValueError) as exc:
        return RuntimeStatus(
            name="vLLM",
            endpoint=base_url,
            available=False,
            error=f"vLLM probe failed: {exc}",
        )


def _probe_mlx(base_url: str) -> RuntimeStatus:
    """Probe MLX server's ``/v1/models`` endpoint (OpenAI-compatible API)."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    if not _tcp_probe(host, port):
        return RuntimeStatus(
            name="MLX (Apple Silicon)",
            endpoint=base_url,
            available=False,
            error=(
                f"Cannot reach MLX server at {base_url}. "
                "Start the server with `mlx_lm.server --model <model>`."
            ),
        )

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/models")
        if resp.status_code == 200:
            payload = resp.json()
            models = [m["id"] for m in payload.get("data", [])]
            return RuntimeStatus(
                name="MLX (Apple Silicon)",
                endpoint=base_url,
                available=True,
                models=models,
            )
        return RuntimeStatus(
            name="MLX (Apple Silicon)",
            endpoint=base_url,
            available=False,
            error=f"MLX server returned HTTP {resp.status_code}",
        )
    except (httpx.HTTPError, ValueError) as exc:
        return RuntimeStatus(
            name="MLX (Apple Silicon)",
            endpoint=base_url,
            available=False,
            error=f"MLX server probe failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROBE_FUNCTION_NAMES: dict[str, str] = {
    "ollama": "_probe_ollama",
    "vllm": "_probe_vllm",
    "mlx": "_probe_mlx",
}


def probe_runtime(runtime: str, base_url: str | None = None) -> RuntimeStatus:
    """Probe a local runtime and return its status.

    Args:
        runtime: One of ``"ollama"``, ``"vllm"``, ``"mlx"``.
        base_url: Override the default endpoint. Uses the registry default
            when ``None``.

    Returns:
        A :class:`RuntimeStatus` with availability and model list.
    """
    info = RUNTIME_REGISTRY.get(runtime)
    if info is None:
        return RuntimeStatus(
            name=runtime,
            endpoint="",
            available=False,
            error=f"Unknown runtime: {runtime}",
        )

    url = base_url or info["default_url"]
    probe_fn_name = _PROBE_FUNCTION_NAMES.get(runtime)
    if probe_fn_name is None:
        return RuntimeStatus(
            name=info["display_name"],
            endpoint=url,
            available=False,
            error=f"No probe implemented for runtime: {runtime}",
        )

    probe_fn = globals()[probe_fn_name]
    return probe_fn(url)


def get_install_instructions(runtime: str) -> dict[str, str]:
    """Return step-by-step installation instructions for a runtime.

    Returns a dict with ``title``, ``steps`` (newline-separated), and
    ``docs_url`` keys.
    """
    info = RUNTIME_REGISTRY.get(runtime)
    if info is None:
        return {
            "title": runtime,
            "steps": "No installation instructions available for this runtime.",
            "docs_url": "",
        }

    display_name = info["display_name"]
    docs_url = info["docs_url"]

    instructions: dict[str, dict[str, str]] = {
        "ollama": {
            "title": display_name,
            "steps": (
                "1. Download Ollama from https://ollama.com/download\n"
                "2. Install and run the application\n"
                "3. Pull a model: `ollama pull llama3.1:8b`\n"
                "4. Verify: `curl http://localhost:11434/api/tags`"
            ),
            "docs_url": docs_url,
        },
        "mlx": {
            "title": display_name,
            "steps": (
                "1. Install MLX: `pip install mlx mlx-lm`\n"
                "2. Start the server: "
                "`mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`\n"
                "3. Verify: `curl http://localhost:8080/v1/models`"
            ),
            "docs_url": docs_url,
        },
        "vllm": {
            "title": display_name,
            "steps": (
                "1. Install vLLM: `pip install vllm`\n"
                "2. Start the server: `vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct`\n"
                "3. Verify: `curl http://localhost:8000/v1/models`"
            ),
            "docs_url": docs_url,
        },
    }

    return instructions.get(runtime, {"title": display_name, "steps": "", "docs_url": docs_url})
