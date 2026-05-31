"""Local configuration loader for EURPE.

Reads ``config.yaml`` from the project root by default. If the file is missing,
``ensure_config_file`` will copy ``config.example.yaml`` into place. The
configuration is validated with Pydantic models so the rest of the application
can rely on a typed, well-shaped object.

All runtime defaults assume **offline operation**: no network access is required
to load or validate configuration.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from eurpe.security.allowlist import AllowlistEntry

# Repository root resolves to .../eu-research-projects (two parents up from this file:
# src/eurpe/config.py -> src/eurpe -> src -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "config.example.yaml"


class ModelsConfig(BaseModel):
    """Local model runtime settings.

    No URL is dialed at config-load time. Base URLs and API-key environment
    variable names are recorded for the generation module to use later when
    explicitly invoked by the user.
    """

    runtime: str = Field(
        default="ollama",
        description=(
            "ollama | openai | openrouter | groq | lmstudio | vllm | llamacpp | anthropic | gemini"
        ),
    )
    llm_model: str = Field(default="llama3.1:8b")
    embedding_model: str = Field(default="nomic-embed-text")
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_base_url: str | None = Field(
        default=None,
        description="Optional provider/engine base URL override for generation.",
    )
    llm_api_key_env: str | None = Field(
        default=None,
        description="Environment variable name containing the provider API key/token.",
    )

    @field_validator("runtime")
    @classmethod
    def _runtime_must_be_known(cls, value: str) -> str:
        normalised = value.lower().replace("-", "").replace("_", "")
        aliases = {
            "llama.cpp": "llamacpp",
            "llamacpp": "llamacpp",
            "lmstudio": "lmstudio",
            "lm": "lmstudio",
        }
        normalised = aliases.get(normalised, normalised)
        allowed = {
            "ollama",
            "openai",
            "openrouter",
            "groq",
            "lmstudio",
            "vllm",
            "llamacpp",
            "anthropic",
            "gemini",
        }
        if normalised not in allowed:
            raise ValueError(f"runtime must be one of {sorted(allowed)}, got {value!r}")
        return normalised


class EurpeConfig(BaseModel):
    """Top-level EURPE configuration."""

    corpus_path: Path = Field(default=Path("./data/corpus"))
    index_path: Path = Field(default=Path("./data/index"))
    # ``runtime_dir`` is the parent for short-lived state owned by the
    # FastAPI service: staging uploads, parse-token records, and the
    # YAML sidecar archive written by ``POST /api/ingestion/confirm``.
    # Kept separate from ``corpus_path`` (where curated proposals live) so
    # an operator can wipe runtime state without touching the corpus, and
    # separate from ``index_path`` (Chroma's home) so a backend swap does
    # not collide with upload bookkeeping.
    runtime_dir: Path = Field(default=Path("./data/runtime"))
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    offline_mode: bool = Field(default=True)
    # Default-deny network policy: loopback is always allowed; anything
    # else must be opted in here. Empty list (the default) is the
    # secure default. See ``eurpe.security.NetworkPolicyGate``.
    network_allowlist: list[AllowlistEntry] = Field(default_factory=list)
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _log_level_known(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    def resolve_paths(self, base: Path | None = None) -> EurpeConfig:
        """Return a copy with relative paths resolved against ``base``.

        Defaults to the repository root so that running ``eurpe smoke`` from any
        working directory produces consistent paths.
        """
        anchor = (base or REPO_ROOT).resolve()
        corpus = self.corpus_path
        index = self.index_path
        runtime = self.runtime_dir
        if not corpus.is_absolute():
            corpus = (anchor / corpus).resolve()
        if not index.is_absolute():
            index = (anchor / index).resolve()
        if not runtime.is_absolute():
            runtime = (anchor / runtime).resolve()
        return self.model_copy(
            update={
                "corpus_path": corpus,
                "index_path": index,
                "runtime_dir": runtime,
            }
        )

    def network_audit_log_path(self) -> Path:
        """Path to the JSONL audit log for outbound network attempts.

        Lives under ``runtime_dir`` because the log is short-lived
        operational state (matches the rationale for staging uploads
        and parse-token records). Wiping ``runtime_dir`` re-creates a
        clean log on the next gate construction.
        """

        return self.runtime_dir / "network-audit.log"

    def analytics_log_path(self) -> Path:
        """Path to the JSONL log of local analytics events.

        Lives under ``runtime_dir`` so wiping runtime state removes the
        log. AC3 of issue #13 requires that this file never leaves the
        runtime directory unless the user explicitly runs ``eurpe
        analytics export``.
        """

        return self.runtime_dir / "analytics-events.log"


def ensure_config_file(
    config_path: Path = DEFAULT_CONFIG_PATH,
    example_path: Path = EXAMPLE_CONFIG_PATH,
) -> Path:
    """Make sure a ``config.yaml`` exists, copying from the example if needed.

    Returns the path that was used. No network access is performed.
    """
    if config_path.exists():
        return config_path
    if not example_path.exists():
        raise FileNotFoundError(
            f"Neither {config_path} nor {example_path} exists; cannot bootstrap config."
        )
    shutil.copyfile(example_path, config_path)
    return config_path


def load_config(config_path: Path | None = None) -> EurpeConfig:
    """Load and validate the EURPE configuration from ``config_path``.

    If ``config_path`` is ``None``, the repository-root ``config.yaml`` is used.
    The file must already exist — call :func:`ensure_config_file` first if you
    want bootstrap behaviour.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping; got {type(raw).__name__}")
    return EurpeConfig.model_validate(raw)


def ensure_runtime_dirs(config: EurpeConfig) -> None:
    """Create the corpus, index, and runtime directories on disk if they do not exist."""
    config.corpus_path.mkdir(parents=True, exist_ok=True)
    config.index_path.mkdir(parents=True, exist_ok=True)
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
