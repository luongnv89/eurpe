"""Tests for ``eurpe.config``.

Covers the example configuration shipped at the repo root and the validation
logic baked into the Pydantic models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eurpe.config import EXAMPLE_CONFIG_PATH, EurpeConfig, load_config


REQUIRED_TOP_LEVEL_KEYS = {
    "corpus_path",
    "index_path",
    "models",
    "offline_mode",
    "log_level",
}

REQUIRED_MODEL_KEYS = {
    "runtime",
    "llm_model",
    "embedding_model",
    "ollama_base_url",
}


def test_example_config_file_exists() -> None:
    assert EXAMPLE_CONFIG_PATH.exists(), (
        f"Expected example config at {EXAMPLE_CONFIG_PATH}"
    )


def test_example_config_has_required_keys() -> None:
    raw = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(raw.keys())
    assert REQUIRED_MODEL_KEYS.issubset(raw["models"].keys())


def test_example_config_loads_into_typed_model() -> None:
    config = load_config(EXAMPLE_CONFIG_PATH)
    assert isinstance(config, EurpeConfig)
    # Offline-by-default is a release-blocking requirement; keep this assertion strict.
    assert config.offline_mode is True
    assert config.models.runtime in {"ollama", "mlx", "vllm"}
    assert config.log_level == "INFO"


def test_resolve_paths_makes_paths_absolute(tmp_path: Path) -> None:
    config = load_config(EXAMPLE_CONFIG_PATH).resolve_paths(base=tmp_path)
    assert config.corpus_path.is_absolute()
    assert config.index_path.is_absolute()
    assert str(config.corpus_path).startswith(str(tmp_path.resolve()))


def test_invalid_runtime_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "corpus_path: ./data/corpus\n"
        "index_path: ./data/index\n"
        "models:\n"
        "  runtime: gpt5cloud\n"
        "  llm_model: x\n"
        "  embedding_model: y\n"
        "  ollama_base_url: http://localhost:11434\n"
        "offline_mode: true\n"
        "log_level: INFO\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):  # noqa: B017 — pydantic raises ValidationError
        load_config(bad)
