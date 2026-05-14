"""Shared offline ``config.yaml`` writer for EURPE tests.

The offline contract — Ollama URL pointing at an unreachable port so
``make_embedder`` / ``make_llm_client`` fall back to their deterministic
stubs — was previously duplicated as a private ``_write_offline_config``
in :mod:`tests.test_retrieval_cli` and :mod:`tests.test_generation_cli`.
A single source of truth here means changing the contract (e.g., new
field, different fallback port) is a one-line edit.

The shape produced is identical to the pre-refactor inlined version so
existing tests pass unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_offline_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write a ``config.yaml`` under ``tmp_path`` that disables Ollama.

    The default body matches the structure used by
    ``tests/test_retrieval_cli.py`` and ``tests/test_generation_cli.py``
    before this refactor:

    * ``corpus_path`` and ``index_path`` are pinned under ``tmp_path``.
    * ``models.ollama_base_url`` points at ``http://localhost:1`` —
      port 1 is reserved by IANA and never has a listener, so the
      factory falls back to the deterministic stub on the first call.
    * ``offline_mode: true``.

    ``overrides`` are merged shallowly onto the default top-level dict;
    nested keys (e.g., a different LLM model under ``models``) require
    a complete ``models`` block in the override since this is a shallow
    merge by design — explicit beats clever.
    """

    cfg_path = tmp_path / "config.yaml"
    body: dict[str, Any] = {
        "corpus_path": str(tmp_path / "corpus"),
        "index_path": str(tmp_path / "index"),
        "models": {
            "runtime": "ollama",
            "llm_model": "llama3.1:8b",
            "embedding_model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:1",  # unreachable on purpose
        },
        "offline_mode": True,
        "log_level": "INFO",
    }
    body.update(overrides)
    cfg_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return cfg_path
