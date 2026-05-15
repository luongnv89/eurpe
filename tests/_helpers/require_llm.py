"""The ``EURPE_E2E_REQUIRE_LLM`` gate (issue #48).

When an operator sets ``EURPE_E2E_REQUIRE_LLM=1``, they are asserting
that the E2E pipeline will use a real LLM (Ollama) for the generation
step. If the deterministic offline stub is hit instead — because Ollama
is unreachable, the model is not pulled, or ``models.ollama_base_url``
in ``config.yaml`` is wrong — the suite must fail loudly so the
operator notices, instead of silently accepting placeholder text as a
valid draft.

The gate is implemented as a single function so the production E2E
test (:mod:`tests.e2e.test_full_pipeline`) and the fast-tier unit
tests (:mod:`tests.test_e2e_require_llm`) share the exact same body.
A drift between the two would otherwise be the most plausible failure
mode of this gate, and one a unit test of a *copy* would not catch.

Contract:

* ``EURPE_E2E_REQUIRE_LLM`` unset (or any value other than the literal
  string ``"1"``) → no-op. Strict-equality matching keeps the env-var
  contract simple and avoids the ``"0" is truthy in Python`` foot-gun.
* ``EURPE_E2E_REQUIRE_LLM == "1"`` AND ``payload["model"]`` equals
  :attr:`DeterministicLLMClient.MODEL_NAME` → ``AssertionError`` with
  a message naming the env var, the offending model, and the next
  steps (start Ollama, pull the model, check config).
* A missing ``model`` key (``payload.get("model")`` returning ``None``)
  is intentionally not flagged — Pydantic validation upstream
  guarantees ``GenerationDraft.model`` is non-empty, so a missing key
  here is a schema bug that should surface at its actual layer rather
  than being re-reported as a stub leak by this gate.
"""

from __future__ import annotations

import os
from typing import Any

from eurpe.generation.llm import DeterministicLLMClient


def assert_real_llm_was_used(payload: dict[str, Any]) -> None:
    """Fail if ``EURPE_E2E_REQUIRE_LLM=1`` but the draft came from the stub.

    See module docstring for the full contract.
    """

    if os.environ.get("EURPE_E2E_REQUIRE_LLM") != "1":
        return

    assert payload.get("model") != DeterministicLLMClient.MODEL_NAME, (
        "EURPE_E2E_REQUIRE_LLM=1 but the draft was produced by the "
        f"deterministic stub (model={payload.get('model')!r}).\n"
        "Start Ollama and verify config:\n"
        "  ollama serve &\n"
        "  ollama pull llama3.1:8b\n"
        "Confirm `models.ollama_base_url` in config.yaml is reachable.\n"
        "See README §Tests for the full setup."
    )
