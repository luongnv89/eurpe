"""Unit tests for the ``EURPE_E2E_REQUIRE_LLM`` gate (issue #48).

The gate lives in :mod:`tests._helpers.require_llm` and is used by the
E2E pipeline test in :mod:`tests.e2e.test_full_pipeline`. These tests
exercise the gate directly — both branches (env unset → no-op; env
set + stub detected → loud failure) — without spinning up the slow
pipeline twice.

Because both the E2E test and these unit tests import
``assert_real_llm_was_used`` from the same module, a passing unit test
here is genuine evidence that the production gate behaves as
documented.
"""

from __future__ import annotations

import pytest

from eurpe.generation.llm import DeterministicLLMClient
from tests._helpers.require_llm import assert_real_llm_was_used


def test_gate_is_noop_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2 — when EURPE_E2E_REQUIRE_LLM is unset, stub usage is fine."""

    monkeypatch.delenv("EURPE_E2E_REQUIRE_LLM", raising=False)

    stub_payload = {"model": DeterministicLLMClient.MODEL_NAME}

    # Must not raise.
    assert_real_llm_was_used(stub_payload)


def test_gate_is_noop_when_env_var_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``EURPE_E2E_REQUIRE_LLM=1`` activates the gate.

    Any other value (including ``0`` or ``true`` or empty) is treated
    as unset. This keeps the env-var contract simple and avoids the
    ``"0" is truthy in Python`` foot-gun.
    """

    monkeypatch.setenv("EURPE_E2E_REQUIRE_LLM", "0")

    stub_payload = {"model": DeterministicLLMClient.MODEL_NAME}

    # Must not raise — only the literal string "1" activates the gate.
    assert_real_llm_was_used(stub_payload)


def test_gate_is_noop_when_env_var_is_true_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EURPE_E2E_REQUIRE_LLM=true`` is also treated as unset.

    The documented contract is strict equality with ``"1"``; the word
    ``true`` is not honoured. An operator who set it expecting "true =
    on" will not get the gate, but neither will they get a confusing
    intermediate state. Documents and pins the contract.
    """

    monkeypatch.setenv("EURPE_E2E_REQUIRE_LLM", "true")

    stub_payload = {"model": DeterministicLLMClient.MODEL_NAME}

    # Must not raise.
    assert_real_llm_was_used(stub_payload)


def test_gate_fails_when_stub_is_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC1 — env var set + stub model name ⇒ AssertionError, named env var."""

    monkeypatch.setenv("EURPE_E2E_REQUIRE_LLM", "1")

    stub_payload = {"model": DeterministicLLMClient.MODEL_NAME}

    with pytest.raises(AssertionError) as excinfo:
        assert_real_llm_was_used(stub_payload)

    # AC1 requires a clear, actionable message. The env var name must
    # appear so the operator can search for it; the stub model name
    # must appear so they know which model was actually used.
    msg = str(excinfo.value)
    assert "EURPE_E2E_REQUIRE_LLM" in msg
    assert DeterministicLLMClient.MODEL_NAME in msg
    assert "ollama" in msg.lower()


def test_gate_passes_when_real_llm_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var set + real-LLM model name ⇒ no error."""

    monkeypatch.setenv("EURPE_E2E_REQUIRE_LLM", "1")

    real_payload = {"model": "llama3.1:8b"}

    # Must not raise.
    assert_real_llm_was_used(real_payload)


def test_gate_does_not_raise_on_missing_model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed payload missing ``model`` is not flagged as a stub leak.

    ``payload.get("model")`` returns ``None``, which is not equal to
    ``DeterministicLLMClient.MODEL_NAME``, so the strict-equality
    check passes through. That's the correct call: a missing
    ``model`` field is a separate (and more serious) bug than a stub
    leak. ``GenerationDraft.model`` is a Pydantic ``min_length=1``
    field, so upstream validation already guarantees the key is
    present and non-empty; this gate only flags the stub-leak case.
    """

    monkeypatch.setenv("EURPE_E2E_REQUIRE_LLM", "1")

    malformed_payload: dict[str, str] = {}

    # Must not raise.
    assert_real_llm_was_used(malformed_payload)


def test_deterministic_stub_model_name_constant_stable() -> None:
    """Smoke test that ``MODEL_NAME`` is the expected value.

    The gate's correctness depends on the constant. If someone renames
    or removes ``MODEL_NAME``, this test fails fast — much louder than
    a subtle E2E false-pass would be.
    """

    assert DeterministicLLMClient.MODEL_NAME == "deterministic-stub-v1"
