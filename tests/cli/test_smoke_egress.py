"""Tests for the egress-policy probe inside ``eurpe smoke``.

AC3 from issue #12: "Release smoke test fails if a non-opt-in outbound
request succeeds." We exercise both directions:

* Default config → probe DENIED → smoke exits 0.
* Misconfigured config that allowlists TEST-NET → probe ALLOWED →
  smoke exits non-zero with a clear regression message.

The TEST-NET-1 prefix (192.0.2.0/24, RFC 5737) is documentation-only
and never routed on the Internet, so no real packet ever leaves the
machine even if the gate misbehaves — the test stays fully offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from eurpe.cli import app
from eurpe.security.audit import _reset_handlers_for_tests


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    cfg_path = tmp_path / "config.yaml"
    body: dict[str, Any] = {
        "corpus_path": str(tmp_path / "corpus"),
        "index_path": str(tmp_path / "index"),
        "runtime_dir": str(tmp_path / "runtime"),
        "models": {
            "runtime": "ollama",
            "llm_model": "llama3.1:8b",
            "embedding_model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:1",
        },
        "offline_mode": True,
        "log_level": "INFO",
    }
    body.update(overrides)
    cfg_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return cfg_path


@pytest.fixture(autouse=True)
def _clean_audit_handlers() -> None:
    """Detach cached audit handlers so each test starts with a fresh logger."""

    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


def test_smoke_passes_when_test_net_probe_denied(tmp_path: Path) -> None:
    """Default config: no allowlist → TEST-NET probe denied → exit 0."""

    cfg_path = _write_config(tmp_path)
    result = CliRunner().invoke(app, ["smoke", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "TEST-NET probe denied as expected" in result.output
    # Audit log MUST be created with the denial record.
    audit_log = tmp_path / "runtime" / "network-audit.log"
    assert audit_log.exists()
    assert "DENIED" in audit_log.read_text(encoding="utf-8")


def test_smoke_fails_when_test_net_probe_allowed(tmp_path: Path) -> None:
    """If TEST-NET is allowlisted (broken config), smoke must fail loudly."""

    cfg_path = _write_config(
        tmp_path,
        network_allowlist=[
            {
                "host": "192.0.2.1",
                "port": 443,
                "reason": "intentionally bad regression test",
            }
        ],
    )
    result = CliRunner().invoke(app, ["smoke", "--config", str(cfg_path)])
    assert result.exit_code != 0
    assert "Security regression" in result.output
    assert "TEST-NET probe was ALLOWED" in result.output


def test_smoke_audit_log_does_not_leak_payload(tmp_path: Path) -> None:
    """The probe must not write a prompt/body/header field into the audit log.

    This pins the AC2 content-safety contract end-to-end: even if the
    audit module is refactored, a stray field like "body" or "prompt"
    or "headers" would be caught here.
    """

    cfg_path = _write_config(tmp_path)
    result = CliRunner().invoke(app, ["smoke", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    audit_log = tmp_path / "runtime" / "network-audit.log"
    content = audit_log.read_text(encoding="utf-8")
    for forbidden in ("body", "headers", "query", "prompt", "completion"):
        assert forbidden not in content.lower(), (
            f"audit log must not contain {forbidden!r}; saw {content!r}"
        )
