"""Tests for :class:`eurpe.security.policy.NetworkPolicyGate`.

These are PURE decision tests — no real sockets are opened. The gate
exists precisely to short-circuit before any socket call, so a test
that requires the network would mean the gate's contract is broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eurpe.security import (
    AllowlistEntry,
    Decision,
    EgressDeniedError,
    NetworkPolicyGate,
)
from eurpe.security.audit import _reset_handlers_for_tests
from eurpe.security.policy import _is_loopback


@pytest.fixture(autouse=True)
def _clean_audit_handlers() -> None:
    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


@pytest.fixture
def audit_log(tmp_path: Path) -> Path:
    """Audit log path under tmp_path so each test starts clean."""

    return tmp_path / "network-audit.log"


# ---------------------------------------------------------------------------
# _is_loopback helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST",
        "Localhost",
        "127.0.0.1",
        "127.255.255.254",
        "::1",
        "[::1]",
        # IPv4-mapped IPv6 loopback. Python's ipaddress module reports
        # ``::ffff:127.0.0.1`` as is_loopback=True (the embedded v4 is
        # in the loopback /8). Pinned here so a future refactor that
        # changes the loopback helper can't silently drop this case.
        "::ffff:127.0.0.1",
        "[::ffff:127.0.0.1]",
    ],
)
def test_is_loopback_recognises_loopback_hosts(host: str) -> None:
    assert _is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        "1.1.1.1",
        "192.0.2.1",  # TEST-NET-1
        "8.8.8.8",
        "",
        "not.a.real.host.invalid",
    ],
)
def test_is_loopback_rejects_non_loopback(host: str) -> None:
    assert _is_loopback(host) is False


# ---------------------------------------------------------------------------
# .check() decisions
# ---------------------------------------------------------------------------


def test_loopback_allowed_without_allowlist(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    decision = gate.check("127.0.0.1", 11434, "http", "/api/x", "test")
    assert decision is Decision.ALLOWED


def test_localhost_string_allowed_without_allowlist(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    assert gate.check("localhost", 11434, "http", "/api/x", "test") is Decision.ALLOWED


def test_ipv6_loopback_allowed_without_allowlist(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    assert gate.check("::1", 443, "https", "/", "test") is Decision.ALLOWED


def test_ipv4_mapped_ipv6_loopback_allowed_without_allowlist(
    audit_log: Path,
) -> None:
    """``::ffff:127.0.0.1`` is the IPv4-mapped form of 127.0.0.1.

    Pinned so the gate keeps allowing it; a regression here would
    surprise anyone who configures Ollama to bind on a v6 socket and
    receives v4-mapped connections from local clients.
    """

    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    assert gate.check("::ffff:127.0.0.1", 11434, "http", "/api/x", "test") is Decision.ALLOWED


def test_non_loopback_denied_without_allowlist(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    with pytest.raises(EgressDeniedError) as excinfo:
        gate.check("example.com", 443, "https", "/api/x", "test")
    # The error message must not include any payload — only
    # connection-level metadata.
    msg = str(excinfo.value)
    assert "example.com" in msg
    assert "443" in msg
    assert "test" in msg


def test_allowlist_match_allowed(audit_log: Path) -> None:
    gate = NetworkPolicyGate(
        allowlist=[AllowlistEntry(host="example.com", port=443, reason="mirror")],
        audit_log_path=audit_log,
    )
    assert gate.check("example.com", 443, "https", "/", "test") is Decision.ALLOWED


def test_allowlist_match_is_case_insensitive(audit_log: Path) -> None:
    gate = NetworkPolicyGate(
        allowlist=[AllowlistEntry(host="example.com", port=443, reason="x")],
        audit_log_path=audit_log,
    )
    assert gate.check("EXAMPLE.COM", 443, "https", "/", "test") is Decision.ALLOWED


def test_allowlist_mismatch_port_denied(audit_log: Path) -> None:
    gate = NetworkPolicyGate(
        allowlist=[AllowlistEntry(host="example.com", port=443, reason="x")],
        audit_log_path=audit_log,
    )
    with pytest.raises(EgressDeniedError):
        gate.check("example.com", 8443, "https", "/", "test")


# ---------------------------------------------------------------------------
# Audit log content
# ---------------------------------------------------------------------------


def _read_lines(audit_log: Path) -> list[dict]:
    if not audit_log.exists():
        return []
    return [
        json.loads(line)
        for line in audit_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_audit_log_written_for_allowed_decision(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    gate.check("127.0.0.1", 11434, "http", "/api/embeddings", "test_source")
    records = _read_lines(audit_log)
    assert len(records) == 1
    rec = records[0]
    assert rec["host"] == "127.0.0.1"
    assert rec["port"] == 11434
    assert rec["scheme"] == "http"
    assert rec["decision"] == "ALLOWED"
    assert rec["reason"] == "loopback"
    assert rec["source"] == "test_source"


def test_audit_log_written_for_denied_decision(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    with pytest.raises(EgressDeniedError):
        gate.check("example.com", 443, "https", "/api/secret", "test_source")
    records = _read_lines(audit_log)
    assert len(records) == 1
    rec = records[0]
    assert rec["decision"] == "DENIED"
    assert rec["reason"] == "not on allowlist"
    assert rec["host"] == "example.com"


def test_audit_log_records_allowlist_match_reason(audit_log: Path) -> None:
    gate = NetworkPolicyGate(
        allowlist=[AllowlistEntry(host="example.com", port=443, reason="approved mirror")],
        audit_log_path=audit_log,
    )
    gate.check("example.com", 443, "https", "/", "test")
    rec = _read_lines(audit_log)[0]
    assert rec["decision"] == "ALLOWED"
    assert "approved mirror" in rec["reason"]


def test_audit_log_path_is_redacted_to_first_segment(audit_log: Path) -> None:
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    gate.check("127.0.0.1", 80, "http", "/api/embed/v1/proposal-12345", "test")
    rec = _read_lines(audit_log)[0]
    # First segment only; nothing user-supplied (proposal-12345) leaks.
    assert rec["path"] == "/api"


def test_audit_log_drops_query_string(audit_log: Path) -> None:
    """A path containing '?' is not parsed by the gate, but the redactor
    should still produce a clean first-segment string regardless."""

    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    # The gate accepts ``path`` opaquely; the audit module redacts it
    # to one segment. Even a malformed path with a query string in it
    # collapses to the first segment.
    gate.check("127.0.0.1", 80, "http", "/api?secret=xyz", "test")
    rec = _read_lines(audit_log)[0]
    # First segment captured; the redactor doesn't try to strip a
    # query string because the gate never gets a query string in the
    # path argument from real call sites — they pass just the path.
    # The point is: nothing past the first '/' lands in the log.
    assert "secret" not in rec["path"]


def test_audit_log_never_contains_forbidden_fields(audit_log: Path) -> None:
    """The JSONL line MUST NOT carry body / headers / query / prompt fields.

    Bare minimum check that pins the AC2 content-safety contract.
    """

    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    gate.check("127.0.0.1", 11434, "http", "/api", "src1")
    with pytest.raises(EgressDeniedError):
        gate.check("example.com", 443, "https", "/api", "src2")
    text = audit_log.read_text(encoding="utf-8")
    for forbidden in ("body", "headers", "query", "prompt", "completion", "passage"):
        assert (
            forbidden not in text.lower()
        ), f"audit log must not contain {forbidden!r}; saw {text!r}"


def test_audit_log_is_appended_not_truncated(audit_log: Path) -> None:
    """Multiple checks against the same gate accumulate lines."""

    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    gate.check("127.0.0.1", 1, "tcp", "/", "a")
    gate.check("127.0.0.1", 2, "tcp", "/", "b")
    gate.check("127.0.0.1", 3, "tcp", "/", "c")
    records = _read_lines(audit_log)
    assert len(records) == 3
    assert [r["source"] for r in records] == ["a", "b", "c"]


def test_audit_log_one_line_per_attempt(audit_log: Path) -> None:
    """Each check produces exactly one record, and each line is valid JSON."""

    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    gate.check("localhost", 11434, "http", "/api", "src")
    raw = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    # No trailing comma, single object, parses with strict json.
    assert json.loads(raw[0])["source"] == "src"


def test_audit_log_parent_must_exist(tmp_path: Path) -> None:
    """A missing parent directory raises — the gate must not silently
    swallow audit failures."""

    missing = tmp_path / "does-not-exist" / "audit.log"
    gate = NetworkPolicyGate(allowlist=[], audit_log_path=missing)
    with pytest.raises(FileNotFoundError):
        gate.check("127.0.0.1", 1, "tcp", "/", "x")


def test_egress_denied_error_message_excludes_secret_path(audit_log: Path) -> None:
    """The exception message MUST NOT carry the full request path.

    The path is a likely leak vector — proposal IDs, hash slugs, etc.
    The exception is a control-flow signal, not a data carrier.
    """

    gate = NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)
    with pytest.raises(EgressDeniedError) as excinfo:
        gate.check(
            "example.com",
            443,
            "https",
            "/api/secret-proposal-abc123",
            "test",
        )
    assert "secret-proposal-abc123" not in str(excinfo.value)
