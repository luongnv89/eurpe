"""Tests for :mod:`eurpe.security.audit` — the JSONL audit writer.

The audit log is the single source of truth for "what egress was
attempted". Bugs here would silently weaken the AC2 contract from
issue #12: "Any outbound request attempt is logged locally without
storing proposal content, prompts, retrieved passages, or generated
draft text."

These tests pin:

* JSONL line shape (one valid JSON object per line, exact field set).
* Path redaction collapses to first segment.
* The handler is reused for repeated calls (no duplicates).
* No forbidden fields land in the log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from eurpe.security.audit import (
    _AUDIT_LOGGER_NAME,
    _child_logger_name,
    _redact_path,
    _reset_handlers_for_tests,
    log_attempt,
)


@pytest.fixture(autouse=True)
def _clean_audit_handlers() -> None:
    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


# ---------------------------------------------------------------------------
# Path redactor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", "/"),
        ("/", "/"),
        ("/api", "/api"),
        ("/api/embeddings", "/api"),
        ("/api/embed/v1/foo/bar", "/api"),
        ("api/embed", "/api"),
        ("/secret/proposal-id-12345", "/secret"),
    ],
)
def test_redact_path_collapses_to_first_segment(raw: str, expected: str) -> None:
    assert _redact_path(raw) == expected


# ---------------------------------------------------------------------------
# log_attempt shape
# ---------------------------------------------------------------------------


def _read_one(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1, f"expected exactly 1 line, got: {lines!r}"
    return json.loads(lines[0])


def test_log_attempt_writes_exact_field_set(tmp_path: Path) -> None:
    audit_log = tmp_path / "audit.log"
    log_attempt(
        audit_log,
        host="Example.COM",
        port=443,
        scheme="HTTPS",
        path="/api/embeddings/v1",
        decision="ALLOWED",
        reason="loopback",
        source="ollama_embedder.embed",
    )
    rec = _read_one(audit_log)
    # Exact field set — adding a field here means a new piece of
    # potentially-leakable metadata; this test forces the change to
    # be reviewed.
    assert set(rec.keys()) == {
        "timestamp",
        "host",
        "port",
        "scheme",
        "path",
        "decision",
        "reason",
        "source",
    }
    # Case-normalisation
    assert rec["host"] == "example.com"
    assert rec["scheme"] == "https"
    # Redaction
    assert rec["path"] == "/api"
    # Pass-through
    assert rec["port"] == 443
    assert rec["decision"] == "ALLOWED"
    assert rec["source"] == "ollama_embedder.embed"
    # ISO-8601 UTC suffix
    assert rec["timestamp"].endswith("Z")


def test_log_attempt_appends_multiple_lines(tmp_path: Path) -> None:
    audit_log = tmp_path / "audit.log"
    for i in range(3):
        log_attempt(
            audit_log,
            host="127.0.0.1",
            port=i + 1,
            scheme="tcp",
            path="/",
            decision="ALLOWED",
            reason="loopback",
            source=f"src-{i}",
        )
    lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # Each line parses to JSON independently.
    for line in lines:
        json.loads(line)


def test_log_attempt_handler_is_reused_across_calls(tmp_path: Path) -> None:
    """Calling log_attempt repeatedly on the same path must NOT attach
    duplicate handlers to the audit logger — otherwise each line would
    be written N times."""

    audit_log = tmp_path / "audit.log"
    for _ in range(5):
        log_attempt(
            audit_log,
            host="127.0.0.1",
            port=11434,
            scheme="http",
            path="/api",
            decision="ALLOWED",
            reason="loopback",
            source="src",
        )
    # Should have exactly 5 lines, not 5*N.
    lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    # And the per-path child logger should have exactly ONE
    # FileHandler pointing at this path.
    child = logging.getLogger(_child_logger_name(audit_log))
    file_handlers = [
        h
        for h in child.handlers
        if isinstance(h, logging.FileHandler)
        and Path(h.baseFilename).resolve() == audit_log.resolve()
    ]
    assert len(file_handlers) == 1


def test_log_attempt_does_not_propagate_to_root(tmp_path: Path, caplog) -> None:
    """Audit lines must not leak into root logger / stdout / stderr."""

    audit_log = tmp_path / "audit.log"
    with caplog.at_level(logging.DEBUG):
        log_attempt(
            audit_log,
            host="example.com",
            port=443,
            scheme="https",
            path="/x",
            decision="DENIED",
            reason="not on allowlist",
            source="src",
        )
    # caplog captures records that propagate to root; audit must not.
    # The per-path child loggers all live under _AUDIT_LOGGER_NAME, so
    # filter by the parent namespace to catch any leak from any path.
    propagated = [r for r in caplog.records if r.name.startswith(_AUDIT_LOGGER_NAME)]
    assert propagated == []


def test_log_attempt_records_decision_string(tmp_path: Path) -> None:
    audit_log = tmp_path / "audit.log"
    log_attempt(
        audit_log,
        host="example.com",
        port=443,
        scheme="https",
        path="/",
        decision="DENIED",
        reason="not on allowlist",
        source="src",
    )
    rec = _read_one(audit_log)
    assert rec["decision"] == "DENIED"


def test_log_attempt_isolates_writes_per_path(tmp_path: Path) -> None:
    """Two gates with different audit paths must NOT cross-contaminate.

    Regression guard for a process-global handler cache that, before
    the per-path child-logger refactor, attached every handler to the
    single ``eurpe.security.audit`` logger — every ``log_attempt`` call
    would then write to BOTH files. With per-path child loggers, a
    write to path A appears only in file A.
    """

    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    log_attempt(
        log_a,
        host="127.0.0.1",
        port=1,
        scheme="http",
        path="/api/a",
        decision="ALLOWED",
        reason="loopback",
        source="src-a",
    )
    log_attempt(
        log_b,
        host="127.0.0.1",
        port=2,
        scheme="http",
        path="/api/b",
        decision="ALLOWED",
        reason="loopback",
        source="src-b",
    )

    a_rec = _read_one(log_a)
    b_rec = _read_one(log_b)
    assert a_rec["source"] == "src-a"
    assert a_rec["port"] == 1
    assert b_rec["source"] == "src-b"
    assert b_rec["port"] == 2


def test_log_attempt_keyword_only_arguments(tmp_path: Path) -> None:
    """Positional arguments past the path MUST be rejected — the field
    set is small enough that misordered calls would silently swap
    decision and reason. Keyword-only enforces clarity at the call site.
    """

    audit_log = tmp_path / "audit.log"
    # First arg is positional (the path); everything else is keyword-only.
    with pytest.raises(TypeError):
        log_attempt(
            audit_log,
            "example.com",  # type: ignore[misc]
            443,
            "https",
            "/",
            "ALLOWED",
            "loopback",
            "src",
        )
