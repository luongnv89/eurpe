"""JSONL audit log for outbound network attempts.

Every call to :meth:`NetworkPolicyGate.check` writes one line — a single
JSON object — to ``cfg.runtime_dir / "network-audit.log"``. The line
contains ONLY connection-level metadata: when, where, what scheme, what
decision, why. It does NOT contain:

* request body or response body
* HTTP headers (auth tokens, cookies)
* URL query strings
* anything that could be proposal content, retrieved passages, prompts,
  or generated draft text

This is the acceptance criterion the whole package exists to satisfy:

    "Any outbound request attempt is logged locally without storing
    proposal content, prompts, retrieved passages, or generated draft
    text."

To make accidentally violating that contract harder, this module accepts
ONLY the fields the gate provides — there is no ``extra``, ``payload``,
or ``meta`` parameter. Add a field here, and only here, if a new piece
of metadata genuinely needs to be recorded; the review of that change
is the chokepoint that protects the contract.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# Dedicated logger so we can attach a FileHandler without polluting the
# root logger or the rest of the application. ``propagate=False`` (set
# in :func:`_get_audit_logger`) keeps these lines out of stdout.
_AUDIT_LOGGER_NAME = "eurpe.security.audit"

# Cache of ``{audit_log_path: FileHandler}`` keyed by absolute path. We
# memoise so a process that creates multiple ``NetworkPolicyGate``
# instances (one per factory call, e.g.) doesn't accumulate duplicate
# handlers on the logger.
_HANDLERS: dict[str, logging.FileHandler] = {}
_LOCK = Lock()


def _redact_path(path: str) -> str:
    """Keep only the first path segment.

    ``/api/embed/v1/foo`` → ``/api`` — enough to distinguish "an
    embedding request" from "a model-list request" while dropping any
    user-supplied segments (proposal IDs, hash slugs, etc.) that could
    leak corpus content.

    Edge cases:

    * empty string or ``/`` → ``"/"``
    * path without a leading slash → still keeps the first segment,
      prefixed with ``/`` so the log shape is uniform.
    """

    if not path or path == "/":
        return "/"
    # Strip leading slashes first so split doesn't produce a leading
    # empty token; then take the first non-empty segment.
    stripped = path.lstrip("/")
    if not stripped:
        return "/"
    first_segment = stripped.split("/", 1)[0]
    return f"/{first_segment}"


def _get_audit_logger(audit_log_path: Path) -> logging.Logger:
    """Return the audit logger, attaching a :class:`FileHandler` if needed.

    The handler is attached lazily and cached by absolute path so:

    * multiple gates writing to the same path share one handler;
    * tests using ``tmp_path`` get their own handler per test and don't
      leak lines into a sibling test's log.

    The parent directory MUST already exist — the gate guarantees this
    via ``EurpeConfig.runtime_dir`` being created by
    :func:`eurpe.config.ensure_runtime_dirs`. We assert that contract
    here so the failure mode is loud (a missing runtime_dir is a
    bootstrapping bug, not an audit-log bug).
    """

    logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    # Don't propagate to root — these lines must not leak into stdout
    # or any other handler the application installs for general logging.
    logger.propagate = False
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    key = str(audit_log_path.resolve())
    with _LOCK:
        existing = _HANDLERS.get(key)
        if existing is not None and existing in logger.handlers:
            return logger
        # Either no cached handler or it was removed (e.g., test teardown).
        # Build a fresh one.
        parent = audit_log_path.parent
        if not parent.exists():
            raise FileNotFoundError(
                f"network-audit.log parent directory does not exist: {parent}. "
                "Call eurpe.config.ensure_runtime_dirs(config) before constructing "
                "a NetworkPolicyGate."
            )
        handler = logging.FileHandler(audit_log_path, mode="a", encoding="utf-8")
        # Plain message — we serialise JSON ourselves, no extra formatter
        # decorations (timestamps, levels) so the file is line-delimited JSON.
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        # Remove any stale handler pointing at the same path before adding.
        if existing is not None:
            try:
                logger.removeHandler(existing)
                existing.close()
            except Exception:  # pragma: no cover - defensive
                pass
        logger.addHandler(handler)
        _HANDLERS[key] = handler
    return logger


def log_attempt(
    audit_log_path: Path,
    *,
    host: str,
    port: int,
    scheme: str,
    path: str,
    decision: str,
    reason: str,
    source: str,
) -> None:
    """Append one JSONL record describing an egress attempt.

    Parameters are keyword-only to make accidental field-order swaps at
    the call site impossible. The accepted set is fixed and minimal —
    if you find yourself wanting to add a parameter, stop and re-read
    the module docstring first.

    Fields written, in this exact shape:

    * ``timestamp`` — UTC ISO-8601 with seconds + ``Z`` suffix
    * ``host`` — lowercase hostname / IP (no scheme, no port)
    * ``port`` — TCP port int
    * ``scheme`` — ``http`` | ``https`` | ``tcp`` | ...
    * ``path`` — first segment only, prefixed with ``/``; query string dropped
    * ``decision`` — ``ALLOWED`` | ``DENIED``
    * ``reason`` — short human-readable rationale (e.g. ``loopback``,
      ``not on allowlist``, ``allowlist match: telemetry-mirror``)
    * ``source`` — call-site label (e.g. ``ollama_embedder.embed``)

    No body, no headers, no query string, no auth, no payload. Adding a
    field here REQUIRES a review pass against the AC stated in the
    module docstring.
    """

    record = {
        "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": host.lower(),
        "port": int(port),
        "scheme": scheme.lower(),
        "path": _redact_path(path),
        "decision": decision,
        "reason": reason,
        "source": source,
    }
    logger = _get_audit_logger(audit_log_path)
    logger.info(json.dumps(record, sort_keys=True, separators=(",", ":")))


def _reset_handlers_for_tests() -> None:
    """Detach and close every cached audit handler. Test-only.

    Pytest reuses the same process across tests, so handlers attached
    to a ``tmp_path`` that no longer exists become stale. Tests that
    need a clean logger state can call this from a fixture teardown.
    """

    logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    with _LOCK:
        for handler in list(_HANDLERS.values()):
            try:
                logger.removeHandler(handler)
                handler.close()
            except Exception:  # pragma: no cover - defensive
                pass
        _HANDLERS.clear()
