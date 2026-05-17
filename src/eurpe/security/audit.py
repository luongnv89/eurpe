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

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

# Parent namespace for every audit logger. Each distinct audit-log
# path gets its OWN child logger ``eurpe.security.audit.<hash>``; the
# parent itself is never written to. Per-path isolation is what stops
# two ``NetworkPolicyGate`` instances with different audit paths from
# cross-contaminating each other's JSONL files when both run in the
# same process. ``propagate=False`` (set in :func:`_get_audit_logger`)
# keeps these lines out of stdout.
_AUDIT_LOGGER_NAME = "eurpe.security.audit"

# Cache of ``{absolute_path: FileHandler}`` keyed by resolved absolute
# path. We memoise so a process that creates multiple gates pointing at
# the SAME file (one per factory call, e.g.) doesn't accumulate
# duplicate handlers on that path's child logger.
_HANDLERS: dict[str, logging.FileHandler] = {}
_LOCK = Lock()


def _child_logger_name(audit_log_path: Path) -> str:
    """Return the child-logger name used for one audit-log path.

    Hash of the absolute path → short hex digest. We don't put the
    path itself in the name because the logging-tree convention is
    dotted segments without slashes or spaces, and arbitrary
    filesystem paths violate that.
    """

    digest = hashlib.sha1(
        str(audit_log_path.resolve()).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]
    return f"{_AUDIT_LOGGER_NAME}.{digest}"


def _redact_path(path: str) -> str:
    """Keep only the first path segment, stripping any query / fragment.

    ``/api/embed/v1/foo`` → ``/api`` — enough to distinguish "an
    embedding request" from "a model-list request" while dropping any
    user-supplied segments (proposal IDs, hash slugs, etc.) that could
    leak corpus content.

    Defense-in-depth: even though no real call site passes a query
    string in the ``path`` argument, we still strip everything past
    ``?`` and ``#`` so a future regression at a call site can't write
    secret query parameters into the audit log.

    Edge cases:

    * empty string or ``/`` → ``"/"``
    * path without a leading slash → still keeps the first segment,
      prefixed with ``/`` so the log shape is uniform.
    """

    if not path or path == "/":
        return "/"
    # Drop a stray query string or fragment first. Anything after
    # '?' or '#' is treated as opaque user data we never want logged.
    for sep in ("?", "#"):
        idx = path.find(sep)
        if idx != -1:
            path = path[:idx]
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
    """Return the per-path child logger, attaching a :class:`FileHandler` if needed.

    Each distinct audit-log path gets its own child logger
    (``eurpe.security.audit.<hash>``) with its own handler. Two gates
    pointing at different files share NO handlers, so a write to one
    file cannot leak into the other. ``propagate=False`` on every
    child stops the line from bubbling to the parent / root.

    Caching by absolute path means:

    * multiple gates writing to the SAME path share one handler (no
      duplicate writes);
    * tests using ``tmp_path`` get their own child logger per path
      and don't leak lines into a sibling test's log.

    The parent directory MUST already exist — the gate guarantees this
    via ``EurpeConfig.runtime_dir`` being created by
    :func:`eurpe.config.ensure_runtime_dirs`. We assert that contract
    here so the failure mode is loud (a missing runtime_dir is a
    bootstrapping bug, not an audit-log bug).
    """

    logger = logging.getLogger(_child_logger_name(audit_log_path))
    # Don't propagate to the parent / root — these lines must not leak
    # into stdout or any other handler the application installs for
    # general logging, AND they must not leak into a sibling path's
    # child logger via the shared parent.
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
            except Exception:  # pragma: no cover - defensive; best-effort cleanup  # nosec B110
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
        "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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

    with _LOCK:
        for key, handler in list(_HANDLERS.items()):
            # ``key`` is the resolved absolute path string used to
            # build the child-logger name; reconstruct it to detach.
            child = logging.getLogger(_child_logger_name(Path(key)))
            try:
                child.removeHandler(handler)
                handler.close()
            except Exception:  # pragma: no cover - defensive; best-effort cleanup  # nosec B110
                pass
        _HANDLERS.clear()
