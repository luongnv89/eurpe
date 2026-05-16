"""Exception hierarchy for :mod:`eurpe.security`.

A single base (:class:`SecurityError`) so callers — most importantly the
embedder / LLM factories and the smoke CLI — can ``except SecurityError``
once and surface a clean ``error: ...`` line on every security-layer
failure. Mirrors the ``IndexingError`` / ``IngestionError`` shape used
elsewhere in the codebase so the package feels idiomatic alongside its
siblings.

Subclasses identify *category* rather than *cause*:

* :class:`EgressDeniedError` — raised by
  :class:`~eurpe.security.policy.NetworkPolicyGate` when an outbound
  request is denied. The error message is intentionally short and never
  contains body / header / query data — the audit log is the canonical
  record, this exception is only the control-flow signal.
"""

from __future__ import annotations


class SecurityError(Exception):
    """Base class for any failure inside :mod:`eurpe.security`.

    Catch this in callers (CLI, factories) when you want one branch for
    "the security layer blocked or broke". Subclasses carry more
    specific intent for handlers that care (e.g., :class:`EgressDeniedError`).
    """


class EgressDeniedError(SecurityError):
    """Raised when :class:`NetworkPolicyGate.check` denies an outbound request.

    The message names only ``host:port`` and the call source label — never
    the request body, headers, or query string. The full structured record
    of the denial lives in the JSONL audit log at
    ``cfg.runtime_dir / "network-audit.log"``; this exception is a
    control-flow signal, not a data carrier.
    """
