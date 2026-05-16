"""Network policy gate: default-deny with a typed allowlist.

The :class:`NetworkPolicyGate` is the single chokepoint every egress
call site MUST consult before opening a connection. The gate's
``.check(host, port, scheme, path, source)`` method returns a
:class:`Decision` and writes one JSONL line to the audit log,
regardless of the outcome:

* :attr:`Decision.ALLOWED` — the call site MAY proceed to open the
  connection.
* :attr:`Decision.DENIED` — the gate raises :class:`EgressDeniedError`
  before returning, so a caller that ignored the return value still
  cannot proceed.

The allow rules, in priority order, are:

1. **Loopback fast-path.** ``localhost`` and any IP that
   :class:`ipaddress.ip_address` reports as ``is_loopback`` are
   allowed without consulting the allowlist. This keeps the Ollama
   daemon at ``localhost:11434`` reachable in the default
   offline-mode configuration — the offline contract is "no outbound
   *Internet* traffic", not "no local IPC".
2. **Allowlist match.** A typed :class:`AllowlistEntry` whose
   ``host`` (case-insensitive) and ``port`` match.
3. **Otherwise: DENY.**

The factory :func:`make_network_policy` mirrors the
``make_embedder`` / ``make_llm_client`` pattern: takes a config-shaped
object (duck-typed to avoid an import cycle through ``eurpe.config``)
and returns a configured gate.
"""

from __future__ import annotations

import enum
import ipaddress
import logging
from pathlib import Path

from eurpe.security.allowlist import AllowlistEntry
from eurpe.security.audit import log_attempt
from eurpe.security.errors import EgressDeniedError

logger = logging.getLogger(__name__)


class Decision(enum.StrEnum):
    """Result of a :meth:`NetworkPolicyGate.check`.

    String-valued so the variant name appears verbatim in the JSONL
    audit log without a custom serialiser.
    """

    ALLOWED = "ALLOWED"
    DENIED = "DENIED"


def _is_loopback(host: str) -> bool:
    """Return ``True`` for loopback hostnames and IP literals.

    Handles three cases:

    * ``localhost`` (case-insensitive) — not a valid input to
      :func:`ipaddress.ip_address`, so we shortcut it.
    * IPv4 literals like ``127.0.0.1``.
    * IPv6 literals like ``::1`` or ``[::1]``.

    Anything that isn't a recognised IP literal AND isn't ``localhost``
    is treated as a non-loopback hostname — we do NOT do DNS
    resolution (would be a network call, which the gate exists to
    prevent).
    """

    if not host:
        return False
    normalised = host.lower().strip()
    if normalised == "localhost":
        return True
    # Strip brackets from IPv6 literals (``[::1]``) before parsing.
    if normalised.startswith("[") and normalised.endswith("]"):
        normalised = normalised[1:-1]
    try:
        return ipaddress.ip_address(normalised).is_loopback
    except ValueError:
        return False


class NetworkPolicyGate:
    """Default-deny gate consulted before every outbound network call.

    Construct one via :func:`make_network_policy` (factory wires the
    config); direct construction is fine for tests that want a
    specific allowlist.

    Thread-safety: the gate has no mutable state beyond the
    immutable allowlist tuple it captures at ``__init__``, and the
    audit log writer (:mod:`eurpe.security.audit`) is internally
    serialised by Python's logging machinery. Calling ``.check`` from
    multiple threads is safe.
    """

    def __init__(
        self,
        *,
        allowlist: list[AllowlistEntry] | tuple[AllowlistEntry, ...] | None,
        audit_log_path: Path,
        offline_mode: bool = True,
    ) -> None:
        # Freeze the allowlist into a tuple so callers can't mutate the
        # gate's policy after construction without going through a
        # rebuild — making behaviour easier to reason about.
        self._allowlist: tuple[AllowlistEntry, ...] = tuple(allowlist or ())
        self._audit_log_path = audit_log_path
        self._offline_mode = bool(offline_mode)

    @property
    def offline_mode(self) -> bool:
        """Whether the gate was constructed under offline mode.

        Currently advisory — the decision logic is identical in both
        modes today (loopback + allowlist). Exposed so tests and the
        CLI can render the current posture without re-reading config.
        """

        return self._offline_mode

    @property
    def audit_log_path(self) -> Path:
        """Absolute path of the JSONL audit log this gate writes to."""

        return self._audit_log_path

    @property
    def allowlist(self) -> tuple[AllowlistEntry, ...]:
        """Read-only view of the configured allowlist entries."""

        return self._allowlist

    def check(
        self,
        host: str,
        port: int,
        scheme: str,
        path: str,
        source: str,
    ) -> Decision:
        """Decide ALLOWED or DENIED for ``(host, port)`` and audit it.

        Parameters mirror the fields written to the JSONL log:

        * ``host`` — hostname or IP literal. Case-insensitive
          comparison; lowercased in the log.
        * ``port`` — TCP port (int).
        * ``scheme`` — informational for the log only (``http``,
          ``https``, ``tcp``, ...). Does not affect the decision.
        * ``path`` — request path. Recorded redacted to first segment
          only; never used in the decision.
        * ``source`` — call-site label like ``ollama_embedder.embed``;
          required so an auditor can find the originating call site.

        Returns :class:`Decision.ALLOWED` when the call may proceed.
        On deny, raises :class:`EgressDeniedError` and never returns —
        callers that catch the exception observe the raise, not a
        ``Decision.DENIED`` return value. The return type is annotated
        as :class:`Decision` (rather than ``Literal[Decision.ALLOWED]``)
        so the enum remains the single source of truth for downstream
        readers; the runtime guarantee is "allow → return, deny → raise".
        """

        decision, reason = self._decide(host, port)
        log_attempt(
            self._audit_log_path,
            host=host,
            port=port,
            scheme=scheme,
            path=path,
            decision=decision.value,
            reason=reason,
            source=source,
        )
        if decision is Decision.DENIED:
            # Message is short and free of body / header / query data
            # by design — the audit log is the canonical record.
            raise EgressDeniedError(
                f"Egress denied: {host}:{port} from {source!r} "
                f"({reason}). See {self._audit_log_path} for the audit record."
            )
        return decision

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _decide(self, host: str, port: int) -> tuple[Decision, str]:
        """Pure decision: loopback → allow, allowlist → allow, else deny.

        Separated from :meth:`check` so unit tests can pin the
        decision logic without the side effect of writing the audit
        log. Returns ``(decision, reason)`` where ``reason`` is the
        short human-readable rationale that goes into the log.
        """

        if _is_loopback(host):
            return Decision.ALLOWED, "loopback"
        for entry in self._allowlist:
            if entry.matches(host, port):
                return Decision.ALLOWED, f"allowlist match: {entry.reason}"
        return Decision.DENIED, "not on allowlist"


def make_network_policy(config: object) -> NetworkPolicyGate:
    """Build a :class:`NetworkPolicyGate` for ``config``.

    Mirrors :func:`eurpe.retrieval.embeddings.make_embedder` —
    duck-typed access to the fields we need, so callers can pass
    either an :class:`~eurpe.config.EurpeConfig` or a partial mock in
    tests.

    Fields consulted on ``config``:

    * ``offline_mode`` — bool, defaults to ``True``.
    * ``network_allowlist`` — list of :class:`AllowlistEntry`, defaults
      to ``[]`` (empty = secure default, only loopback allowed).
    * ``network_audit_log_path()`` — callable returning the absolute
      Path of the JSONL audit log. If absent, falls back to
      ``runtime_dir / "network-audit.log"``.
    """

    offline_mode = bool(getattr(config, "offline_mode", True))
    allowlist = getattr(config, "network_allowlist", None) or []

    # Prefer the method (config-validated path) so a future config
    # change that moves the file is reflected in one place. Fall back
    # to the runtime_dir join for partial mocks in tests.
    audit_log_getter = getattr(config, "network_audit_log_path", None)
    if callable(audit_log_getter):
        audit_log_path = audit_log_getter()
    else:
        runtime_dir = getattr(config, "runtime_dir", None)
        if runtime_dir is None:
            raise ValueError(
                "make_network_policy requires config.runtime_dir or "
                "config.network_audit_log_path() to be set."
            )
        audit_log_path = Path(runtime_dir) / "network-audit.log"

    return NetworkPolicyGate(
        allowlist=list(allowlist),
        audit_log_path=Path(audit_log_path),
        offline_mode=offline_mode,
    )
