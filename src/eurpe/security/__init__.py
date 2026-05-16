"""Network policy gate, allowlisting, and audit logging.

Implements the offline-mode invariant from issue #12:

    * Offline mode is enabled by default for proposal processing.
    * Any outbound request attempt is logged locally without storing
      proposal content, prompts, retrieved passages, or generated draft
      text.
    * Release smoke test fails if a non-opt-in outbound request
      succeeds.

The package is a leaf module — it MUST NOT import from
:mod:`eurpe.retrieval` or :mod:`eurpe.generation`. Those packages
import *from* here (gate + factory), giving us a clean one-way
dependency that prevents cycles.

Public surface (re-exported here):

* :class:`NetworkPolicyGate` — the gate consulted before every egress.
* :class:`AllowlistEntry` — typed allowlist row.
* :class:`Decision` — ALLOWED / DENIED enum used in the audit log.
* :class:`EgressDeniedError` — raised by the gate on deny.
* :class:`SecurityError` — base for any failure in this package.
* :func:`make_network_policy` — config → gate factory.
"""

from eurpe.security.allowlist import AllowlistEntry
from eurpe.security.errors import EgressDeniedError, SecurityError
from eurpe.security.policy import Decision, NetworkPolicyGate, make_network_policy

__all__ = [
    "AllowlistEntry",
    "Decision",
    "EgressDeniedError",
    "NetworkPolicyGate",
    "SecurityError",
    "make_network_policy",
]
