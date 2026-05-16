"""Typed allowlist entries for :class:`NetworkPolicyGate`.

A single Pydantic v2 model — :class:`AllowlistEntry` — describes one
``host:port`` exception to the default-deny network policy. The entry
carries a ``reason`` field that operators MUST fill in; the reason is
copied verbatim into the audit log so a future reviewer can answer
"why was this allowlisted?" without re-reading the config.

The model is intentionally minimal: no scheme, no path, no method. The
gate operates at the connection level, not the request level — once a
host:port is allowed, ALL requests to it are allowed. If you need
per-path allowlisting, that is a separate (more invasive) feature and
should be added with its own ADR.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AllowlistEntry(BaseModel):
    """One host:port exception to the default-deny network policy.

    Fields:

    * ``host`` — lowercase hostname or IP literal. Validated for shape
      only; we never resolve DNS at config-load time (resolution would
      be a network call, breaking the offline contract).
    * ``port`` — TCP port in the valid IANA range ``[1, 65535]``.
    * ``reason`` — free-text justification copied into the audit log.
      Required and non-empty so a future reviewer always sees *why*
      this host was added.
    """

    # Forbid extras: a typo like ``hots: example.com`` would silently
    # produce a useless entry. Strict-mode catches it at load time.
    model_config = ConfigDict(extra="forbid")

    host: str = Field(min_length=1, description="Lowercase hostname or IP literal.")
    port: int = Field(ge=1, le=65535, description="TCP port in [1, 65535].")
    reason: str = Field(min_length=1, description="Why this host is allowlisted.")

    @field_validator("host")
    @classmethod
    def _host_shape(cls, value: str) -> str:
        """Normalise + sanity-check the host string.

        Rejects schemes (``http://``), paths (``/foo``), whitespace, and
        empty strings. We do NOT do DNS lookup — that would be a
        network call, which violates the offline contract this whole
        package exists to enforce.
        """

        stripped = value.strip()
        if not stripped:
            raise ValueError("host must be non-empty")
        if " " in stripped or "\t" in stripped:
            raise ValueError(f"host must not contain whitespace; got {stripped!r}")
        if "://" in stripped:
            raise ValueError(
                f"host must not include a scheme; got {stripped!r}. "
                "Provide just the hostname (e.g., 'example.com'), not a URL."
            )
        if "/" in stripped:
            raise ValueError(
                f"host must not include a path; got {stripped!r}. "
                "Provide just the hostname, not a URL."
            )
        return stripped.lower()

    def matches(self, host: str, port: int) -> bool:
        """Return ``True`` iff ``host`` (case-insensitive) and ``port`` match.

        Used by :class:`NetworkPolicyGate` to scan its allowlist. Kept
        as a method (rather than overriding ``__eq__``) so equality
        between :class:`AllowlistEntry` instances stays
        Pydantic-default (field-by-field including ``reason``).
        """

        return self.host == host.lower() and self.port == port
