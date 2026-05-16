"""Tests for :class:`eurpe.security.allowlist.AllowlistEntry`.

The validator is the load-bearing piece: a malformed entry in
``config.yaml`` MUST fail at load time rather than silently producing
a no-op (which would leave the operator believing they had opted in to
a host they did not). Tests below pin every reject path plus the
case-insensitive ``.matches()`` semantics.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eurpe.security.allowlist import AllowlistEntry


def test_valid_entry_round_trips() -> None:
    entry = AllowlistEntry(host="Example.com", port=443, reason="test mirror")
    # Validator lowercases the host so a config typed as 'Example.com'
    # still matches an incoming 'example.com' from urllib.parse.
    assert entry.host == "example.com"
    assert entry.port == 443
    assert entry.reason == "test mirror"


def test_matches_is_case_insensitive() -> None:
    entry = AllowlistEntry(host="example.com", port=443, reason="x")
    assert entry.matches("EXAMPLE.COM", 443) is True
    assert entry.matches("Example.com", 443) is True
    assert entry.matches("example.com", 443) is True


def test_matches_requires_exact_port() -> None:
    entry = AllowlistEntry(host="example.com", port=443, reason="x")
    assert entry.matches("example.com", 443) is True
    assert entry.matches("example.com", 8443) is False


def test_matches_does_not_match_subdomain() -> None:
    """Allowlist must be host-exact — subdomains are NOT inherited.

    A bug here would silently broaden the policy: allowlisting
    ``example.com`` and matching ``evil.example.com`` would be a
    serious regression.
    """

    entry = AllowlistEntry(host="example.com", port=443, reason="x")
    assert entry.matches("evil.example.com", 443) is False
    assert entry.matches("example.com.evil", 443) is False


@pytest.mark.parametrize(
    "bad_host",
    [
        "",
        " ",
        "has whitespace",
        "http://example.com",
        "https://example.com",
        "example.com/path",
        "/example.com",
    ],
)
def test_invalid_host_shape_rejected(bad_host: str) -> None:
    with pytest.raises(ValidationError):
        AllowlistEntry(host=bad_host, port=443, reason="x")


@pytest.mark.parametrize("bad_port", [0, -1, 65536, 999999])
def test_invalid_port_rejected(bad_port: int) -> None:
    with pytest.raises(ValidationError):
        AllowlistEntry(host="example.com", port=bad_port, reason="x")


def test_reason_required_non_empty() -> None:
    """Empty ``reason`` must fail; the field is mandatory by design."""

    with pytest.raises(ValidationError):
        AllowlistEntry(host="example.com", port=443, reason="")


def test_extra_fields_rejected() -> None:
    """A typo like ``hots:`` must NOT silently become a valid entry."""

    with pytest.raises(ValidationError):
        AllowlistEntry.model_validate(
            {"host": "example.com", "port": 443, "reason": "x", "hots": "typo"}
        )
