"""Unit tests for :func:`eurpe.ingestion.hashing.compute_content_hash`.

The util is small but load-bearing — the dedup helper rejects an
incoming proposal as a hard duplicate solely on the strength of the
returned digest. The tests below pin three behaviours:

1. Shape — the digest is 64 lowercase hex chars (sha256 hexdigest),
   matching the regex enforced by :class:`ProposalMetadata`.
2. Stability — identical bytes hash identically; a single-byte change
   produces a different digest.
3. Streaming — a multi-megabyte file is hashed without loading the
   whole buffer at once. We do not measure memory directly; instead we
   verify a known-vector digest against a synthesised 2 MiB file, which
   exercises the loop body (the hash matches the canonical sha256 of
   zeros only if every read block fed in).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from eurpe.ingestion.hashing import compute_content_hash


def test_compute_content_hash_returns_64_lowercase_hex(tmp_path: Path) -> None:
    """The digest is the lowercase 64-char sha256 hexdigest of file bytes."""

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"hello world")
    digest = compute_content_hash(pdf)
    assert len(digest) == 64
    assert digest.islower()
    # Known canonical sha256 of b"hello world".
    assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_compute_content_hash_is_deterministic(tmp_path: Path) -> None:
    """Hashing the same bytes twice yields the same digest."""

    pdf = tmp_path / "stable.pdf"
    pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 1024)
    a = compute_content_hash(pdf)
    b = compute_content_hash(pdf)
    assert a == b


def test_compute_content_hash_changes_with_one_byte_change(tmp_path: Path) -> None:
    """A one-byte change to the file produces a different digest."""

    pdf = tmp_path / "diff.pdf"
    pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 1024)
    first = compute_content_hash(pdf)

    pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 1023 + b"y")
    second = compute_content_hash(pdf)
    assert first != second


def test_compute_content_hash_streams_multimegabyte_file(tmp_path: Path) -> None:
    """A 2 MiB file hashes to the canonical sha256 of its bytes.

    Exercises the streaming loop — the digest matches the canonical
    value only if every 64 KiB block was fed in, so a silently broken
    loop would produce a different result.
    """

    pdf = tmp_path / "big.pdf"
    # 2 MiB of NUL bytes. Chosen because zero-bytes have a canonical
    # sha256 we can compute up front in this same test.
    payload = b"\x00" * (2 * 1024 * 1024)
    pdf.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert compute_content_hash(pdf) == expected
