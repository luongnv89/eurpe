"""Streamed sha256 hashing of PDF (or any binary) file contents.

Used by the duplicate-detection layer so an incoming upload can be
compared against the ``proposal.content_hash`` field stored on every
chunk in the index. The function streams the file in fixed-size chunks
rather than loading the full byte sequence into memory — proposals are
often tens of megabytes and the HTTP request handler runs synchronously,
so a non-streaming hash would bloat both peak memory and request
latency.

The chunk size (64 KiB) is a deliberate middle ground: large enough that
syscall overhead disappears on real-world disks, small enough that the
working set stays inside the L2 cache of every machine we ship to.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 64 KiB per read — see module docstring for the rationale.
_CHUNK_BYTES = 64 * 1024


def compute_content_hash(pdf_path: Path) -> str:
    """Stream-hash the bytes at ``pdf_path`` and return the lowercase sha256 hex.

    ``hashlib.sha256().hexdigest()`` returns lowercase 64 hex characters,
    matching the :func:`ProposalMetadata._content_hash_well_formed`
    validator's regex. Reads happen in 64 KiB blocks so a multi-megabyte
    proposal never lands in memory at once.
    """

    digest = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        # ``iter`` with a sentinel reads until EOF; the partial trailing
        # block (< _CHUNK_BYTES) is included verbatim by .read(n).
        for block in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
