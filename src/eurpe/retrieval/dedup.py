"""Duplicate-detection decision for incoming proposal ingests.

The HTTP confirm route and the ``eurpe index build`` CLI share the same
question before they upsert chunks into the index: *should this proposal
be allowed to proceed, replace something, or be rejected outright?* The
answer depends on what is already in the index, and we want both
callers to compute it the same way so the operator's experience is
consistent.

Three cases are recognised, encoded in :class:`DuplicateAction`:

* ``BLOCK_HARD`` — the incoming PDF is byte-identical to one already in
  the index. Re-ingesting it would do nothing useful. The HTTP route
  surfaces a 409; the CLI logs a skip line to stderr, continues the
  batch, and exits 0 (the per-bucket summary reports the skip count).
* ``REINDEX`` — the incoming PDF has a different hash but its archive
  filename (``document_id``) already exists. This is the
  corrected-version case: the operator dropped in an updated PDF under
  the same archive name. Callers automatically delete-then-upsert.
* ``BLOCK_SOFT`` — different hash and different document_id, but the
  ``(proposal_title, call_id)`` pair already names another record.
  Callers warn the operator and refuse to proceed unless ``force=True``
  is explicitly set. When forced, the existing record is replaced.

``NONE`` is the happy path — no duplicate suspected.

The function is intentionally pure (no I/O outside the index query) so
unit-testing the four branches needs nothing beyond a real
:class:`ChromaIndex` and an embedder stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from eurpe.retrieval.index import ChromaIndex


class DuplicateAction(StrEnum):
    """How the caller should respond to a duplicate-detection result."""

    #: No duplicate suspected; proceed with a fresh upsert.
    NONE = "none"
    #: Corrected version of an existing document — delete-then-upsert.
    REINDEX = "reindex"
    #: Byte-identical content already in the index; refuse unconditionally.
    BLOCK_HARD = "block_hard"
    #: Title+call match but the new bytes differ; block unless forced.
    BLOCK_SOFT = "block_soft"


@dataclass(frozen=True)
class DuplicateDecision:
    """Result of :func:`evaluate_duplicate`.

    ``conflicting_document_id`` is the existing record's archive stem
    that the caller would delete on a forced replace or a reindex; it
    is ``None`` for ``NONE`` and may be set for ``BLOCK_HARD`` (purely
    informational — the operator can find the existing record).
    """

    action: DuplicateAction
    reason: str
    conflicting_document_id: str | None = None


def evaluate_duplicate(
    *,
    index: ChromaIndex,
    content_hash: str,
    proposal_title: str | None,
    call_id: str,
    new_document_id: str,
) -> DuplicateDecision:
    """Decide what to do with an incoming proposal given the current index.

    Parameters
    ----------
    index:
        Open :class:`ChromaIndex` to query. The function is read-only.
    content_hash:
        Lowercase 64-char sha256 hex of the incoming PDF bytes — produced
        by :func:`eurpe.ingestion.hashing.compute_content_hash`.
    proposal_title:
        Optional title from the confirmed metadata. ``None`` or empty
        disables the title+call_id branch (see module docstring).
    call_id:
        Call identifier as it will be stored on the new record.
    new_document_id:
        The :attr:`CitationAnchor.document_id` the caller intends to use
        for the new chunks (HTTP route: archive-filename stem; CLI: PDF
        filename stem).

    Returns
    -------
    DuplicateDecision
        Carries the action plus a human-readable reason suitable for
        surfacing verbatim in an HTTP error body or a CLI warning line.

    Order of checks
    ---------------
    1. **Hash match** anywhere in the index → ``BLOCK_HARD``. Re-uploading
       byte-identical content is never useful regardless of whether the
       existing record's document_id matches.
    2. **document_id match** with no hash match → ``REINDEX``. The
       operator dropped a corrected PDF under the same archive name.
    3. **Title + call_id match** elsewhere (and the title is set on both
       sides) → ``BLOCK_SOFT``. The caller decides whether to honour
       ``force=True``.
    4. Otherwise → ``NONE``.
    """

    hash_matches = index.find_by_content_hash(content_hash)
    if hash_matches:
        return DuplicateDecision(
            action=DuplicateAction.BLOCK_HARD,
            reason=(
                f"PDF is byte-identical to existing document "
                f"{hash_matches[0]!r} (sha256 {content_hash})."
            ),
            conflicting_document_id=hash_matches[0],
        )

    if index.find_by_document_id(new_document_id) > 0:
        return DuplicateDecision(
            action=DuplicateAction.REINDEX,
            reason=(
                f"document_id {new_document_id!r} already exists with different "
                "bytes; re-indexing in place."
            ),
            conflicting_document_id=new_document_id,
        )

    title_matches = index.find_by_title_and_call(proposal_title, call_id)
    # ``find_by_title_and_call`` already returns ``[]`` when title is
    # None/empty, so we do not have to re-check the title here. Exclude
    # ``new_document_id`` from the matches because a corrected version
    # under the same stem would otherwise look like a soft-duplicate.
    other_matches = [d for d in title_matches if d != new_document_id]
    if other_matches:
        return DuplicateDecision(
            action=DuplicateAction.BLOCK_SOFT,
            reason=(
                f"a different document {other_matches[0]!r} already exists with "
                f"the same proposal_title={proposal_title!r} and call_id={call_id!r}."
            ),
            conflicting_document_id=other_matches[0],
        )

    return DuplicateDecision(
        action=DuplicateAction.NONE,
        reason="no duplicate detected",
        conflicting_document_id=None,
    )
