"""Service facade for ingesting a parsed proposal into the index.

The ingestion service is the UI-independent seam for the common
"parse a PDF, optionally check for duplicates, then chunk+upsert"
chain. Before this issue the chain was duplicated between
``api/routes/ingest.py::confirm`` and ``retrieval/cli.py``; the
duplicate-detection branch (``REINDEX`` vs ``BLOCK_*`` vs ``NONE``)
was open-coded in each. Lifting it here keeps the four-way branch on
one well-tested code path.

What the service does NOT own
-----------------------------
HTTP-layer concerns — staging tokens, archive copying, YAML sidecar
persistence — stay in :mod:`eurpe.api.routes.ingest`. Those are
artefacts of the two-step upload+confirm flow the React UI uses; a
CLI batch ingest never produces them. Encoding them into the service
would push HTTP semantics down into a layer that has no business
knowing about them.

What this service DOES own
--------------------------
* Running the parser when only a :class:`pathlib.Path` is supplied.
* Computing the content hash (so the duplicate-detection contract on
  ``content_hash`` is met from one place).
* Calling :func:`eurpe.retrieval.dedup.evaluate_duplicate` and
  branching on the result.
* Calling :func:`eurpe.retrieval.pipeline.index_proposal` /
  :func:`reindex_proposal` based on the duplicate decision.
* Returning a structured :class:`IngestionResult` that carries the
  chunk count + the duplicate decision so callers can surface a
  consistent operator message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eurpe.ingestion.docling_parser import DoclingProposalParser
from eurpe.ingestion.hashing import compute_content_hash
from eurpe.ingestion.models import ParsedProposal
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.retrieval.dedup import (
    DuplicateAction,
    DuplicateDecision,
    evaluate_duplicate,
)
from eurpe.retrieval.index import ChromaIndex
from eurpe.retrieval.pipeline import index_proposal, reindex_proposal
from eurpe.schema import ProposalMetadata


class IngestionRequest(BaseModel):
    """Input to :meth:`IngestionService.ingest_proposal`.

    Two construction modes:

    * ``parsed`` already in hand (HTTP confirm route already parsed
      once at the ``/parse`` step).
    * ``pdf_path`` only (CLI batch ingest) — the service parses lazily
      using the supplied parser.

    Exactly one of ``parsed`` / ``pdf_path`` must be set; the validator
    enforces this at construction time so a caller mixing both gets a
    deterministic :class:`pydantic.ValidationError` rather than a
    silent parser re-run inside the service.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: ProposalMetadata = Field(
        description=(
            "Confirmed proposal metadata. ``content_hash`` may be unset; "
            "the service computes it from ``pdf_path`` when needed."
        )
    )
    parsed: ParsedProposal | None = Field(
        default=None,
        description="Pre-parsed proposal output from DoclingProposalParser.parse().",
    )
    pdf_path: Path | None = Field(
        default=None,
        description="On-disk PDF to parse, if ``parsed`` is not supplied.",
    )
    force: bool = Field(
        default=False,
        description=(
            "Override BLOCK_SOFT duplicate decisions and proceed with a "
            "reindex. Mirrors ``ConfirmRequest.force`` so HTTP/CLI behave the "
            "same way."
        ),
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "The ``CitationAnchor.document_id`` to use for the new chunks. "
            "Defaults to the PDF filename stem; the HTTP route supplies the "
            "archive-stem variant."
        ),
    )

    @model_validator(mode="after")
    def _enforce_parsed_xor_pdf_path(self) -> Self:
        """Reject ``IngestionRequest`` instances with both / neither input set.

        Pydantic catches the error at construction time so FastAPI's
        ``model_validate`` step returns a 422 with the field-level
        message instead of letting the service raise a ``ValueError``
        deeper in the pipeline.
        """

        if self.parsed is not None and self.pdf_path is not None:
            raise ValueError(
                "IngestionRequest must supply exactly one of `parsed` or `pdf_path`."
            )
        if self.parsed is None and self.pdf_path is None:
            raise ValueError(
                "IngestionRequest must supply either `parsed` or `pdf_path`."
            )
        return self


class IngestionResult(BaseModel):
    """Output of :meth:`IngestionService.ingest_proposal`.

    Carries everything the existing HTTP/CLI call sites already report:

    * ``chunks_added`` — number of chunks upserted into the index.
    * ``duplicate_decision`` — what the dedup layer decided. ``NONE``
      for the happy path; ``REINDEX`` / forced ``BLOCK_SOFT`` for the
      replace path. ``BLOCK_HARD`` and refused ``BLOCK_SOFT`` raise
      instead of returning so the caller can map to HTTP 409 / CLI
      stderr without inspecting the result.
    * ``replaced_document_id`` — the existing record's stem that was
      deleted on REINDEX or forced replace; ``None`` on a fresh
      ingest. Mirrors ``ConfirmResponse.replaced_document_id``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunks_added: int = Field(ge=0)
    duplicate_decision: DuplicateAction = Field(default=DuplicateAction.NONE)
    duplicate_reason: str | None = Field(default=None)
    replaced_document_id: str | None = Field(default=None)


class DuplicateRefusedError(Exception):
    """Raised when a hard duplicate (or an un-forced soft duplicate) blocks ingest.

    The caller decides how to surface this — HTTP route turns it into
    a 409, CLI logs and exits 1. Carrying the
    :class:`DuplicateDecision` lets both pick the right verb.
    """

    def __init__(self, decision: DuplicateDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class IngestionService:
    """Orchestrate parse → duplicate-check → chunk → upsert.

    Stateless aside from the injected dependencies. The same instance
    can serve many ingests in sequence and many ingests in parallel
    (the underlying parser/chunker/index are thread-safe).
    """

    def __init__(
        self,
        *,
        parser: DoclingProposalParser,
        chunker: HierarchicalChunker,
        index: ChromaIndex,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._index = index

    def ingest_proposal(self, request: IngestionRequest) -> IngestionResult:
        """Run the full ingest chain for one proposal.

        Errors:

        * :class:`DuplicateRefusedError` — duplicate detection refused the
          ingest (BLOCK_HARD, or BLOCK_SOFT without ``force=True``).
        * :class:`eurpe.ingestion.errors.IngestionError` — propagated
          from the parser.
        * :class:`eurpe.retrieval.errors.IndexingError` — propagated
          from the index.
        """

        parsed = self._resolve_parsed(request)
        proposal_with_hash = self._ensure_content_hash(request)

        document_id = request.document_id or self._derive_document_id(request, parsed)

        decision = evaluate_duplicate(
            index=self._index,
            content_hash=proposal_with_hash.content_hash or "",
            proposal_title=proposal_with_hash.proposal_title,
            call_id=proposal_with_hash.call_id,
            new_document_id=document_id,
        )
        if decision.action is DuplicateAction.BLOCK_HARD:
            raise DuplicateRefusedError(decision)
        if decision.action is DuplicateAction.BLOCK_SOFT and not request.force:
            raise DuplicateRefusedError(decision)

        replaced_document_id: str | None = None
        if decision.action in (DuplicateAction.REINDEX, DuplicateAction.BLOCK_SOFT):
            replaced_document_id = decision.conflicting_document_id

        if replaced_document_id is not None:
            chunks_added = reindex_proposal(
                parsed,
                proposal_with_hash,
                chunker=self._chunker,
                index=self._index,
                replaced_document_id=replaced_document_id,
            )
        else:
            chunks_added = index_proposal(
                parsed,
                proposal_with_hash,
                chunker=self._chunker,
                index=self._index,
            )

        duplicate_reason = (
            decision.reason if decision.action is not DuplicateAction.NONE else None
        )
        return IngestionResult(
            chunks_added=chunks_added,
            duplicate_decision=decision.action,
            duplicate_reason=duplicate_reason,
            replaced_document_id=replaced_document_id,
        )

    def _resolve_parsed(self, request: IngestionRequest) -> ParsedProposal:
        # The IngestionRequest model_validator already enforces the
        # parsed-XOR-pdf_path invariant, so the branches below are
        # exhaustive without needing duplicate runtime checks.
        if request.parsed is not None:
            return request.parsed
        assert request.pdf_path is not None  # narrowed by the validator
        return self._parser.parse(request.pdf_path)

    def _ensure_content_hash(self, request: IngestionRequest) -> ProposalMetadata:
        proposal = request.proposal
        if proposal.content_hash:
            return proposal
        if request.pdf_path is None:
            # Caller pre-parsed but forgot the hash — without bytes on
            # disk we can't compute one. Surface as a programmer error.
            raise ValueError(
                "IngestionRequest.proposal.content_hash is missing and no "
                "pdf_path was supplied to compute it."
            )
        return proposal.model_copy(
            update={"content_hash": compute_content_hash(request.pdf_path)}
        )

    @staticmethod
    def _derive_document_id(request: IngestionRequest, parsed: ParsedProposal) -> str:
        # CLI default: the on-disk filename stem. The HTTP route
        # overrides this with the archive stem because the staged
        # token-prefixed filename would leak into chunk metadata
        # otherwise.
        if request.pdf_path is not None:
            return request.pdf_path.stem
        if parsed.source_path:
            return Path(parsed.source_path).stem
        raise ValueError(
            "IngestionService cannot derive document_id: supply "
            "IngestionRequest.document_id or pdf_path."
        )
