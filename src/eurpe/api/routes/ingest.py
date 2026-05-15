"""HTTP routes for the two-step proposal-ingestion flow.

Flow shape (issue #10, AC #1)
-----------------------------
1. The operator drag-drops a PDF in the React UI. The browser POSTs the
   file to ``POST /api/ingestion/parse``. The route stages the upload
   under ``cfg.runtime_dir/staging``, runs Docling, makes a best-effort
   draft :class:`ProposalMetadata` (programme / call / topic inferred
   from the filename via the existing intake extractor), and hands back
   a :class:`ParseResponse` with a ``parse_token``.

2. The UI renders :class:`~eurpe.api.schemas.ConfirmRequest` fields
   pre-populated with the draft. The operator reviews, edits anything
   wrong (programme / call_id / topic_id / outcome are *required* and
   the form rejects empties — AC #2), and POSTs the final values plus
   the ``parse_token`` to ``POST /api/ingestion/confirm``.

3. ``POST /api/ingestion/confirm`` reconstructs the
   :class:`ProposalMetadata` server-side (filling ``source_path`` from
   the staged file the token points at), persists a YAML sidecar next
   to the staged PDF (AC #3), and runs
   :func:`eurpe.retrieval.pipeline.index_proposal` to chunk and index
   the proposal.

A third helper route ``GET /api/ingestion/enums`` returns the closed
enum vocabularies (programme, source_status) so the UI Selects never
hand-type strings — adding a new enum member upstream automatically
appears in the form.

Privacy guards
--------------
* ``DoclingProposalParser`` is constructed with ``offline=cfg.offline_mode``
  via the dependency provider. In offline mode (the production default)
  Docling does not download OCR weights.
* Every route returns 422 / 404 on validation problems rather than 500
  so an operator can reason about failures without reading the server log.
* The ``filename`` arriving on the multipart form is sanitised for path
  traversal *before* it is joined with ``staging_dir`` — see
  :func:`_safe_pdf_filename`. Python's path resolution would silently
  consume ``../`` if the validation ran on the joined path instead.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from eurpe.api.dependencies import (
    get_chunker,
    get_config,
    get_index,
    get_parser,
    get_token_store,
)
from eurpe.api.schemas import (
    ConfirmRequest,
    ConfirmResponse,
    EnumsResponse,
    ParseResponse,
)
from eurpe.api.storage import ParseTokenStore
from eurpe.config import EurpeConfig
from eurpe.ingestion.docling_parser import DoclingProposalParser
from eurpe.ingestion.errors import IngestionError
from eurpe.intake.extractor import extract_topic_context_from_text
from eurpe.retrieval import (
    ChromaIndex,
    HierarchicalChunker,
    IndexingError,
    index_proposal,
)
from eurpe.schema import Programme, ProposalMetadata, SourceStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe_pdf_filename(raw: str | None) -> str:
    """Return the basename of ``raw`` after rejecting traversal attempts.

    The filename arrives on a multipart form field controlled by the
    client. We refuse anything that:

    * is missing or empty,
    * contains a path separator (``/`` or ``\\``),
    * contains ``..`` anywhere (parent-directory escape),
    * does not end in ``.pdf`` (case-insensitive),
    * collapses to nothing once stripped.

    Validating the *raw* string is what closes the path-traversal hole;
    Python's :class:`Path` would silently resolve ``../etc/passwd.pdf``
    against the staging dir to a perfectly valid string and we'd never
    see the ``..`` part. The basename is returned for the staging path
    so even a "harmless" subdirectory name cannot accidentally be honoured.
    """

    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload missing filename",
        )
    stripped = raw.strip()
    if not stripped or "/" in stripped or "\\" in stripped or ".." in stripped:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsafe filename: {raw!r}",
        )
    if not stripped.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only .pdf uploads are accepted",
        )
    return os.path.basename(stripped)


def _stage_upload(
    upload: UploadFile, *, store: ParseTokenStore, token: str, safe_name: str
) -> Path:
    """Copy ``upload`` to ``store.staging_dir`` and return the on-disk path.

    Filename layout: ``<token>__<safe_name>``. Prefixing with the token
    keeps two uploads of the same filename from clobbering each other in
    the staging dir; the human-readable suffix is preserved so an operator
    debugging from a shell sees what the file *is*.
    """

    target = store.staging_dir / f"{token}__{safe_name}"
    # Stream the upload through a tmp+replace pattern so a half-written
    # file cannot be observed by ``parser.parse``. The tmp suffix is
    # chosen so it is filtered out by ``parser.supports`` even if a
    # crash leaves it behind (``.tmp`` ≠ ``.pdf``).
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    upload.file.seek(0)
    with tmp_path.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    os.replace(tmp_path, target)
    return target


def _build_draft(original_filename: str, parsed_title: str | None) -> dict:
    """Best-effort metadata draft from the operator's filename + parsed title.

    Reuses :func:`extract_topic_context_from_text` to recover programme /
    call_id / topic_id from the filename string. The function lives in
    :mod:`eurpe.intake.extractor` and intentionally duplicates the test
    helper's regex constants — see the extractor's module docstring for
    the rationale (production never imports from ``tests/``).

    Why ``original_filename`` instead of the staged path: the staged path
    is prefixed with the parse-token UUID (see :func:`_stage_upload`), and
    the regex extractor would happily match digits inside the UUID before
    the real call/topic identifiers in the operator's filename.
    """

    # Feed both the filename and the document title so a generous match
    # window catches programme tokens that appear in either.
    seed = original_filename
    if parsed_title:
        seed = f"{seed}\n{parsed_title}"
    ctx = extract_topic_context_from_text(seed)

    draft: dict = {}
    if ctx.programme is not None:
        draft["programme"] = ctx.programme.value
    if ctx.call_id:
        draft["call_id"] = ctx.call_id
    if ctx.topic_id:
        draft["topic_id"] = ctx.topic_id
    if parsed_title:
        draft["proposal_title"] = parsed_title
    return draft


def _persist_sidecar(
    pdf_path: Path,
    proposal: ProposalMetadata,
    *,
    runtime_dir: Path,
) -> Path:
    """Write a YAML sidecar capturing ``proposal`` next to a stable archive.

    The sidecar lives under ``<runtime_dir>/proposals/`` (NOT next to the
    staged PDF in ``<runtime_dir>/staging/``) so the confirm route can
    safely delete the staged file once chunks are indexed without losing
    the persisted metadata. Sidecar filename is the staged PDF's basename
    with ``.metadata.yaml`` appended.

    The serialisation goes through ``model_dump(mode="json")`` first to
    coerce :class:`SourceStatus` and :class:`Programme` (StrEnum members)
    back to plain strings — PyYAML's safe representer matches on exact
    type rather than subclass, so dumping the enum directly raises
    ``RepresenterError``. See ``SourceStatus.__doc__`` for the full
    explanation; the same pattern is used in
    ``tests/test_schema.py::test_serialise_via_model_dump_json``.
    """

    sidecar_dir = runtime_dir / "proposals"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"{pdf_path.stem}.metadata.yaml"
    body = proposal.model_dump(mode="json")
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    os.replace(tmp, sidecar_path)
    return sidecar_path


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/enums", response_model=EnumsResponse)
def get_enums() -> EnumsResponse:
    """Return the closed enum vocabularies the UI Selects must use.

    Sourced from the Python enums directly so adding a new
    :class:`Programme` or :class:`SourceStatus` member ripples into the
    UI on the next request — no string lists hand-typed in TypeScript.
    """

    return EnumsResponse(
        programme=[member.value for member in Programme],
        source_status=[member.value for member in SourceStatus],
    )


@router.post("/parse", response_model=ParseResponse)
def parse(
    file: UploadFile = File(...),
    cfg: EurpeConfig = Depends(get_config),
    parser: DoclingProposalParser = Depends(get_parser),
    store: ParseTokenStore = Depends(get_token_store),
) -> ParseResponse:
    """Stage an uploaded PDF, run Docling, return a draft + parse token.

    The route writes the upload to ``<runtime_dir>/staging`` under a
    token-prefixed name, runs the Docling parser, and persists the draft
    metadata + parsed-at timestamp to the parse-token store. The returned
    ``parse_token`` is what the UI sends back to ``/confirm``.

    A failed parse returns 400 (the file is bad) — the staging file is
    removed so we don't accumulate junk for known-bad uploads.
    """

    safe_name = _safe_pdf_filename(file.filename)
    token = store.new_token()
    try:
        pdf_path = _stage_upload(file, store=store, token=token, safe_name=safe_name)
    except OSError as exc:  # pragma: no cover - filesystem error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to stage upload: {exc}",
        ) from exc

    try:
        parsed = parser.parse(pdf_path)
    except IngestionError as exc:
        # The Docling failure is the operator's problem (bad PDF) — surface
        # the message verbatim and clean up the staged file so we don't
        # leave dead bytes in the runtime dir.
        try:
            pdf_path.unlink()
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed to parse PDF: {exc}",
        ) from exc

    draft = _build_draft(safe_name, parsed.title)
    record = store.put(
        token=token,
        pdf_path=pdf_path,
        draft=draft,
    )
    return ParseResponse(
        parse_token=record.token,
        source_path=str(pdf_path),
        parsed_at=record.parsed_at,
        suggested=draft,
        page_count=parsed.page_count,
        title=parsed.title,
        section_count=len(parsed.sections),
    )


@router.post("/confirm", response_model=ConfirmResponse)
def confirm(
    body: ConfirmRequest,
    cfg: EurpeConfig = Depends(get_config),
    parser: DoclingProposalParser = Depends(get_parser),
    chunker: HierarchicalChunker = Depends(get_chunker),
    index: ChromaIndex = Depends(get_index),
    store: ParseTokenStore = Depends(get_token_store),
) -> ConfirmResponse:
    """Persist the operator-confirmed metadata and index the staged PDF.

    Pydantic validation on :class:`ConfirmRequest` already guarantees the
    closed-enum invariants (``programme`` and ``outcome``) and the
    required-field rules (``call_id`` non-empty, ``year`` in [2014, 2099]).
    AC #2 of issue #10 is what motivates the `extra="forbid"` + closed-enum
    discipline on the wire model.

    The confirmed :class:`ProposalMetadata` is reconstructed *here*,
    server-side, with ``source_path`` taken from the staged PDF and
    ``ingested_at`` left to the model's default factory (UTC now). That
    keeps two server-controlled fields off the wire.
    """

    record = store.get(body.parse_token)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"unknown or expired parse_token: {body.parse_token}. "
                "Re-upload the PDF to start a new parse."
            ),
        )
    if not record.pdf_path.exists():
        # The staged PDF was deleted out from under us — treat as
        # expired so the operator gets a clear "re-upload" error.
        store.delete(body.parse_token)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="staged PDF is no longer available; re-upload to start a new parse.",
        )

    proposal = ProposalMetadata(
        programme=body.programme,
        call_id=body.call_id,
        topic_id=body.topic_id,
        year=body.year,
        outcome=body.outcome,
        proposal_title=body.proposal_title,
        consortium_acronym=body.consortium_acronym,
        source_path=str(record.pdf_path),
        language=body.language,
        ingested_at=datetime.now(UTC),
    )

    try:
        parsed = parser.parse(record.pdf_path)
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed to re-parse staged PDF: {exc}",
        ) from exc

    sidecar_path = _persist_sidecar(record.pdf_path, proposal, runtime_dir=cfg.runtime_dir)

    try:
        chunks_added = index_proposal(
            parsed,
            proposal,
            chunker=chunker,
            index=index,
        )
    except IndexingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"index upsert failed: {exc}",
        ) from exc

    # The token is consumed on success — the staged PDF is no longer
    # needed because the chunks are in the index and the sidecar
    # captures the metadata.
    store.delete(body.parse_token)

    return ConfirmResponse(
        chunks_added=chunks_added,
        collection=index.collection_name,
        sidecar_path=str(sidecar_path),
    )
