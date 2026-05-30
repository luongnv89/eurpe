"""HTTP routes for the React section-drafting workspace (issue #15).

Three endpoints back the drafting workspace:

* ``GET /api/generation/enums`` — closed enum vocabularies for the UI
  Selects (``section_type`` + ``programme``). Mirrors how
  ``/api/ingestion/enums`` keeps the ingestion form in sync with the
  Python enums.
* ``GET /api/generation/profiles`` — the drafting profiles bundled with
  this build. The UI Select pre-populates with this list and posts the
  ``programme`` value back to ``/section`` as ``profile_programme``.
* ``POST /api/generation/section`` — drive
  :class:`~eurpe.generation.GenerationService.generate_section` end-to-end
  and return the resulting draft + citations. The route is intentionally
  thin: input validation is delegated to Pydantic on
  :class:`~eurpe.api.schemas.GenerateSectionRequest`; the service handles
  profile lookup, retrieval, prompt assembly, LLM call, and citation
  validation.

Error mapping
-------------
* ``LLMUnavailableError`` (Ollama down, no fallback) → ``503 Service
  Unavailable`` so the React client can surface a "start Ollama / use
  --offline" message rather than treating it as an operator bug.
* ``GenerationError`` (hallucinated marker, retriever returned nothing
  workable) → ``500 Internal Server Error``. The detail string is the
  exception message so an operator gets the actual reason in the UI.
* ``SecurityError`` (network policy denied egress) → ``403 Forbidden``
  before any prompt leaves the machine.
* ``FileNotFoundError`` / ``ValueError`` (profile lookup) → ``400 Bad
  Request``. The operator chose a profile that the server cannot load.
* Pydantic validation failures bubble up as the default 422 with the
  FastAPI ``detail`` envelope the React client already understands
  (see ``frontend/src/features/ingest/api.ts``).

Privacy guards
--------------
The generation service runs every request through the workflow's
analytics logger — the logger lives under ``runtime_dir`` and never
leaves the machine unless the operator explicitly exports it. No
request body is logged outside that path.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from eurpe.api.dependencies import get_generation_service
from eurpe.api.schemas import (
    CitationPayload,
    DraftingProfileSummary,
    FetchCallRequest,
    FetchCallResponse,
    GenerateSectionRequest,
    GenerateSectionResponse,
    GenerationEnumsResponse,
    IterateSectionRequest,
    IterateSectionResponse,
    IterationRecordPayload,
    ProfilesResponse,
)
from eurpe.generation import (
    CitationRef,
    GenerationDraft,
    GenerationError,
    GenerationRequest,
    GenerationService,
    IterationRecord,
    LLMUnavailableError,
    SectionGenerationRequest,
    SectionIterationRequest,
)
from eurpe.generation.profiles import list_available_profiles, load_profile
from eurpe.intake.call_fetcher import (
    InvalidPortalURLError,
    PortalUnavailableError,
    TopicNotFoundError,
    fetch_call_context,
)
from eurpe.schema import Programme, SectionType
from eurpe.security import SecurityError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generation", tags=["generation"])


@router.get("/enums", response_model=GenerationEnumsResponse)
def get_enums() -> GenerationEnumsResponse:
    """Return the closed enum vocabularies the drafting UI Selects must use.

    Sourced from the Python enums directly so adding a new
    :class:`SectionType` or :class:`Programme` member ripples into the
    drafting workspace on the next request — no hand-typed string list
    in the React code.
    """

    return GenerationEnumsResponse(
        section_type=[member.value for member in SectionType],
        programme=[member.value for member in Programme],
    )


@router.get("/profiles", response_model=ProfilesResponse)
def get_profiles() -> ProfilesResponse:
    """Return the drafting profiles bundled with this build.

    The UI Select uses this list to offer profile choices. An empty
    list is a legal state — the workspace falls back to generic
    workflow guidance when no profile is selected.

    Profile loading is best-effort: a malformed YAML on disk surfaces
    as a logged warning and the entry is skipped, so a single bad
    profile cannot break the workspace.
    """

    summaries: list[DraftingProfileSummary] = []
    for programme in list_available_profiles():
        try:
            profile = load_profile(programme)
        except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - skip path
            logger.warning(
                "Skipping malformed drafting profile for %s: %s",
                programme.value,
                exc,
            )
            continue
        summaries.append(
            DraftingProfileSummary(programme=programme, name=profile.name),
        )
    return ProfilesResponse(profiles=summaries)


@router.post("/section", response_model=GenerateSectionResponse)
def generate_section(
    body: GenerateSectionRequest,
    service: GenerationService = Depends(get_generation_service),
) -> GenerateSectionResponse:
    """Drive the section-generation workflow for one operator request.

    Pydantic validation on :class:`GenerateSectionRequest` already
    enforces the AC #4 invariants — ``user_intent`` non-empty,
    ``top_k_examples`` in [1, 20], known enum members — so an
    invalid request fails fast with a 422 before the workflow runs.

    The route converts the wire request into a
    :class:`SectionGenerationRequest`, hands it to the service, and
    serialises the returned :class:`GenerationDraft` into the wire
    response. The transformation is a 1-to-1 field copy; the wire
    model exists so the workflow's internal field set can evolve
    without breaking the React client.
    """

    workflow_request = GenerationRequest(
        section_type=body.section_type,
        user_intent=body.user_intent,
        call_context=body.call_context,
        target_programme=body.target_programme,
        top_k_examples=body.top_k_examples,
        lessons_learned=body.lessons_learned,
    )
    service_request = SectionGenerationRequest(
        request=workflow_request,
        profile_programme=body.profile_programme,
    )

    try:
        draft = service.generate_section(service_request)
    except LLMUnavailableError as exc:
        # 503 is the right status for a transient backend dependency
        # that the client can retry once Ollama is running.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM unavailable: {exc}",
        ) from exc
    except GenerationError as exc:
        # GenerationError covers hallucinated markers and the workflow
        # rejecting its own output — treat as 500 because the workflow
        # is the contract holder, not the operator.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"generation failed: {exc}",
        ) from exc
    except SecurityError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"network policy denied generation: {exc}",
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        # Both come from :func:`load_profile` when the requested
        # ``profile_programme`` does not match a bundled YAML. The
        # operator picked a bad profile, so this is a 400.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed to load drafting profile: {exc}",
        ) from exc

    return _draft_to_response(draft)


def _draft_to_response(draft: GenerationDraft) -> GenerateSectionResponse:
    """Convert an internal :class:`GenerationDraft` into the wire response.

    Shared helper so the :func:`generate_section` and
    :func:`iterate_section` routes serialise drafts identically (and a
    future ``GET /api/generation/section/{id}`` would too). Mirrors
    the inverse of :func:`_request_to_internal` below — both keep the
    wire surface stable while the workflow's internal field set
    evolves.
    """

    citations = [
        CitationPayload(
            citation_id=c.citation_id,
            source_status=c.source_status,
            programme=c.programme,
            call_id=c.call_id,
            proposal_title=c.proposal_title,
            section_heading=c.section_heading,
            page=c.page,
            chunk_id=c.chunk_id,
            snippet=c.snippet,
        )
        for c in draft.citations
    ]
    iterations = [
        IterationRecordPayload(
            iteration_index=it.iteration_index,
            changes_summary=it.changes_summary,
            requirements_checked=list(it.requirements_checked),
            critique_text=it.critique_text,
            generated_at=it.generated_at,
        )
        for it in draft.iterations
    ]
    return GenerateSectionResponse(
        section_type=draft.section_type,
        text=draft.text,
        citations=citations,
        model=draft.model,
        generated_at=draft.generated_at,
        drafting_profile=draft.drafting_profile,
        iterations=iterations,
    )


def _response_to_draft(payload: GenerateSectionResponse) -> GenerationDraft:
    """Rehydrate a wire :class:`GenerateSectionResponse` into a :class:`GenerationDraft`.

    Used by :func:`iterate_section` to turn the client-supplied
    ``prior_draft`` back into the internal :class:`GenerationDraft`
    the critic loop expects. The rehydrated draft is the *minimum*
    needed by the loop (text, citations, section_type, iterations);
    other internal fields (``prompt_used``, ``request``, ``topic_context``)
    are reconstructed from the matching iterate request so the loop
    can compose the augmented request without storing them.
    """

    citations = [
        CitationRef(
            citation_id=c.citation_id,
            source_status=c.source_status,
            programme=c.programme,
            call_id=c.call_id,
            proposal_title=c.proposal_title,
            section_heading=c.section_heading,
            page=c.page,
            chunk_id=c.chunk_id,
            snippet=c.snippet,
        )
        for c in payload.citations
    ]
    iterations = [
        IterationRecord(
            iteration_index=it.iteration_index,
            changes_summary=it.changes_summary,
            requirements_checked=list(it.requirements_checked),
            critique_text=it.critique_text,
            generated_at=it.generated_at,
        )
        for it in payload.iterations
    ]
    # ``prompt_used`` and ``request`` are populated with stub values that
    # satisfy GenerationDraft's invariants. The critic loop only reads
    # ``text`` / ``citations`` / ``section_type`` / ``iterations`` from
    # the prior draft, so the stubs never leak into output (the refined
    # draft carries the fresh prompt from the regeneration step).
    placeholder_request = GenerationRequest(
        section_type=payload.section_type,
        user_intent="(rehydrated — original intent supplied in IterateSectionRequest)",
    )
    return GenerationDraft(
        section_type=payload.section_type,
        text=payload.text,
        citations=citations,
        prompt_used="(prior prompt — not echoed by the wire response)",
        model=payload.model,
        generated_at=payload.generated_at,
        request=placeholder_request,
        drafting_profile=payload.drafting_profile,
        iterations=iterations,
    )


@router.post("/section/iterate", response_model=IterateSectionResponse)
def iterate_section(
    body: IterateSectionRequest,
    service: GenerationService = Depends(get_generation_service),
) -> IterateSectionResponse:
    """Run one critic+regenerate iteration on top of the supplied prior draft.

    AC #1 of issue #16 ("user can set critic iterations between 1 and
    5") is enforced at the Pydantic boundary on
    :attr:`IterateSectionRequest.max_iterations`. AC #2 ("user can stop
    the loop after any completed iteration") is satisfied by the
    per-iteration shape of this endpoint — the client simply stops
    calling. AC #3 ("each iteration records what changed and which
    call/profile requirements were checked") is delivered by the
    :class:`IterationRecordPayload` appended to the returned draft's
    ``iterations`` list.

    Error mapping mirrors :func:`generate_section`.
    """

    workflow_request = GenerationRequest(
        section_type=body.section_type,
        user_intent=body.user_intent,
        call_context=body.call_context,
        target_programme=body.target_programme,
        top_k_examples=body.top_k_examples,
        lessons_learned=body.lessons_learned,
    )
    prior_draft = _response_to_draft(body.prior_draft)
    iteration_request = SectionIterationRequest(
        request=workflow_request,
        profile_programme=body.profile_programme,
        prior_draft=prior_draft,
        max_iterations=body.max_iterations,
    )

    try:
        result = service.iterate_section(iteration_request)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM unavailable: {exc}",
        ) from exc
    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"iteration failed: {exc}",
        ) from exc
    except SecurityError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"network policy denied generation: {exc}",
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed to load drafting profile: {exc}",
        ) from exc

    return IterateSectionResponse(
        draft=_draft_to_response(result.draft),
        iteration_index=result.iteration_index,
        max_iterations=result.max_iterations,
        stopped=result.stopped,
    )


@router.post("/fetch-call", response_model=FetchCallResponse)
def fetch_call(body: FetchCallRequest) -> FetchCallResponse:
    """Auto-fill structured call context from a Funding & Tenders Portal URL (issue #67).

    The route is intentionally side-effect-free: it makes one outbound
    request to the public SEDIA search API (``api.tech.ec.europa.eu``)
    and returns the call_id / topic_id / topic_title triple. It does
    not persist anything, does not touch the corpus index, and does
    not invoke the LLM.

    This is the one route in EURPE that intentionally calls out to the
    internet — see ``eurpe.intake.call_fetcher`` module docstring for
    the network-policy rationale.

    Error mapping
    -------------
    * :class:`InvalidPortalURLError` → 422 (operator pasted a non-portal
      URL or a malformed link — they can fix this).
    * :class:`TopicNotFoundError` → 404 (URL parsed cleanly but SEDIA
      had no entry; usually a very recent topic).
    * :class:`PortalUnavailableError` → 502 (network failure or SEDIA
      schema break; the operator did nothing wrong).
    """

    try:
        result = fetch_call_context(body.url)
    except InvalidPortalURLError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except TopicNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PortalUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return FetchCallResponse(
        call_id=result.call_id,
        topic_id=result.topic_id,
        topic_title=result.topic_title,
        expected_outcomes=result.expected_outcomes,
        scope=result.scope,
        call_title=result.call_title,
        source_url=result.source_url,
    )
