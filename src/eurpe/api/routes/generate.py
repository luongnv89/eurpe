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
    GenerateSectionRequest,
    GenerateSectionResponse,
    GenerationEnumsResponse,
    ProfilesResponse,
)
from eurpe.generation import (
    GenerationError,
    GenerationRequest,
    GenerationService,
    LLMUnavailableError,
    SectionGenerationRequest,
)
from eurpe.generation.profiles import list_available_profiles, load_profile
from eurpe.schema import Programme, SectionType

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
    except (FileNotFoundError, ValueError) as exc:
        # Both come from :func:`load_profile` when the requested
        # ``profile_programme`` does not match a bundled YAML. The
        # operator picked a bad profile, so this is a 400.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed to load drafting profile: {exc}",
        ) from exc

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
    return GenerateSectionResponse(
        section_type=draft.section_type,
        text=draft.text,
        citations=citations,
        model=draft.model,
        generated_at=draft.generated_at,
        drafting_profile=draft.drafting_profile,
    )
