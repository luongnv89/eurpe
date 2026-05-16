"""Service facade for section-draft generation.

The generation service is the named entry point Task 3.1 references:

    "Generate action calls the generation service and displays draft
    output with citations." — tasks.md, Task 3.1 acceptance criteria.

It wraps :class:`SectionGenerationWorkflow` so the React UI, the CLI,
and any future REST endpoint hand in a single
:class:`SectionGenerationRequest` and get back a
:class:`GenerationDraft`. The workflow is already well-factored;
this service exists to give the UI one stable seam rather than asking
it to import the workflow class directly.

Profile loading
---------------
The profile lookup happens here on purpose: the UI Selects send a
:class:`Programme` enum value over the wire, but the workflow expects
a :class:`DraftingProfile` instance. The service is where that
mapping lives so neither the route handler nor the React component
has to import :func:`load_profile`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from eurpe.generation.models import GenerationDraft, GenerationRequest
from eurpe.generation.profiles import DraftingProfile, load_profile
from eurpe.generation.workflow import SectionGenerationWorkflow
from eurpe.schema import Programme


class SectionGenerationRequest(BaseModel):
    """Input to :meth:`GenerationService.generate_section`.

    Wraps the existing :class:`GenerationRequest` so the service can
    add request-level toggles (``profile_programme``, future
    ``stream: bool``, etc.) without changing the
    generation-workflow signature.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: GenerationRequest = Field(
        description=(
            "The structured generation request — section_type, intent, "
            "programme, topic context, top_k_examples, lessons_learned."
        )
    )
    profile_programme: Programme | None = Field(
        default=None,
        description=(
            "Programme whose drafting profile should be applied. The "
            "service resolves this via :func:`load_profile`; pass ``None`` "
            "to use the workflow's generic guidance."
        ),
    )


class GenerationService:
    """Wrap :class:`SectionGenerationWorkflow` behind a single seam.

    Stateless aside from the injected workflow. The workflow itself
    holds the retriever + LLM + analytics; the service only adds the
    profile-programme → :class:`DraftingProfile` lookup.
    """

    def __init__(self, workflow: SectionGenerationWorkflow) -> None:
        self._workflow = workflow

    @property
    def workflow(self) -> SectionGenerationWorkflow:
        """Read-only access — useful in tests."""

        return self._workflow

    def generate_section(self, request: SectionGenerationRequest) -> GenerationDraft:
        """Drive the section-generation workflow for one request.

        Errors:

        * :class:`eurpe.generation.errors.GenerationError` and
          :class:`eurpe.generation.errors.LLMUnavailableError` —
          propagated from the workflow. The service does not swallow
          them: the caller (CLI / HTTP) is responsible for mapping
          them to the right exit code / status code.
        * :class:`FileNotFoundError` / :class:`ValueError` — propagated
          from :func:`load_profile` when the requested profile does
          not exist on disk. Callers should surface this as a 400 /
          exit 1 (a missing profile is the operator's input error,
          not a server fault).
        """

        profile: DraftingProfile | None = None
        if request.profile_programme is not None:
            profile = load_profile(request.profile_programme)
        return self._workflow.run(request.request, profile=profile)
