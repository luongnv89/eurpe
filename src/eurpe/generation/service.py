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

from eurpe.generation.critic import CriticAgent
from eurpe.generation.critic_loop import (
    DEFAULT_MAX_ITERATIONS,
    MAX_ITERATIONS_CEILING,
    CriticLoopWorkflow,
    IterationResult,
)
from eurpe.generation.llm import LLMClient
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


class SectionIterationRequest(BaseModel):
    """Input to :meth:`GenerationService.iterate_section` (Task 3.2 / issue #16).

    Drives one critic+regenerate pass on top of a previously produced
    draft. The caller (CLI / HTTP / UI) owns the loop: each call
    advances the iteration counter by one and returns the refined
    draft plus the cumulative iteration history.

    Backward compatible with :class:`SectionGenerationRequest`: the
    same ``request`` + ``profile_programme`` fields drive the
    regeneration, with the prior draft + cap added.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: GenerationRequest = Field(
        description=(
            "The structured generation request used by the original draft. "
            "Programme filter, topic context, top-k, and lessons-learned "
            "carry through unchanged to every iteration; only ``user_intent`` "
            "is augmented with the critique."
        )
    )
    profile_programme: Programme | None = Field(
        default=None,
        description=(
            "Programme whose drafting profile should be applied. Must match "
            "the profile used for the original draft so the requirements "
            "list stays consistent across iterations."
        ),
    )
    prior_draft: GenerationDraft = Field(
        description=(
            "The most recent draft. The critic reads this draft, "
            "produces a critique, and the workflow regenerates against "
            "the augmented intent. Carries the cumulative iteration "
            "history forward."
        ),
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=MAX_ITERATIONS_CEILING,
        description=(
            "User-configured ceiling on iteration count. Bounded to "
            "[1, 5] to match AC #1 of issue #16. The service returns "
            "``IterationResult.stopped=True`` once the cap is reached; "
            "the caller (UI) uses that to disable the ‘Refine’ button."
        ),
    )


class GenerationService:
    """Wrap :class:`SectionGenerationWorkflow` behind a single seam.

    Stateless aside from the injected workflow. The workflow itself
    holds the retriever + LLM + analytics; the service only adds the
    profile-programme → :class:`DraftingProfile` lookup.

    Optional critic agent
    ---------------------
    When constructed with ``critic_llm`` (an :class:`LLMClient`), the
    service exposes :meth:`iterate_section` to drive the Task 3.2
    critic loop. Defaults to the same LLM client the workflow uses
    when ``critic_llm`` is ``None`` so the service stays useful out of
    the box. Pass an explicit (smaller / cheaper) client to wire a
    dedicated critic backend.
    """

    def __init__(
        self,
        workflow: SectionGenerationWorkflow,
        *,
        critic_llm: LLMClient | None = None,
    ) -> None:
        self._workflow = workflow
        # Default the critic LLM to the workflow's LLM so a single-
        # argument construction still gives a fully functional critic
        # loop. Callers that want a dedicated critic LLM pass one in.
        self._critic = CriticAgent(critic_llm or workflow.llm)
        self._critic_loop = CriticLoopWorkflow(
            workflow=workflow,
            critic=self._critic,
        )

    @property
    def workflow(self) -> SectionGenerationWorkflow:
        """Read-only access — useful in tests."""

        return self._workflow

    @property
    def critic_loop(self) -> CriticLoopWorkflow:
        """Read-only access to the critic loop — useful in tests."""

        return self._critic_loop

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

    def iterate_section(self, request: SectionIterationRequest) -> IterationResult:
        """Drive one critic+regenerate iteration on top of ``request.prior_draft``.

        AC #1 of issue #16 ("user can set critic iterations between 1
        and 5 before generation") is enforced at the Pydantic boundary
        on :class:`SectionIterationRequest.max_iterations`. AC #2
        ("user can stop the loop after any completed iteration") is
        satisfied by the per-iteration shape — the caller simply does
        not call again. AC #3 ("each iteration records what changed
        and which call/profile requirements were checked") is
        delivered by the :class:`IterationRecord` appended to the
        returned draft's ``iterations`` list.

        Errors mirror :meth:`generate_section`.
        """

        profile: DraftingProfile | None = None
        if request.profile_programme is not None:
            profile = load_profile(request.profile_programme)
        return self._critic_loop.iterate(
            prior_draft=request.prior_draft,
            request=request.request,
            max_iterations=request.max_iterations,
            profile=profile,
        )
