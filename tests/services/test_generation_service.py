"""Tests for :class:`eurpe.generation.GenerationService`.

Uses the existing ``deterministic_workflow`` fixture from conftest to
exercise the service end-to-end without touching Ollama. Covers AC #3:
one happy path + one error path.
"""

from __future__ import annotations

import pytest

from eurpe.generation import (
    GenerationRequest,
    GenerationService,
    LLMUnavailableError,
    SectionGenerationRequest,
    SectionGenerationWorkflow,
)
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    RetrievalPolicy,
    SourceStatusAwareRetriever,
)
from eurpe.schema import Programme, SectionType


def test_generation_service_runs_workflow_happy_path(deterministic_workflow) -> None:
    """Service returns a populated draft when the workflow succeeds."""

    service = GenerationService(workflow=deterministic_workflow)
    request = SectionGenerationRequest(
        request=GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="describe a deep learning methodology",
            target_programme=Programme.HORIZON_EUROPE,
            top_k_examples=3,
        ),
    )
    draft = service.generate_section(request)

    assert draft.section_type is SectionType.METHODOLOGY
    assert draft.text
    assert draft.model == "deterministic-stub-v1"
    assert draft.drafting_profile is None


def test_generation_service_resolves_profile_programme(deterministic_workflow) -> None:
    """Passing ``profile_programme`` causes the workflow to apply the profile.

    Happy-path coverage of the service's only non-trivial logic (the
    ``load_profile`` lookup). Uses a programme that ships with a bundled
    profile — see :mod:`eurpe.generation.profiles`.
    """

    from eurpe.generation.profiles import list_available_profiles, load_profile

    available = list_available_profiles()
    if not available:
        pytest.skip("no drafting profiles bundled with this build")

    programme = available[0]
    expected_profile_name = load_profile(programme).name

    service = GenerationService(workflow=deterministic_workflow)
    request = SectionGenerationRequest(
        request=GenerationRequest(
            section_type=SectionType.METHODOLOGY,
            user_intent="describe a deep learning methodology",
            target_programme=Programme.HORIZON_EUROPE,
        ),
        profile_programme=programme,
    )
    draft = service.generate_section(request)
    assert draft.drafting_profile == expected_profile_name


def test_generation_service_propagates_llm_unavailable(tmp_path) -> None:
    """An LLM that raises propagates :class:`LLMUnavailableError` unchanged.

    The service is a thin wrapper — it must NOT swallow LLM errors,
    because the CLI / HTTP layer is responsible for mapping them to the
    right exit code / status code.
    """

    class _BrokenLLM:
        model = "broken-stub"

        def generate(self, prompt: str) -> str:  # noqa: ARG002 - unused on purpose
            raise LLMUnavailableError("daemon offline")

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="generation_service_error_path",
    )
    retriever = SourceStatusAwareRetriever(index, policy=RetrievalPolicy(relevance_threshold=0.0))
    workflow = SectionGenerationWorkflow(retriever=retriever, llm=_BrokenLLM())  # type: ignore[arg-type]
    service = GenerationService(workflow=workflow)

    with pytest.raises(LLMUnavailableError):
        service.generate_section(
            SectionGenerationRequest(
                request=GenerationRequest(
                    section_type=SectionType.METHODOLOGY,
                    user_intent="anything",
                )
            )
        )


def test_generation_service_missing_profile_raises(deterministic_workflow, monkeypatch) -> None:
    """A programme with no bundled profile surfaces a load error to the caller.

    Forces the error by redirecting the service-module's
    :func:`load_profile` to a stub that always raises — the bundled
    HORIZON_EUROPE profile exists on disk, so we can't trigger the
    error without redirecting the lookup.
    """

    import eurpe.generation.service as service_module

    def _empty_load_profile(programme):  # noqa: ARG001 - signature mirrors real loader
        raise FileNotFoundError(f"no profile for {programme}")

    monkeypatch.setattr(service_module, "load_profile", _empty_load_profile)
    service = GenerationService(workflow=deterministic_workflow)
    with pytest.raises(FileNotFoundError):
        service.generate_section(
            SectionGenerationRequest(
                request=GenerationRequest(
                    section_type=SectionType.METHODOLOGY,
                    user_intent="anything",
                ),
                profile_programme=Programme.HORIZON_EUROPE,
            )
        )


def test_generation_service_workflow_property_returns_injected_workflow(
    deterministic_workflow,
) -> None:
    """The ``workflow`` property exposes the injected workflow unchanged."""

    service = GenerationService(workflow=deterministic_workflow)
    assert service.workflow is deterministic_workflow
