"""TestClient coverage for the FastAPI generation routes (issue #15).

The tests live in the fast tier — they avoid Ollama by:

* pointing :func:`eurpe.api.dependencies.set_config_path` at a tmp-dir
  ``config.yaml`` written by ``write_offline_config`` (the Ollama URL
  there is unreachable so ``make_llm_client`` falls back to the
  deterministic stub), and
* monkeypatching :func:`eurpe.retrieval.embeddings._ollama_reachable` to
  False so ``make_embedder`` selects the deterministic-hash embedder
  without firing a real TCP attempt.

Each test wraps its work in ``_reset_state`` so cache state from earlier
tests does not bleed across (the dependency cache caches the open Chroma
client by config-path; without a reset two consecutive tests would race
on a stale file lock).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eurpe.api import dependencies as deps
from eurpe.api.main import app
from eurpe.generation import (
    GenerationDraft,
    GenerationError,
    IterationResult,
    LLMUnavailableError,
    SectionGenerationRequest,
    SectionIterationRequest,
)
from eurpe.generation.models import CitationRef, IterationRecord
from eurpe.schema import Programme, SectionType, SourceStatus
from tests._helpers.offline import write_offline_config


@pytest.fixture
def configured_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Yield a TestClient wired to a tmp-dir offline config + stub LLM.

    ``make_embedder`` probes Ollama with ``socket.create_connection`` to
    decide whether to fall back to the deterministic embedder. We force
    the unreachable path here so the deterministic embedder is selected
    without firing a real TCP attempt.

    The default ``get_generation_service`` provider is left in place;
    tests that need to inject a custom service body do so via
    ``app.dependency_overrides`` in the test itself.
    """

    monkeypatch.setattr(
        "eurpe.retrieval.embeddings._ollama_reachable",
        lambda *_args, **_kwargs: False,
    )
    cfg_path = write_offline_config(tmp_path)
    deps.set_config_path(cfg_path)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(deps.get_generation_service, None)
        deps.reset_dependency_caches()


class _StubService:
    """Minimal stand-in for :class:`GenerationService`.

    Used by the error-path tests so we can deterministically raise
    each error type the route is expected to translate to an HTTP
    status — without depending on the retriever / LLM happening to
    fail in the right way.
    """

    def __init__(self, behaviour: str) -> None:
        self._behaviour = behaviour

    def generate_section(self, request: SectionGenerationRequest) -> GenerationDraft:
        if self._behaviour == "llm_unavailable":
            raise LLMUnavailableError("Ollama daemon offline")
        if self._behaviour == "generation_error":
            raise GenerationError("the LLM cited an unknown source")
        if self._behaviour == "missing_profile":
            raise FileNotFoundError("profile not bundled")
        if self._behaviour == "happy":
            citation = CitationRef(
                citation_id=1,
                source_status=SourceStatus.FUNDED,
                programme=Programme.HORIZON_EUROPE,
                call_id="HORIZON-CL5-2024-D3-02",
                proposal_title="Stubbed Source Proposal",
                section_heading="1.2 Methodology",
                page=8,
                chunk_id="stub-chunk-1",
                snippet="Methodology snippet [1] from a funded proposal.",
            )
            return GenerationDraft(
                section_type=request.request.section_type,
                text="Stubbed draft body with marker [1].",
                citations=[citation],
                prompt_used="(stub prompt)",
                model="deterministic-stub-v1",
                request=request.request,
                drafting_profile=None,
            )
        raise AssertionError(f"unknown behaviour: {self._behaviour}")

    def iterate_section(self, request: SectionIterationRequest) -> IterationResult:
        """Mirror :meth:`generate_section` for the /iterate route tests.

        Returns a deterministic :class:`IterationResult` with one critic
        record appended so the route can be exercised without the real
        critic LLM round-trip.
        """

        if self._behaviour == "llm_unavailable":
            raise LLMUnavailableError("Ollama daemon offline")
        if self._behaviour == "generation_error":
            raise GenerationError("the critic produced an invalid response")
        if self._behaviour == "missing_profile":
            raise FileNotFoundError("profile not bundled")
        if self._behaviour == "happy":
            new_index = request.prior_draft.total_iterations() + 1
            record = IterationRecord(
                iteration_index=new_index,
                changes_summary=(
                    f"Iteration {new_index} expanded the draft. "
                    "text +50 chars, citations +0."
                ),
                requirements_checked=[
                    "default-section-guidance",
                    "validation strategy",
                ],
                critique_text=(
                    f"Critic iteration {new_index}: please add a validation "
                    "strategy and clarify the data source."
                ),
            )
            iterations = list(request.prior_draft.iterations) + [record]
            refined = request.prior_draft.model_copy(
                update={
                    "text": request.prior_draft.text + "\n\n_Refined._",
                    "iterations": iterations,
                }
            )
            return IterationResult(
                draft=refined,
                iteration_index=new_index,
                max_iterations=request.max_iterations,
                stopped=(new_index >= request.max_iterations),
            )
        raise AssertionError(f"unknown behaviour: {self._behaviour}")


def _override_with(client: TestClient, behaviour: str) -> None:
    """Swap the generation service provider for a :class:`_StubService`.

    Pulled out as a helper so each test reads with one line of
    arrangement and the cleanup is centralised in the fixture's
    ``finally`` block.
    """

    del client  # only the app's dependency_overrides matters
    app.dependency_overrides[deps.get_generation_service] = lambda: _StubService(behaviour)


def test_generation_enums_endpoint_returns_python_enum_values(
    configured_app: TestClient,
) -> None:
    """``GET /api/generation/enums`` mirrors the Python enums exactly.

    Sourced from :class:`SectionType` and :class:`Programme` directly so
    adding a new member upstream automatically surfaces in the UI
    without a code-gen step.
    """

    response = configured_app.get("/api/generation/enums")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["section_type"] == [member.value for member in SectionType]
    assert body["programme"] == [member.value for member in Programme]


def test_profiles_endpoint_lists_bundled_profiles(configured_app: TestClient) -> None:
    """``GET /api/generation/profiles`` returns the YAML-bundled profiles."""

    response = configured_app.get("/api/generation/profiles")
    assert response.status_code == 200, response.text
    body = response.json()
    profiles = body["profiles"]
    assert isinstance(profiles, list)
    # Two profiles ship with the repo: horizon_europe + digital_europe.
    # Assert non-empty and that the expected shape is present.
    assert len(profiles) >= 1
    for entry in profiles:
        assert set(entry.keys()) == {"programme", "name"}
        assert entry["programme"] in {member.value for member in Programme}
        assert isinstance(entry["name"], str) and entry["name"]


def test_generate_section_happy_path_returns_draft_and_citations(
    configured_app: TestClient,
) -> None:
    """A well-formed request comes back as a populated draft envelope."""

    _override_with(configured_app, "happy")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe a deep learning methodology",
        "call_context": "",
        "target_programme": Programme.HORIZON_EUROPE.value,
        "top_k_examples": 3,
        "lessons_learned": False,
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["section_type"] == SectionType.METHODOLOGY.value
    assert body["text"]
    assert body["model"] == "deterministic-stub-v1"
    assert len(body["citations"]) == 1
    cite = body["citations"][0]
    assert cite["citation_id"] == 1
    assert cite["source_status"] == SourceStatus.FUNDED.value
    assert cite["programme"] == Programme.HORIZON_EUROPE.value
    assert cite["call_id"] == "HORIZON-CL5-2024-D3-02"


def test_generate_section_rejects_empty_intent(configured_app: TestClient) -> None:
    """Pydantic 422 — ``user_intent`` cannot be blank (AC #4 server-side).

    The React client validates first, but the route's Pydantic model
    must reject a bypassed client too. ``extra='forbid'`` plus
    ``min_length=1`` on ``user_intent`` produces a clean 422.
    """

    _override_with(configured_app, "happy")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "",
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 422, response.text
    assert "user_intent" in response.text


def test_generate_section_rejects_unknown_section_type(
    configured_app: TestClient,
) -> None:
    """Unknown ``section_type`` enum value fails fast with 422."""

    _override_with(configured_app, "happy")
    payload = {
        "section_type": "not-a-section",
        "user_intent": "describe something",
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 422, response.text


def test_generate_section_rejects_top_k_out_of_range(configured_app: TestClient) -> None:
    """``top_k_examples`` must respect the 1..20 bounds (AC #4)."""

    _override_with(configured_app, "happy")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe something",
        "top_k_examples": 50,
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 422, response.text


def test_generate_section_rejects_extra_fields(configured_app: TestClient) -> None:
    """``extra='forbid'`` — a typo in a field name fails the request loudly."""

    _override_with(configured_app, "happy")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe something",
        "totally_made_up_field": "oops",
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 422, response.text


def test_generate_section_maps_llm_unavailable_to_503(
    configured_app: TestClient,
) -> None:
    """``LLMUnavailableError`` propagates as 503 Service Unavailable."""

    _override_with(configured_app, "llm_unavailable")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe something",
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 503, response.text
    assert "LLM unavailable" in response.text


def test_generate_section_maps_generation_error_to_500(
    configured_app: TestClient,
) -> None:
    """``GenerationError`` propagates as 500 with the workflow message."""

    _override_with(configured_app, "generation_error")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe something",
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 500, response.text
    assert "generation failed" in response.text


def test_generate_section_maps_missing_profile_to_400(
    configured_app: TestClient,
) -> None:
    """A missing drafting profile is the operator's input error → 400."""

    _override_with(configured_app, "missing_profile")
    payload = {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe something",
        "profile_programme": Programme.HORIZON_EUROPE.value,
    }
    response = configured_app.post("/api/generation/section", json=payload)
    assert response.status_code == 400, response.text
    assert "drafting profile" in response.text


def test_get_generation_service_provider_returns_cached_singleton(
    configured_app: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dependency provider memoises the service per (config_path, collection).

    Wraps :func:`deps.get_generation_service` so we touch the real
    provider (without the stub override) and assert two calls return
    the same instance. Asserts the caching contract that keeps
    Chroma's file lock from being re-acquired on every request.
    """

    del configured_app  # using the fixture's offline config side-effect

    cfg = deps.get_config()
    first = deps.get_generation_service(cfg)
    second = deps.get_generation_service(cfg)
    assert first is second


# ---------------------------------------------------------------------------
# POST /api/generation/section/iterate — Task 3.2 / issue #16
# ---------------------------------------------------------------------------


def _initial_draft_payload() -> dict:
    """Construct a wire-format draft envelope to pass as ``prior_draft``."""

    return {
        "section_type": SectionType.METHODOLOGY.value,
        "text": "Initial draft with marker [1].",
        "citations": [
            {
                "citation_id": 1,
                "source_status": SourceStatus.FUNDED.value,
                "programme": Programme.HORIZON_EUROPE.value,
                "call_id": "HORIZON-CL5-2024-D3-02",
                "proposal_title": "Stubbed Source Proposal",
                "section_heading": "1.2 Methodology",
                "page": 8,
                "chunk_id": "stub-chunk-1",
                "snippet": "Methodology snippet [1] from a funded proposal.",
            }
        ],
        "model": "deterministic-stub-v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "drafting_profile": None,
        "iterations": [],
    }


def _iterate_payload(*, max_iterations: int = 3) -> dict:
    """Construct an IterateSectionRequest payload."""

    return {
        "section_type": SectionType.METHODOLOGY.value,
        "user_intent": "describe a deep learning methodology",
        "call_context": "",
        "target_programme": Programme.HORIZON_EUROPE.value,
        "profile_programme": None,
        "top_k_examples": 3,
        "lessons_learned": False,
        "max_iterations": max_iterations,
        "prior_draft": _initial_draft_payload(),
    }


class TestIterateSectionRoute:
    def test_happy_path_returns_refined_draft_with_iteration_record(
        self, configured_app: TestClient
    ) -> None:
        """A well-formed iterate request returns one refined draft + record."""

        _override_with(configured_app, "happy")
        response = configured_app.post(
            "/api/generation/section/iterate",
            json=_iterate_payload(max_iterations=3),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["iteration_index"] == 2
        assert body["max_iterations"] == 3
        assert body["stopped"] is False  # 2 < 3
        draft = body["draft"]
        assert len(draft["iterations"]) == 1
        record = draft["iterations"][0]
        # AC #3 — both halves present and non-empty.
        assert record["iteration_index"] == 2
        assert record["changes_summary"]
        assert record["requirements_checked"]
        assert record["critique_text"]

    def test_stop_flag_set_when_iteration_index_equals_max(
        self, configured_app: TestClient
    ) -> None:
        """``stopped=True`` when this iteration is the last permitted."""

        _override_with(configured_app, "happy")
        # max_iterations=2 means iteration 2 is the last; the stub
        # returns stopped=True.
        response = configured_app.post(
            "/api/generation/section/iterate",
            json=_iterate_payload(max_iterations=2),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["iteration_index"] == 2
        assert body["stopped"] is True

    @pytest.mark.parametrize("invalid", [0, 1, 6, 100])
    def test_max_iterations_outside_two_to_five_returns_422(
        self, configured_app: TestClient, invalid: int
    ) -> None:
        """AC #1 boundary — Pydantic ge=2, le=5 on /iterate surfaces a 422.

        ``max_iterations=1`` is rejected at the wire because the iterate
        endpoint, by definition, runs *past* the first pass: a cap of 1
        leaves no room for the critic. The workspace-facing AC #1 ceiling
        of 1-5 still holds — the ``1`` choice maps to "no critic at all"
        and uses the single-pass /section endpoint, not /iterate.
        """

        _override_with(configured_app, "happy")
        payload = _iterate_payload()
        payload["max_iterations"] = invalid
        response = configured_app.post(
            "/api/generation/section/iterate", json=payload
        )
        assert response.status_code == 422, response.text

    def test_extra_field_forbidden(self, configured_app: TestClient) -> None:
        """``extra='forbid'`` keeps typos loud."""

        _override_with(configured_app, "happy")
        payload = _iterate_payload()
        payload["typo_field"] = "oops"
        response = configured_app.post(
            "/api/generation/section/iterate", json=payload
        )
        assert response.status_code == 422, response.text

    def test_missing_prior_draft_returns_422(self, configured_app: TestClient) -> None:
        """``prior_draft`` is required — bypassed clients get a clean 422."""

        _override_with(configured_app, "happy")
        payload = _iterate_payload()
        del payload["prior_draft"]
        response = configured_app.post(
            "/api/generation/section/iterate", json=payload
        )
        assert response.status_code == 422, response.text

    def test_llm_unavailable_maps_to_503(self, configured_app: TestClient) -> None:
        _override_with(configured_app, "llm_unavailable")
        response = configured_app.post(
            "/api/generation/section/iterate", json=_iterate_payload()
        )
        assert response.status_code == 503, response.text

    def test_generation_error_maps_to_500(self, configured_app: TestClient) -> None:
        _override_with(configured_app, "generation_error")
        response = configured_app.post(
            "/api/generation/section/iterate", json=_iterate_payload()
        )
        assert response.status_code == 500, response.text
        assert "iteration failed" in response.text

    def test_missing_profile_maps_to_400(self, configured_app: TestClient) -> None:
        _override_with(configured_app, "missing_profile")
        payload = _iterate_payload()
        payload["profile_programme"] = Programme.HORIZON_EUROPE.value
        response = configured_app.post(
            "/api/generation/section/iterate", json=payload
        )
        assert response.status_code == 400, response.text
