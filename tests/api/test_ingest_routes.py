"""TestClient coverage for the FastAPI ingestion routes.

The tests live in the fast tier — they avoid Docling and ChromaDB
HuggingFace fallbacks by:

* overriding the ``get_parser`` dependency with a hand-built stub that
  returns a small :class:`ParsedProposal`, and
* pointing :func:`eurpe.api.dependencies.set_config_path` at a tmp-dir
  ``config.yaml`` written by ``write_offline_config`` (the same helper
  the CLI tests use). That config picks the deterministic-hash embedder
  fallback because its Ollama URL is unreachable.

Each test wraps its work in ``_reset_state`` so cache state from earlier
tests does not bleed across (the dependency cache caches the open Chroma
client by config-path; without a reset two consecutive tests would race
on a stale file lock).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from eurpe.api import dependencies as deps
from eurpe.api.main import app
from eurpe.api.routes.ingest import router as ingest_router  # noqa: F401 - ensure import
from eurpe.ingestion.models import ParsedProposal, ParsedSection
from eurpe.schema import Programme, SourceStatus
from tests._helpers.offline import write_offline_config


class _StubParser:
    """Pretend to be :class:`DoclingProposalParser` for fast-tier tests.

    Returns a small :class:`ParsedProposal` so the route exercises the
    full real chunker + index path without dragging Docling (and the
    ~40 MB OCR weights it tries to download by default in non-offline
    mode) into the test suite.
    """

    def __init__(self, *, title: str = "Synthetic Test Proposal") -> None:
        self._title = title

    def parse(self, path: Path) -> ParsedProposal:
        return ParsedProposal(
            source_path=str(path.resolve()),
            title=self._title,
            sections=[
                ParsedSection(
                    heading="1. Excellence",
                    level=1,
                    text=(
                        "Excellence body text describing the proposal scientific ambition. "
                        "It includes a methodology paragraph that the chunker can split. "
                        "Long enough to exercise the chunker's overlap logic without depending "
                        "on Docling."
                    ),
                    page_start=1,
                    page_end=1,
                ),
                ParsedSection(
                    heading="2. Impact",
                    level=1,
                    text=(
                        "Impact narrative for downstream commercialisation. "
                        "Adds a second section so the chunker emits more than one chunk."
                    ),
                    page_start=2,
                    page_end=2,
                ),
            ],
            page_count=2,
            parser="stub",
        )


@pytest.fixture
def configured_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Yield a TestClient wired to a tmp-dir offline config + stub parser.

    ``make_embedder`` probes Ollama with ``socket.create_connection`` to
    decide whether to fall back to the deterministic embedder. We force
    the unreachable path here so the deterministic embedder is selected
    without firing a real TCP attempt (which the ``no_network`` fixture,
    when present, would treat as a hard failure).
    """

    monkeypatch.setattr(
        "eurpe.retrieval.embeddings._ollama_reachable",
        lambda *_args, **_kwargs: False,
    )
    cfg_path = write_offline_config(tmp_path)
    deps.set_config_path(cfg_path)
    app.dependency_overrides[deps.get_parser] = lambda: _StubParser()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(deps.get_parser, None)
        deps.reset_dependency_caches()


def _make_pdf_bytes() -> bytes:
    """Return ~64 bytes of placeholder data that satisfies the upload route.

    The route's PDF validation is filename-only (``.pdf`` suffix); the
    stub parser ignores the file contents entirely, so any well-formed
    bytes are fine.
    """

    return b"%PDF-1.4\n%fake-pdf-for-tests\n" * 4


def test_enums_endpoint_returns_python_enum_values(
    configured_app: TestClient, no_network: None
) -> None:
    """``GET /api/ingestion/enums`` mirrors the Python enums exactly.

    Sourced from :class:`Programme` and :class:`SourceStatus` directly so
    adding a new member upstream surfaces in the UI without a code-gen
    step. ``no_network`` is included to assert the route does not phone home.
    """

    del no_network  # fixture side-effect only.
    response = configured_app.get("/api/ingestion/enums")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["programme"] == [member.value for member in Programme]
    assert body["source_status"] == [member.value for member in SourceStatus]


def test_parse_endpoint_returns_token_and_suggested_metadata(
    configured_app: TestClient,
) -> None:
    """A canonical EU filename produces programme + call_id suggestions."""

    file_bytes = _make_pdf_bytes()
    files = {"file": ("HORIZON-CL3-2024-CS-01-883588.pdf", file_bytes, "application/pdf")}
    response = configured_app.post("/api/ingestion/parse", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["parse_token"]
    assert body["source_path"].endswith(".pdf")
    assert body["suggested"]["programme"] == Programme.HORIZON_EUROPE.value
    assert "HORIZON-CL3-2024" in body["suggested"]["call_id"]
    assert body["suggested"]["topic_id"] == "883588"
    # parsed_at should be a valid ISO datetime.
    parsed_at = datetime.fromisoformat(body["parsed_at"])
    assert parsed_at.tzinfo is not None
    assert parsed_at <= datetime.now(UTC)


def test_parse_rejects_path_traversal_filenames(configured_app: TestClient) -> None:
    """Filenames containing ``..`` or path separators must be rejected."""

    file_bytes = _make_pdf_bytes()
    for bad_name in ("../../etc/passwd.pdf", "../escape.pdf", "sub/dir.pdf"):
        files = {"file": (bad_name, file_bytes, "application/pdf")}
        response = configured_app.post("/api/ingestion/parse", files=files)
        assert response.status_code == 400, (bad_name, response.text)
        assert "unsafe filename" in response.text or "missing" in response.text


def test_parse_rejects_non_pdf_extension(configured_app: TestClient) -> None:
    """Only ``.pdf`` uploads are accepted at the parse boundary."""

    files = {"file": ("not-a-pdf.txt", b"hello", "text/plain")}
    response = configured_app.post("/api/ingestion/parse", files=files)
    assert response.status_code == 400, response.text
    assert "pdf" in response.text.lower()


def test_confirm_endpoint_persists_sidecar_and_indexes_chunks(
    configured_app: TestClient, no_network: None
) -> None:
    """Round-trip parse → confirm; sidecar lands on disk and chunks are indexed."""

    del no_network
    file_bytes = _make_pdf_bytes()
    files = {"file": ("HORIZON-CL5-2024-D3-02-883588.pdf", file_bytes, "application/pdf")}
    parse_response = configured_app.post("/api/ingestion/parse", files=files)
    assert parse_response.status_code == 200, parse_response.text
    parse_token = parse_response.json()["parse_token"]

    confirm_body = {
        "parse_token": parse_token,
        "programme": Programme.HORIZON_EUROPE.value,
        "call_id": "HORIZON-CL5-2024-D3-02",
        "topic_id": "HORIZON-CL5-2024-D3-02-01",
        "year": 2024,
        "outcome": SourceStatus.FUNDED.value,
        "proposal_title": "Synthetic Test Proposal",
        "consortium_acronym": "STP",
    }
    confirm_response = configured_app.post("/api/ingestion/confirm", json=confirm_body)
    assert confirm_response.status_code == 200, confirm_response.text
    body = confirm_response.json()
    assert body["chunks_added"] >= 1
    assert body["collection"] == "default"

    sidecar_path = Path(body["sidecar_path"])
    assert sidecar_path.exists(), sidecar_path
    sidecar = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    # The persisted sidecar carries every operator-confirmed field plus
    # the server-stamped source_path. The drift validator on
    # ProposalMetadata + ChunkMetadata is what enforces source_status
    # equality at the chunk layer; here we assert the proposal-level
    # outcome ended up in the sidecar verbatim.
    assert sidecar["programme"] == Programme.HORIZON_EUROPE.value
    assert sidecar["call_id"] == "HORIZON-CL5-2024-D3-02"
    assert sidecar["outcome"] == SourceStatus.FUNDED.value
    assert sidecar["year"] == 2024
    assert sidecar["source_path"].endswith(".pdf")


def test_confirm_rejects_missing_required_programme(
    configured_app: TestClient,
) -> None:
    """Pydantic rejects a missing programme with 422."""

    file_bytes = _make_pdf_bytes()
    files = {"file": ("HORIZON-CL3-2024-CS-01.pdf", file_bytes, "application/pdf")}
    parse_response = configured_app.post("/api/ingestion/parse", files=files)
    parse_token = parse_response.json()["parse_token"]

    response = configured_app.post(
        "/api/ingestion/confirm",
        json={
            "parse_token": parse_token,
            # programme intentionally missing
            "call_id": "HORIZON-CL3-2024-CS-01",
            "year": 2024,
            "outcome": SourceStatus.FUNDED.value,
        },
    )
    assert response.status_code == 422, response.text


def test_confirm_rejects_unknown_programme_value(configured_app: TestClient) -> None:
    """An unknown ``programme`` string is rejected with 422 (closed-enum)."""

    file_bytes = _make_pdf_bytes()
    files = {"file": ("HORIZON-CL3-2024-CS-01.pdf", file_bytes, "application/pdf")}
    parse_response = configured_app.post("/api/ingestion/parse", files=files)
    parse_token = parse_response.json()["parse_token"]

    response = configured_app.post(
        "/api/ingestion/confirm",
        json={
            "parse_token": parse_token,
            "programme": "fictional_programme",
            "call_id": "HORIZON-CL3-2024-CS-01",
            "year": 2024,
            "outcome": SourceStatus.FUNDED.value,
        },
    )
    assert response.status_code == 422, response.text


def test_confirm_rejects_unknown_parse_token(configured_app: TestClient) -> None:
    """An unknown or expired parse_token returns 404."""

    response = configured_app.post(
        "/api/ingestion/confirm",
        json={
            "parse_token": "no-such-token",
            "programme": Programme.HORIZON_EUROPE.value,
            "call_id": "HORIZON-CL3-2024-CS-01",
            "year": 2024,
            "outcome": SourceStatus.FUNDED.value,
        },
    )
    assert response.status_code == 404, response.text
    assert "parse_token" in response.text


def test_confirm_rejects_year_out_of_range(configured_app: TestClient) -> None:
    """``year`` outside [2014, 2099] fails Pydantic validation."""

    file_bytes = _make_pdf_bytes()
    files = {"file": ("HORIZON-CL3-2024-CS-01.pdf", file_bytes, "application/pdf")}
    parse_response = configured_app.post("/api/ingestion/parse", files=files)
    parse_token = parse_response.json()["parse_token"]

    response = configured_app.post(
        "/api/ingestion/confirm",
        json={
            "parse_token": parse_token,
            "programme": Programme.HORIZON_EUROPE.value,
            "call_id": "HORIZON-CL3-2024-CS-01",
            "year": 1999,
            "outcome": SourceStatus.FUNDED.value,
        },
    )
    assert response.status_code == 422, response.text


def test_health_endpoint_still_serves(configured_app: TestClient) -> None:
    """The pre-existing /health endpoint keeps working after the router merge."""

    response = configured_app.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
