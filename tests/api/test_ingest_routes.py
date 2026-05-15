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

    ``flavour`` is a marker word baked into the section text so a test
    can drive the chunker to produce different chunk_ids on a second
    call (used by the corrected-document reindex test).
    """

    def __init__(
        self,
        *,
        title: str = "Synthetic Test Proposal",
        flavour: str = "default",
    ) -> None:
        self._title = title
        self._flavour = flavour

    def parse(self, path: Path) -> ParsedProposal:
        return ParsedProposal(
            source_path=str(path.resolve()),
            title=self._title,
            sections=[
                ParsedSection(
                    heading="1. Excellence",
                    level=1,
                    text=(
                        f"Excellence body text in flavour {self._flavour}. "
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
                        f"Impact narrative in flavour {self._flavour}. "
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
    parse_body = parse_response.json()
    parse_token = parse_body["parse_token"]
    staged_source_path = Path(parse_body["source_path"])

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
    archived_source_path = Path(sidecar["source_path"])
    assert archived_source_path.exists()
    assert archived_source_path.suffix == ".pdf"
    assert archived_source_path.parent == sidecar_path.parent
    assert not staged_source_path.exists()


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


def test_confirm_rejects_missing_required_topic(configured_app: TestClient) -> None:
    """Pydantic rejects a missing topic_id with 422."""

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
            # topic_id intentionally missing
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
            "topic_id": "HORIZON-CL3-2024-CS-01-01",
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


# ---------------------------------------------------------------------------
# Duplicate-detection coverage (issue #11)
# ---------------------------------------------------------------------------


def _parse_and_confirm(
    client: TestClient,
    *,
    filename: str,
    body: bytes,
    confirm_extra: dict | None = None,
) -> tuple[int, dict]:
    """Drive the parse → confirm round trip and return ``(status, body)``.

    Keeps the call sites readable: the dedup tests below all run the
    same shape of request and differ only in the bytes / metadata.
    """

    files = {"file": (filename, body, "application/pdf")}
    parse_response = client.post("/api/ingestion/parse", files=files)
    assert parse_response.status_code == 200, parse_response.text
    parse_token = parse_response.json()["parse_token"]

    confirm_body: dict = {
        "parse_token": parse_token,
        "programme": Programme.HORIZON_EUROPE.value,
        "call_id": "HORIZON-CL5-2024-D3-02",
        "topic_id": "HORIZON-CL5-2024-D3-02-01",
        "year": 2024,
        "outcome": SourceStatus.FUNDED.value,
        "proposal_title": "Synthetic Test Proposal",
        "consortium_acronym": "STP",
    }
    if confirm_extra:
        confirm_body.update(confirm_extra)
    confirm_response = client.post("/api/ingestion/confirm", json=confirm_body)
    return confirm_response.status_code, (
        confirm_response.json() if confirm_response.content else {}
    )


def test_confirm_blocks_hard_duplicate_with_409(configured_app: TestClient) -> None:
    """A second confirm with byte-identical content is rejected with 409."""

    body = _make_pdf_bytes()
    status_a, response_a = _parse_and_confirm(
        configured_app, filename="HORIZON-CL5-2024-D3-02-883588.pdf", body=body
    )
    assert status_a == 200, response_a
    assert response_a["duplicate_warning"] is None
    assert response_a["replaced_document_id"] is None

    status_b, response_b = _parse_and_confirm(
        configured_app, filename="HORIZON-CL5-2024-D3-02-883588.pdf", body=body
    )
    assert status_b == 409, response_b
    detail = response_b["detail"].lower()
    assert "duplicate" in detail


def test_confirm_reindexes_corrected_document_no_orphans(
    configured_app: TestClient, tmp_path: Path
) -> None:
    """A corrected PDF under the same filename reindexes without orphans.

    The stub parser is rebound mid-test so the second ingest emits
    different chunk text (so the chunker produces different chunk_ids)
    and the test can prove the index ends up with the corrected
    chunk count.
    """

    from eurpe.api import dependencies as deps

    app.dependency_overrides[deps.get_parser] = lambda: _StubParser(flavour="original")
    status_a, response_a = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-883588.pdf",
        body=_make_pdf_bytes(),
    )
    assert status_a == 200, response_a
    original_chunks = response_a["chunks_added"]
    assert original_chunks > 0

    # Rebind the parser to emit different chunk text, and use different
    # bytes so the hash differs (otherwise BLOCK_HARD short-circuits).
    app.dependency_overrides[deps.get_parser] = lambda: _StubParser(flavour="corrected")
    status_b, response_b = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-883588.pdf",
        body=_make_pdf_bytes() + b"corrected",
    )
    assert status_b == 200, response_b
    # REINDEX path: the old doc_id was replaced.
    assert response_b["replaced_document_id"] is not None
    # Silent for REINDEX (operator intent is unambiguous).
    assert response_b["duplicate_warning"] is None

    # Verify the index now holds exactly the new chunks for that doc_id —
    # no orphans from the original ingest.
    index = deps.get_index(deps.get_config(), collection="default")
    doc_id = response_b["replaced_document_id"]
    assert index.find_by_document_id(doc_id) == response_b["chunks_added"]


def test_confirm_blocks_soft_duplicate_without_force(
    configured_app: TestClient,
) -> None:
    """Same title+call_id but different bytes and different filename → 409."""

    body_a = _make_pdf_bytes()
    status_a, _ = _parse_and_confirm(
        configured_app, filename="HORIZON-CL5-2024-D3-02-aaaaaa.pdf", body=body_a
    )
    assert status_a == 200

    # Different filename → different document_id; different bytes → different hash.
    status_b, response_b = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-bbbbbb.pdf",
        body=body_a + b"different",
    )
    assert status_b == 409, response_b
    detail = response_b["detail"].lower()
    assert "duplicate suspected" in detail
    assert "force=true" in detail


def test_confirm_replaces_soft_duplicate_with_force_true(
    configured_app: TestClient,
) -> None:
    """``force=true`` allows the soft-duplicate path to replace the existing record."""

    body_a = _make_pdf_bytes()
    status_a, _ = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-aaaaaa.pdf",
        body=body_a,
    )
    assert status_a == 200

    status_b, response_b = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-bbbbbb.pdf",
        body=body_a + b"different",
        confirm_extra={"force": True},
    )
    assert status_b == 200, response_b
    assert response_b["duplicate_warning"] is not None
    assert response_b["replaced_document_id"] is not None


def test_confirm_does_not_archive_when_blocked(
    configured_app: TestClient,
) -> None:
    """A 409 hard-duplicate response must not leave a second archive on disk."""

    from eurpe.api import dependencies as deps

    body = _make_pdf_bytes()
    status_a, response_a = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-883588.pdf",
        body=body,
    )
    assert status_a == 200, response_a
    archived_path = Path(response_a["sidecar_path"]).with_suffix("")
    # ``sidecar_path`` ends in ``.metadata.yaml``; strip both suffixes to
    # find the archive PDF directory.
    archive_dir = archived_path.parent
    before_count = sum(1 for _ in archive_dir.glob("*.pdf"))
    assert before_count >= 1

    status_b, _ = _parse_and_confirm(
        configured_app,
        filename="HORIZON-CL5-2024-D3-02-883588.pdf",
        body=body,
    )
    assert status_b == 409
    after_count = sum(1 for _ in archive_dir.glob("*.pdf"))
    # The blocked second confirm did not produce a second archived PDF.
    assert after_count == before_count
    del deps  # appease the unused-import linter; the import is for the side-effect free path.
