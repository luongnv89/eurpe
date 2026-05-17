"""Tests for ``eurpe.intake.call_fetcher`` (issue #67 backend).

Two layers:

* **URL parsing** — pure-function tests on :func:`extract_topic_id`,
  exercising every shape of portal URL we expect to see plus the
  rejection cases.
* **SEDIA round-trip** — :func:`fetch_call_context` against an
  ``httpx.MockTransport`` so CI does not need network. The mocked
  payload mirrors the real SEDIA response shape captured during the
  pre-implementation portal probe (see issue #67 comment for the raw
  JSON shape).
"""

from __future__ import annotations

import httpx
import pytest

from eurpe.intake.call_fetcher import (
    CallFetchResult,
    InvalidPortalURLError,
    PortalUnavailableError,
    TopicNotFoundError,
    extract_topic_id,
    fetch_call_context,
)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def test_extract_topic_id_canonical_url() -> None:
    """The canonical SPA URL produces the expected uppercase topic ID."""

    url = (
        "https://ec.europa.eu/info/funding-tenders/opportunities/portal/"
        "screen/opportunities/topic-details/HORIZON-CL3-2026-02-CS-ECCC-02"
        "?order=DESC&pageNumber=1&pageSize=50&sortBy=relevance"
        "&keywords=Cybersecurity&isExactMatch=true&status=31094502"
    )
    assert extract_topic_id(url) == "HORIZON-CL3-2026-02-CS-ECCC-02"


def test_extract_topic_id_accepts_trailing_slash_and_json_suffix() -> None:
    """``.json`` suffixes and trailing slashes both peel cleanly."""

    base = "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/"
    assert (
        extract_topic_id(base + "HORIZON-CL3-2026-02-CS-ECCC-02/")
        == "HORIZON-CL3-2026-02-CS-ECCC-02"
    )
    assert (
        extract_topic_id(base + "HORIZON-CL3-2026-02-CS-ECCC-02.json")
        == "HORIZON-CL3-2026-02-CS-ECCC-02"
    )


def test_extract_topic_id_accepts_commission_domain() -> None:
    """The new ``commission.europa.eu`` host is allowed."""

    url = (
        "https://commission.europa.eu/funding-tenders/opportunities/portal/"
        "screen/opportunities/topic-details/HORIZON-CL3-2025-02-CS-ECCC-01"
    )
    assert extract_topic_id(url) == "HORIZON-CL3-2025-02-CS-ECCC-01"


def test_extract_topic_id_rejects_non_eu_host() -> None:
    """URLs pointing elsewhere are rejected so we never call out to a wild host."""

    with pytest.raises(InvalidPortalURLError, match="ec.europa.eu or commission"):
        extract_topic_id(
            "https://example.com/portal/screen/opportunities/topic-details/HORIZON-CL3-2026-02-CS-ECCC-02"
        )


def test_extract_topic_id_rejects_url_without_topic_segment() -> None:
    """URLs not pointing at the topic-details route are rejected with a clear message."""

    with pytest.raises(InvalidPortalURLError, match="topic-details"):
        extract_topic_id("https://ec.europa.eu/info/funding-tenders/")


def test_extract_topic_id_rejects_non_http_scheme() -> None:
    """``file://`` and similar are rejected — only http/https are allowed."""

    with pytest.raises(InvalidPortalURLError, match="scheme"):
        extract_topic_id(
            "file:///info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-CL3-2026-02-CS-ECCC-02"
        )


# ---------------------------------------------------------------------------
# SEDIA round-trip
# ---------------------------------------------------------------------------


# Shape of one ``results`` row in the SEDIA response — kept minimal but
# matches the real payload captured during the portal probe. Tests build
# variations on top of this.
_SEDIA_OK_RESULT: dict[str, object] = {
    "metadata": {
        "identifier": ["HORIZON-CL3-2026-02-CS-ECCC-02"],
        "callIdentifier": ["HORIZON-CL3-2026-02-CS-ECCC"],
        "callTitle": ["Indirectly Managed Action by the ECCC (2026)"],
        "title": [
            "Enhancing the Security, Privacy and Robustness of AI Models and Systems (SecureAI)"
        ],
    },
}

PORTAL_URL = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/portal/"
    "screen/opportunities/topic-details/HORIZON-CL3-2026-02-CS-ECCC-02"
)


def _client_with_handler(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    """Build an :class:`httpx.Client` wired to a MockTransport handler."""

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_fetch_call_context_returns_structured_result() -> None:
    """Happy path: SEDIA returns the matching row, fields land on the dataclass."""

    def handler(request: httpx.Request) -> httpx.Response:
        # The route POSTs to the SEDIA search endpoint with the topic ID
        # in both the query string and the JSON body — assert both so a
        # later refactor that drops one doesn't silently break the contract.
        assert "search-api" in request.url.host or "search-api" in str(request.url)
        assert request.url.params.get("apiKey") == "SEDIA"
        assert request.url.params.get("text") == "HORIZON-CL3-2026-02-CS-ECCC-02"
        return httpx.Response(200, json={"results": [_SEDIA_OK_RESULT]})

    with _client_with_handler(handler) as client:
        result = fetch_call_context(PORTAL_URL, client=client)

    assert isinstance(result, CallFetchResult)
    assert result.call_id == "HORIZON-CL3-2026-02-CS-ECCC"
    assert result.topic_id == "HORIZON-CL3-2026-02-CS-ECCC-02"
    assert result.topic_title.startswith("Enhancing the Security")
    assert result.call_title.startswith("Indirectly Managed Action")
    assert result.source_url == PORTAL_URL
    # v1 contract: outcomes / scope are always empty, see module docstring.
    assert result.expected_outcomes == ""
    assert result.scope == ""


def test_fetch_call_context_filters_to_exact_topic_id_match() -> None:
    """When SEDIA returns multiple rows we pick the exact identifier, not just the first."""

    other = {
        "metadata": {
            "identifier": ["HORIZON-CL3-2026-02-CS-ECCC-03"],
            "callIdentifier": ["HORIZON-CL3-2026-02-CS-ECCC"],
            "callTitle": ["Indirectly Managed Action by the ECCC (2026)"],
            "title": ["A different topic"],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # Note: the higher-scored row is first; we still want the exact match.
        return httpx.Response(200, json={"results": [other, _SEDIA_OK_RESULT]})

    with _client_with_handler(handler) as client:
        result = fetch_call_context(PORTAL_URL, client=client)

    assert result.topic_id == "HORIZON-CL3-2026-02-CS-ECCC-02"
    assert result.topic_title.startswith("Enhancing the Security")


def test_fetch_call_context_raises_topic_not_found_when_no_match() -> None:
    """An empty SEDIA result surfaces as :class:`TopicNotFoundError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with _client_with_handler(handler) as client:
        with pytest.raises(TopicNotFoundError, match="HORIZON-CL3-2026-02-CS-ECCC-02"):
            fetch_call_context(PORTAL_URL, client=client)


def test_fetch_call_context_raises_unavailable_on_5xx() -> None:
    """SEDIA HTTP 503 is reported as :class:`PortalUnavailableError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    with _client_with_handler(handler) as client:
        with pytest.raises(PortalUnavailableError, match="503"):
            fetch_call_context(PORTAL_URL, client=client)


def test_fetch_call_context_raises_unavailable_on_network_error() -> None:
    """A connection error is wrapped as :class:`PortalUnavailableError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure")

    with _client_with_handler(handler) as client:
        with pytest.raises(PortalUnavailableError, match="SEDIA request failed"):
            fetch_call_context(PORTAL_URL, client=client)


def test_fetch_call_context_raises_unavailable_when_required_fields_missing() -> None:
    """A SEDIA row missing call_id / title is treated as an upstream-schema break."""

    broken = {
        "metadata": {
            "identifier": ["HORIZON-CL3-2026-02-CS-ECCC-02"],
            # callIdentifier and title intentionally omitted
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [broken]})

    with _client_with_handler(handler) as client:
        with pytest.raises(PortalUnavailableError, match="missing call_id or title"):
            fetch_call_context(PORTAL_URL, client=client)


def test_fetch_call_context_propagates_url_validation_error() -> None:
    """An invalid URL never triggers a network call — it raises during parse."""

    sentinel: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        sentinel.append(True)
        return httpx.Response(200, json={"results": []})

    with _client_with_handler(handler) as client:
        with pytest.raises(InvalidPortalURLError):
            fetch_call_context("https://example.com/", client=client)

    assert sentinel == [], "no network call should have been made for invalid URL"
