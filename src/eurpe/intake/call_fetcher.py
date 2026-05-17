"""Fetch call / topic metadata from the EU Funding & Tenders Portal.

Implements the backend half of issue #67 ("Auto-fill call context from
EU Funding & Tenders Portal URL").

What this module does
---------------------
Given a portal URL like::

    https://ec.europa.eu/info/funding-tenders/opportunities/portal/
        screen/opportunities/topic-details/HORIZON-CL3-2026-02-CS-ECCC-02?...

extract the topic ID and query the public SEDIA search API to recover
the call ID, topic ID, topic title, and (informational) call title.

What this module deliberately does *not* do
-------------------------------------------
The SEDIA result for current Horizon Europe topics carries only a short
title — it does **not** include Expected Outcomes / Scope text. The
TIP (Topic Information Page) backend that would carry that text is not
publicly reachable for current topics (portal is mid-migration from
``ec.europa.eu`` to ``commission.europa.eu``; see issue #67 comment for
the full endpoint matrix). Pulling Playwright in for one feature is a
~200 MB dependency hit we do not take.

The frontend surfaces a "auto-filled three fields; paste the rest from
the portal" hint so the operator is not surprised.

Network policy
--------------
This module is the one place in EURPE that intentionally crosses the
"offline by default" boundary documented in the README. Outbound calls
go to two hosts only:

* ``ec.europa.eu`` (the portal HTML, used only to validate the URL host)
* ``api.tech.ec.europa.eu`` (the SEDIA search API)

Both URLs are hard-coded; no operator input is interpolated into a host
position. Per the issue's design discussion the feature ships without a
gating toggle, but the in-app UI banner names the destinations so the
behaviour is auditable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# Public SEDIA search endpoint used by the portal frontend itself.
# ``apiKey=SEDIA`` is the public, documented key (it appears in the
# rendered SPA bundle and identifies the dataset, not the caller).
SEDIA_SEARCH_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"

# Two host names accepted on the input URL: the legacy ``ec.europa.eu``
# portal (which currently still serves the topic SPA) and the new
# ``commission.europa.eu`` domain users are starting to be redirected to.
# Anything else is rejected with a 422-style error.
_ACCEPTED_HOSTS: frozenset[str] = frozenset({"ec.europa.eu", "commission.europa.eu"})

# Canonical topic-id shape: the portal uses uppercase identifiers like
# ``HORIZON-CL3-2026-02-CS-ECCC-02``. We accept any alphanumeric +
# dash sequence with at least one dash and at least 8 characters so we
# don't false-match generic words but stay forward-compatible if the
# EU shortens or restructures the scheme. Legacy lowercase IDs (e.g.
# ``sc1-phe-coronavirus-2020-2d``) are also accepted.
_TOPIC_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{6,}-[A-Za-z0-9-]+")

# Default network budget. The SEDIA endpoint usually responds in well
# under a second; 10 s is a generous ceiling that still keeps the UI
# spinner short enough to feel interactive.
DEFAULT_TIMEOUT_SECONDS = 10.0


class CallFetchError(Exception):
    """Base class for call-fetcher failures surfaced to the API layer.

    The route handler maps subclasses to HTTP status codes so the React
    client gets a clean error envelope instead of a stack trace.
    """


class InvalidPortalURLError(CallFetchError):
    """The supplied URL is not a recognised EU portal topic URL.

    Raised when the URL is malformed, the host is not in
    :data:`_ACCEPTED_HOSTS`, or the path does not contain a parseable
    topic identifier. Maps to HTTP 422.
    """


class TopicNotFoundError(CallFetchError):
    """SEDIA returned no result for the requested topic identifier.

    Either the topic was never indexed (very recent calls can take a
    few minutes to appear) or the identifier was mistyped. Maps to
    HTTP 404 so the operator can tell this apart from a server
    problem.
    """


class PortalUnavailableError(CallFetchError):
    """The SEDIA endpoint did not respond or returned an unexpected payload.

    Covers connection errors, HTTP 5xx, and JSON that did not match the
    documented shape. Maps to HTTP 502 — the user did nothing wrong;
    the upstream is having a bad day.
    """


@dataclass(frozen=True)
class CallFetchResult:
    """Structured call/topic context recovered from the portal.

    Mirrors the subset of fields the React drafting workspace's
    structured tab exposes; ``expected_outcomes`` and ``scope`` are
    intentionally always empty in v1 (see module docstring).

    ``call_title`` is informational only — the drafting workspace does
    not currently render it, but downstream code may want it for audit
    or for tooltip text.
    """

    call_id: str
    topic_id: str
    topic_title: str
    expected_outcomes: str
    scope: str
    call_title: str
    source_url: str


def extract_topic_id(url: str) -> str:
    """Pull the topic identifier out of a Funding & Tenders Portal URL.

    Accepts the canonical SPA URL shape
    ``.../topic-details/<TOPIC_ID>?<query>`` plus minor variations
    (trailing slash, ``;`` matrix params, hash fragments).

    Raises :class:`InvalidPortalURLError` when the URL is missing a
    host, points at a non-EU domain, or has no topic identifier in
    the path.
    """

    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:  # pragma: no cover - urlparse rarely raises
        raise InvalidPortalURLError(f"could not parse URL: {exc}") from exc

    if parsed.scheme not in {"http", "https"}:
        raise InvalidPortalURLError("URL must use http or https; " f"got scheme {parsed.scheme!r}")
    if not parsed.hostname:
        raise InvalidPortalURLError("URL is missing a host")
    if parsed.hostname not in _ACCEPTED_HOSTS:
        raise InvalidPortalURLError(
            "URL must point to ec.europa.eu or commission.europa.eu; " f"got {parsed.hostname!r}"
        )

    # Match against the path only — query strings and fragments often
    # contain the topic ID too (e.g. ``?keywords=...``) and we'd
    # mis-grab them. The path segment after ``topic-details/`` is the
    # canonical place.
    marker = "/topic-details/"
    if marker not in parsed.path:
        raise InvalidPortalURLError("URL path does not contain '/topic-details/<topic_id>'")
    tail = parsed.path.split(marker, 1)[1]
    # Strip any trailing slash, matrix-param suffix, or .json extension
    # (the SPA URLs sometimes carry the latter as a logical id).
    tail = tail.split("/", 1)[0].split(";", 1)[0]
    if tail.endswith(".json"):
        tail = tail[: -len(".json")]

    match = _TOPIC_ID_PATTERN.fullmatch(tail)
    if match is None:
        raise InvalidPortalURLError(
            f"could not extract a topic identifier from path segment {tail!r}"
        )
    return match.group(0)


def _build_sedia_query(topic_id: str) -> dict[str, Any]:
    """Compose the JSON body the SEDIA search API expects.

    The shape mirrors what the portal frontend sends — a ``bool/must``
    with a ``term`` filter on ``identifier``. We also pass the topic
    ID in the ``text`` query param at call time, because SEDIA scores
    text matches and an unscored ``term``-only query sometimes returns
    a less-specific row first.
    """

    return {
        "query": {"bool": {"must": [{"term": {"identifier": topic_id}}]}},
        "languages": ["en"],
    }


def _pick_topic_result(
    payload: dict[str, Any],
    topic_id: str,
) -> dict[str, Any] | None:
    """Find the result whose ``metadata.identifier`` exactly matches.

    SEDIA returns up to ``pageSize`` results scored by relevance; the
    requested topic is usually first but we filter rather than trust
    ordering. Returns ``None`` if no result matches.
    """

    for result in payload.get("results", []):
        ident = (result.get("metadata") or {}).get("identifier") or []
        if topic_id in ident:
            return result
    return None


def _extract_single(value: Any) -> str:
    """Return the first element of a SEDIA list field, or ``""``.

    SEDIA wraps every scalar in a one-element list (legacy of its
    Elasticsearch backing). The helper centralises the unwrap so the
    main path stays readable.
    """

    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def fetch_call_context(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> CallFetchResult:
    """Fetch call/topic context for the supplied portal URL.

    ``client`` lets tests inject a stub :class:`httpx.Client` with a
    canned transport; production callers leave it at the default and
    a short-lived client is constructed for the single request.

    Raises one of :class:`InvalidPortalURLError`,
    :class:`TopicNotFoundError`, or :class:`PortalUnavailableError` —
    all of which the route handler maps to clean HTTP status codes.
    """

    topic_id = extract_topic_id(url)
    body = _build_sedia_query(topic_id)
    params = {
        "apiKey": "SEDIA",
        "text": topic_id,
        "pageSize": 5,
        "pageNumber": 1,
    }

    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=timeout)
    try:
        try:
            response = client.post(SEDIA_SEARCH_URL, params=params, json=body)
        except httpx.TimeoutException as exc:
            raise PortalUnavailableError(f"SEDIA request timed out after {timeout:.1f}s") from exc
        except httpx.HTTPError as exc:
            raise PortalUnavailableError(f"SEDIA request failed: {exc}") from exc

        if response.status_code >= 500:
            raise PortalUnavailableError(f"SEDIA returned HTTP {response.status_code}")
        if response.status_code != 200:
            # 4xx from SEDIA is usually a bad apiKey or a mangled query —
            # neither of which the operator can fix, so we treat it as
            # an upstream problem rather than reflecting the code back.
            raise PortalUnavailableError(f"SEDIA returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise PortalUnavailableError(f"SEDIA returned non-JSON payload: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    result = _pick_topic_result(payload, topic_id)
    if result is None:
        raise TopicNotFoundError(f"no SEDIA result for topic identifier {topic_id!r}")

    metadata = result.get("metadata") or {}
    call_id = _extract_single(metadata.get("callIdentifier"))
    topic_title = _extract_single(metadata.get("title"))
    call_title = _extract_single(metadata.get("callTitle"))

    if not call_id or not topic_title:
        # SEDIA returned a row but key fields are missing — almost
        # certainly an upstream-schema change. Log loudly so an
        # operator can grep the runtime log.
        logger.warning(
            "SEDIA result for %s is missing callIdentifier / title: %s",
            topic_id,
            sorted(metadata.keys()),
        )
        raise PortalUnavailableError(f"SEDIA result for {topic_id} is missing call_id or title")

    return CallFetchResult(
        call_id=call_id,
        topic_id=topic_id,
        topic_title=topic_title,
        # See module docstring — these always come back empty in v1.
        expected_outcomes="",
        scope="",
        call_title=call_title,
        source_url=url,
    )
