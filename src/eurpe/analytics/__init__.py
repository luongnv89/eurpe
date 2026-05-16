"""Content-safe local analytics for the EURPE drafting workflow.

Implements the privacy contract from issue #13:

    * Every analytics event is a tiny structured record describing one
      operationally interesting moment (draft start / completion,
      export, feedback).
    * No event field may carry proposal content, retrieved passages,
      generated draft text, citation snippets, prompts, user-intent
      text, or partner-confidential data. The set of field names
      forbidden by the schema includes ``user_intent``, ``prompt``,
      ``text``, ``snippet``, ``passage``, ``body``, ``payload``,
      ``content`` — every event model uses ``extra="forbid"`` so
      attempting to set one is a :class:`pydantic.ValidationError`.
    * The analytics log lives under ``runtime_dir`` and is disabled
      from external export unless the user explicitly invokes
      ``eurpe analytics export`` (CLI). No other code path in this
      package copies the log outside the runtime directory.

The package is a leaf module — it MUST NOT import from
:mod:`eurpe.retrieval` or :mod:`eurpe.generation.workflow`. Those
modules import *from* here (factory + logger + event types), giving us
a clean one-way dependency that prevents cycles and lets the
architectural test in ``tests/analytics/test_no_outbound_io.py``
assert that nothing under :mod:`eurpe.analytics` transitively pulls in
``httpx`` / ``requests`` / ``socket`` / etc.

Public surface (re-exported here):

* :class:`BaseAnalyticsEvent` — abstract base class for events.
* :class:`DraftStartedEvent` — start-of-drafting signal.
* :class:`DraftCompletedEvent` — end-of-drafting signal with timing.
* :class:`ExportEvent` — recorded when the CLI writes a draft to disk.
* :class:`FeedbackEvent` — schema for future thumbs-up / regenerate UX.
* :class:`EventType` — closed enum of event labels.
* :class:`AnalyticsLogger` — instance wrapper for the bare ``log_event``.
* :func:`log_event` — JSONL append.
* :func:`make_analytics_logger` — config → logger factory.
"""

from eurpe.analytics.events import (
    BaseAnalyticsEvent,
    DraftCompletedEvent,
    DraftStartedEvent,
    EventType,
    ExportEvent,
    FeedbackEvent,
)
from eurpe.analytics.factory import make_analytics_logger
from eurpe.analytics.logger import AnalyticsLogger, log_event

__all__ = [
    "AnalyticsLogger",
    "BaseAnalyticsEvent",
    "DraftCompletedEvent",
    "DraftStartedEvent",
    "EventType",
    "ExportEvent",
    "FeedbackEvent",
    "log_event",
    "make_analytics_logger",
]
