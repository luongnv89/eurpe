"""JSONL writer for local analytics events.

Every call to :func:`log_event` (or :meth:`AnalyticsLogger.log`) appends
one line — a single JSON object — to the configured analytics log
under ``cfg.runtime_dir / "analytics-events.log"``. The structural
guarantee is that only fields declared on the pinned
:class:`~eurpe.analytics.events.BaseAnalyticsEvent` subclasses land in
the file. Because every event subclass uses ``extra="forbid"`` +
``frozen=True``, this writer cannot smuggle a stray ``text``,
``snippet``, ``payload``, ``content``, ``passage``, ``prompt``,
``user_intent``, or ``body`` kwarg into the log.

The module mirrors :mod:`eurpe.security.audit` deliberately:

* per-path child logger (``eurpe.analytics.events.<sha1>``) with
  ``propagate=False`` so analytics lines never leak into stdout / the
  root logger / a sibling path's log;
* a process-global handler cache keyed by the resolved absolute path,
  so multiple loggers pointing at the same file share one handler;
* a ``_reset_handlers_for_tests`` hook for the autouse fixture every
  test module in ``tests/analytics`` installs;
* no formatter decorations on the writer — we serialise JSON ourselves
  with ``sort_keys=True, separators=(",", ":")`` so the file is line-
  delimited JSON parseable by any JSONL reader.

The export contract (issue #13 AC3) lives elsewhere — see
:mod:`eurpe.analytics.cli`. This module only writes; it never reads
the log back out, and it never copies the file. Combined with
``propagate=False``, that means analytics data does not leave the
runtime directory through any code path in this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock

from eurpe.analytics.events import BaseAnalyticsEvent

# Parent namespace for every analytics logger. Each distinct
# analytics-log path gets its OWN child logger
# ``eurpe.analytics.events.<hash>``; the parent itself is never written
# to. Per-path isolation stops two AnalyticsLogger instances pointing at
# different files from cross-contaminating each other's JSONL when both
# run in the same process. ``propagate=False`` keeps these lines out of
# stdout / the root logger.
_ANALYTICS_LOGGER_NAME = "eurpe.analytics.events"

# Cache of ``{absolute_path: FileHandler}`` keyed by resolved absolute
# path. We memoise so a process that builds multiple loggers pointing
# at the SAME file doesn't accumulate duplicate handlers on that path's
# child logger.
_HANDLERS: dict[str, logging.FileHandler] = {}
_LOCK = Lock()


def _child_logger_name(analytics_log_path: Path) -> str:
    """Return the child-logger name used for one analytics-log path.

    Hash of the absolute path → short hex digest. The logging-tree
    convention forbids slashes / spaces in logger names, so we hash
    rather than embed the path.
    """

    digest = hashlib.sha1(
        str(analytics_log_path.resolve()).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]
    return f"{_ANALYTICS_LOGGER_NAME}.{digest}"


def _get_analytics_logger(analytics_log_path: Path) -> logging.Logger:
    """Return the per-path child logger, attaching a :class:`FileHandler` if needed.

    Each distinct analytics-log path gets its own child logger
    (``eurpe.analytics.events.<hash>``) with its own handler. Two
    loggers pointing at different files share NO handlers, so a write
    to one file cannot leak into the other. ``propagate=False`` on
    every child stops the line from bubbling to the parent / root /
    stdout — analytics data is not log noise; it must not surface in
    the operator's terminal.

    The parent directory MUST already exist — the consumer guarantees
    this via :func:`eurpe.config.ensure_runtime_dirs`. We surface a
    clear :class:`FileNotFoundError` if it doesn't so the failure mode
    is loud (a missing ``runtime_dir`` is a bootstrapping bug, not an
    analytics bug).
    """

    logger = logging.getLogger(_child_logger_name(analytics_log_path))
    # Don't propagate to the parent / root — these lines must not leak
    # into stdout or any other handler the application installs for
    # general logging, AND they must not leak into a sibling path's
    # child logger via the shared parent.
    logger.propagate = False
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    key = str(analytics_log_path.resolve())
    with _LOCK:
        existing = _HANDLERS.get(key)
        if existing is not None and existing in logger.handlers:
            return logger
        # Either no cached handler or it was removed (e.g., test teardown).
        # Build a fresh one.
        parent = analytics_log_path.parent
        if not parent.exists():
            raise FileNotFoundError(
                f"analytics-events.log parent directory does not exist: {parent}. "
                "Call eurpe.config.ensure_runtime_dirs(config) before constructing "
                "an AnalyticsLogger."
            )
        handler = logging.FileHandler(analytics_log_path, mode="a", encoding="utf-8")
        # Plain message — we serialise JSON ourselves, no extra formatter
        # decorations (timestamps, levels) so the file is line-delimited JSON.
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        # Remove any stale handler pointing at the same path before adding.
        if existing is not None:
            try:
                logger.removeHandler(existing)
                existing.close()
            except Exception:  # pragma: no cover - defensive; best-effort cleanup  # nosec B110
                pass
        logger.addHandler(handler)
        _HANDLERS[key] = handler
    return logger


def log_event(
    analytics_log_path: Path,
    *,
    event: BaseAnalyticsEvent,
) -> None:
    """Append one JSONL record describing an analytics event.

    The ``event`` argument is keyword-only so a future regression that
    swaps the analytics path and the event at the call site is a
    :class:`TypeError`, not a silent miswrite. The event is serialised
    via :meth:`BaseAnalyticsEvent.model_dump` with ``mode="json"`` so
    every field — including the UTC ``timestamp`` and the
    ``EventType`` discriminator — is rendered as a JSON-native scalar.

    The record is dumped with ``sort_keys=True, separators=(",", ":")``
    so the output is canonical (deterministic ordering across runs,
    minimal whitespace). One line per event; no leading or trailing
    whitespace; safe to ``jq`` or ``grep`` over.
    """

    record = event.model_dump(mode="json")
    logger = _get_analytics_logger(analytics_log_path)
    logger.info(json.dumps(record, sort_keys=True, separators=(",", ":")))


def _reset_handlers_for_tests() -> None:
    """Detach and close every cached analytics handler. Test-only.

    Pytest reuses the same process across tests, so handlers attached
    to a ``tmp_path`` that no longer exists become stale. Tests that
    need a clean logger state can call this from an autouse fixture
    teardown — and every test module under ``tests/analytics`` does.
    """

    with _LOCK:
        for key, handler in list(_HANDLERS.items()):
            # ``key`` is the resolved absolute path string used to
            # build the child-logger name; reconstruct it to detach.
            child = logging.getLogger(_child_logger_name(Path(key)))
            try:
                child.removeHandler(handler)
                handler.close()
            except Exception:  # pragma: no cover - defensive; best-effort cleanup  # nosec B110
                pass
        _HANDLERS.clear()


class AnalyticsLogger:
    """Tiny wrapper bundling a log path with a typed ``.log()`` method.

    Construction-time validation: the parent directory of
    ``analytics_log_path`` MUST exist. Tests using ``tmp_path`` satisfy
    this for free; production code satisfies it via
    :func:`eurpe.config.ensure_runtime_dirs`.

    Use this class — rather than the bare :func:`log_event` —
    everywhere the workflow / CLI wants to *optionally* emit events.
    Storing the analytics logger as an attribute on the workflow and
    null-checking it at the call site (``if self._analytics is not
    None:``) is cleaner than carrying a path argument through every
    method.
    """

    def __init__(self, analytics_log_path: Path) -> None:
        path = Path(analytics_log_path)
        parent = path.parent
        if not parent.exists():
            raise FileNotFoundError(
                f"analytics-events.log parent directory does not exist: {parent}. "
                "Call eurpe.config.ensure_runtime_dirs(config) before constructing "
                "an AnalyticsLogger."
            )
        self._analytics_log_path = path

    @property
    def analytics_log_path(self) -> Path:
        """Absolute path of the JSONL log this logger writes to."""

        return self._analytics_log_path

    def log(self, event: BaseAnalyticsEvent) -> None:
        """Append one event to the configured analytics log.

        Thin wrapper over :func:`log_event` — kept as an instance
        method so call sites can pass the bound logger around as an
        opaque dependency without learning the path module.
        """

        log_event(self._analytics_log_path, event=event)
