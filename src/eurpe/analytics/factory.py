"""Config → :class:`AnalyticsLogger` factory.

Mirrors :func:`eurpe.security.policy.make_network_policy`: duck-typed
on the config object so callers can pass either the real
:class:`~eurpe.config.EurpeConfig` or a partial mock in tests. The
explicit decoupling keeps :mod:`eurpe.analytics` free of any
dependency on :mod:`eurpe.config`, preventing an import cycle.

Fields consulted on ``config`` (in priority order):

* ``analytics_log_path`` — callable returning the absolute :class:`Path`
  of the JSONL log. Preferred because the config validates the path in
  one place.
* ``runtime_dir`` — fallback for partial mocks that only set the
  runtime directory; we join the conventional file name there.
"""

from __future__ import annotations

from pathlib import Path

from eurpe.analytics.logger import AnalyticsLogger


def make_analytics_logger(config: object) -> AnalyticsLogger:
    """Build an :class:`AnalyticsLogger` for ``config``.

    Mirrors :func:`eurpe.security.policy.make_network_policy` — same
    duck-typed access pattern, same fallback rule. Used by
    :func:`eurpe.generation.cli.section` to obtain a logger that is
    safe to pass into the workflow.

    Fields consulted on ``config``:

    * ``analytics_log_path()`` — callable returning the absolute
      :class:`Path` of the JSONL log. Preferred when present.
    * ``runtime_dir`` — fallback used when ``analytics_log_path`` is
      not provided. The conventional file name
      ``analytics-events.log`` is joined under it.

    Raises :class:`ValueError` when neither field is available — a
    config that exposes neither cannot be used to construct a logger,
    and an emitter that fails silently here would defeat the AC1
    contract on the issue (event schema is in use).
    """

    analytics_log_getter = getattr(config, "analytics_log_path", None)
    if callable(analytics_log_getter):
        analytics_log_path = analytics_log_getter()
    else:
        runtime_dir = getattr(config, "runtime_dir", None)
        if runtime_dir is None:
            raise ValueError(
                "make_analytics_logger requires config.runtime_dir or "
                "config.analytics_log_path() to be set."
            )
        analytics_log_path = Path(runtime_dir) / "analytics-events.log"

    return AnalyticsLogger(Path(analytics_log_path))
