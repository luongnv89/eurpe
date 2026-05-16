"""Architectural test pinning AC3 of issue #13.

AC3: "Local analytics are disabled from external export unless a user
explicitly exports them."

The user-facing chokepoint is the ``eurpe analytics export`` CLI
(pinned by ``test_cli_export.py``). This file pins the *architectural*
invariant: the :mod:`eurpe.analytics` package itself cannot transitively
import any module capable of opening a network connection or sending
data off the host. Combined with the CLI chokepoint, this means there
is no programmatic path by which analytics data could escape the
runtime directory without the user invoking the export command.

The test:

1. Purges ``eurpe.analytics.*`` from :data:`sys.modules` so a prior
   test's imports don't pollute the baseline.
2. Snapshots :data:`sys.modules`.
3. Imports every module in the package.
4. Computes the delta and asserts no module name in the delta starts
   with any forbidden prefix.

Re-running the test in isolation (where the baseline doesn't already
contain ``socket``, etc., from other tests) is also valid because the
analytics package transitively imports only :mod:`eurpe.config` and
:mod:`eurpe.security` — neither of which pulls ``httpx`` /
``requests`` / ``socket`` / ``aiohttp`` / ``smtplib``.
"""

from __future__ import annotations

import sys

# Forbidden module name prefixes. A module name ``m`` is forbidden if
# ``m == f or m.startswith(f + ".")`` for any prefix ``f`` in this set.
# The set covers every well-known way a Python package can open an
# outbound socket: the popular HTTP clients (httpx, requests, aiohttp),
# stdlib's urllib.request, low-level sockets, and SMTP.
_FORBIDDEN_PREFIXES = {
    "httpx",
    "requests",
    "urllib.request",
    "aiohttp",
    "socket",
    "smtplib",
}


def test_analytics_package_does_not_import_outbound_io() -> None:
    """The :mod:`eurpe.analytics` package must not transitively import any outbound-IO module.

    This is the AC3 architectural backstop: even if a future
    contributor adds a code path that, say, ``httpx.post``s the
    analytics log to a remote sink, this test fails the moment they
    import ``httpx`` from inside :mod:`eurpe.analytics.*` — long
    before the call lands in any release branch.
    """

    # Purge any prior import of the analytics package so the delta
    # reflects the full transitive closure under measurement.
    for name in list(sys.modules):
        if name.startswith("eurpe.analytics"):
            del sys.modules[name]

    baseline = set(sys.modules)

    # Eagerly import every module in the package. ``noqa: F401`` keeps
    # the lint clean; the imports' side effect (loading) is the test.
    import eurpe.analytics  # noqa: F401
    import eurpe.analytics.cli  # noqa: F401
    import eurpe.analytics.events  # noqa: F401
    import eurpe.analytics.factory  # noqa: F401
    import eurpe.analytics.logger  # noqa: F401

    delta = set(sys.modules) - baseline

    forbidden_imported = {
        m for m in delta if any(m == f or m.startswith(f + ".") for f in _FORBIDDEN_PREFIXES)
    }

    assert forbidden_imported == set(), (
        f"eurpe.analytics package transitively imports forbidden modules: "
        f"{sorted(forbidden_imported)}. AC3 of issue #13 forbids any "
        "outbound-IO module in the analytics package's transitive closure."
    )


def test_analytics_package_does_not_import_generation_or_retrieval() -> None:
    """Belt-and-braces: analytics is a leaf module.

    The package is designed so that :mod:`eurpe.retrieval` and
    :mod:`eurpe.generation.workflow` / :mod:`eurpe.generation.llm`
    import *from* it, not the other way around. A reverse-edge here
    would (a) create a cycle and (b) defeat the architectural
    no-outbound-IO assertion above (the LLM client and retrievers
    use httpx).
    """

    for name in list(sys.modules):
        if name.startswith("eurpe.analytics"):
            del sys.modules[name]

    baseline = set(sys.modules)
    import eurpe.analytics  # noqa: F401
    import eurpe.analytics.cli  # noqa: F401
    import eurpe.analytics.events  # noqa: F401
    import eurpe.analytics.factory  # noqa: F401
    import eurpe.analytics.logger  # noqa: F401
    delta = set(sys.modules) - baseline

    leaf_violations = {
        m
        for m in delta
        if m == "eurpe.retrieval"
        or m.startswith("eurpe.retrieval.")
        or m in {"eurpe.generation.workflow", "eurpe.generation.llm"}
    }
    assert leaf_violations == set(), (
        f"eurpe.analytics must be a leaf module but transitively imports: "
        f"{sorted(leaf_violations)}"
    )
