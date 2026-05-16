"""Typer sub-app for ``eurpe analytics ...``.

Currently exposes a single command — ``export`` — which is the **only**
code path in the entire codebase that copies the analytics JSONL log
outside the runtime directory. Issue #13 AC3 forbids any other code
path from doing so; the rest of :mod:`eurpe.analytics` writes to the
log and reads nothing back.

The ``export`` command:

1. Loads config to learn where the source log lives.
2. Refuses if the source log doesn't exist (nothing to export).
3. Refuses to clobber an existing destination unless ``--overwrite``
   was passed.
4. Performs the copy atomically: ``shutil.copyfile`` to a sibling
   ``.tmp`` file, then :meth:`Path.replace` into place. Same pattern
   as ``_atomic_write`` in :mod:`eurpe.generation.cli`.
5. Prints a one-line summary including the event count so the user
   can confirm what they exported.

The output flag is REQUIRED — there is no default destination. This is
the AC3 chokepoint: an export attempt with no explicit ``--output``
fails fast, preventing accidental exports to a current-working
directory artefact whose location the operator may not be tracking.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)

# Sub-Typer mounted onto the top-level app in :mod:`eurpe.cli`. Named
# ``analytics`` so the surface is ``eurpe analytics export ...``.
analytics_app = typer.Typer(
    name="analytics",
    help=(
        "Manage the local analytics event log. Default-disabled from "
        "external export — only the ``export`` command writes the log "
        "outside the runtime directory."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _count_lines(path: Path) -> int:
    """Count non-empty lines in ``path``.

    Used only for the user-facing summary; an off-by-one here is not
    correctness-critical. We read in binary and count newlines so the
    count is robust to mixed encodings — the writer always emits UTF-8
    so this is defensive.
    """

    count = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


@analytics_app.command("export")
def export_events(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help=(
            "Where to copy the analytics JSONL file. Required: no default "
            "destination, per issue #13 AC3 (analytics are disabled from "
            "external export unless the user explicitly invokes this command)."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite the destination if it already exists.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (defaults to the repo-root config.yaml).",
    ),
) -> None:
    """Explicitly export the local analytics JSONL log.

    This is the ONLY code path that copies analytics data outside the
    runtime directory. Required by issue #13 AC3 — analytics are
    disabled from external export unless the user explicitly invokes
    this command. The output flag has no default so a stray
    ``eurpe analytics export`` cannot land the file in an unexpected
    location.
    """

    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    # Ensure runtime_dir exists so the source-path calculation is
    # well-defined even when the user runs ``export`` before any
    # events have been written.
    ensure_runtime_dirs(cfg)

    source = cfg.analytics_log_path()
    if not source.exists():
        typer.echo(
            f"error: analytics log not found at {source}. "
            "No events have been recorded yet.",
            err=True,
        )
        raise typer.Exit(code=1)

    if output.exists() and not overwrite:
        typer.echo(
            f"error: output file already exists: {output} "
            "(pass --overwrite/-f to replace it)",
            err=True,
        )
        raise typer.Exit(code=1)

    # Atomic copy: a sibling ``.tmp`` ensures a single ``replace`` call
    # swaps the file into place without ever leaving a partially-written
    # destination. The tmp lives in the same directory as the target
    # to keep the rename intra-filesystem (cross-device renames fail).
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    try:
        shutil.copyfile(source, tmp_path)
        tmp_path.replace(output)
    except Exception:
        # Defensive: clean up any tmp file before re-raising so a
        # partial state doesn't confuse a follow-up run.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:  # pragma: no cover - defensive
                pass
        raise

    event_count = _count_lines(output)
    typer.echo(f"  exported {event_count} events to {output}")
    raise typer.Exit(code=0)
