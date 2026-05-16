"""Command-line interface for EURPE.

Currently exposes a single ``smoke`` command that verifies the local
installation is consistent without performing any network access. More
commands (ingest, draft, export, ...) will be added in subsequent issues.
"""

from __future__ import annotations

from pathlib import Path

import typer

from eurpe import __version__
from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    EurpeConfig,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.generation.cli import generate_app
from eurpe.ingestion.cli import ingest as ingest_command
from eurpe.retrieval.cli import index_app
from eurpe.security import EgressDeniedError, make_network_policy

# TEST-NET-1 (RFC 5737) — guaranteed not to be a real reachable host,
# so a probe against it is the cleanest "is the default-deny gate
# actually denying?" check that doesn't depend on a hostile DNS
# resolver. If anything else allows a connection here, our config is
# broken in a way the user MUST hear about.
_SMOKE_PROBE_HOST = "192.0.2.1"
_SMOKE_PROBE_PORT = 443
_SMOKE_PROBE_SCHEME = "https"

app = typer.Typer(
    name="eurpe",
    help="EURPE — fully-local AI assistant for drafting EU research proposals.",
    no_args_is_help=True,
    add_completion=False,
)

# Register ``ingest`` as a flat command (not a sub-app). The ingestion
# module owns the implementation and the option/argument signature; here
# we just attach it to the top-level Typer so ``eurpe ingest <pdf>`` works.
# When ingestion grows multiple sub-actions (Issue #4 onwards), we can
# promote this to a sub-Typer without touching the implementation.
app.command("ingest")(ingest_command)

# ``eurpe index build`` / ``eurpe index query`` live in the retrieval
# package; mounting the sub-Typer here keeps the top-level CLI thin
# while letting the retrieval package own its own option signature.
app.add_typer(index_app, name="index")

# ``eurpe generate section`` lives in the generation package — same
# pattern as ``index``. Mounted as a sub-Typer so future ``generate``
# sub-actions (e.g., ``generate full-proposal``) can land without
# touching the top-level CLI bootstrap.
app.add_typer(generate_app, name="generate")


@app.command()
def version() -> None:
    """Print the installed EURPE version."""
    typer.echo(f"eurpe {__version__}")


@app.command()
def smoke(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (defaults to the repo-root config.yaml).",
    ),
) -> None:
    """Verify the local EURPE setup without any network access.

    Steps:
    1. Ensure ``config.yaml`` exists (copy from ``config.example.yaml`` if not).
    2. Load and validate the configuration.
    3. Resolve and create corpus + index directories.
    4. Print a summary of the loaded settings.
    """
    typer.echo("EURPE smoke test — offline only, no network calls.")
    typer.echo("")

    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    if used_path == config_path and not config_path.exists():
        # Defensive: ensure_config_file should have created it.
        raise typer.Exit(code=1)

    config = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(config)

    typer.echo(f"  config file       : {used_path}")
    typer.echo(f"  corpus_path       : {config.corpus_path}")
    typer.echo(f"  index_path        : {config.index_path}")
    typer.echo(f"  models.runtime    : {config.models.runtime}")
    typer.echo(f"  models.llm        : {config.models.llm_model}")
    typer.echo(f"  models.embedding  : {config.models.embedding_model}")
    typer.echo(f"  offline_mode      : {config.offline_mode}")
    typer.echo(f"  log_level         : {config.log_level}")

    # Egress-policy regression check. AC3 from issue #12:
    # "Release smoke test fails if a non-opt-in outbound request
    # succeeds." We exercise the gate against a TEST-NET address
    # that is never in the default allowlist; if the gate fails to
    # deny it (e.g., misconfigured allowlist, gate disabled), exit
    # non-zero with a clear regression message.
    if not _smoke_egress_probe(config):
        typer.echo(
            "  network policy   : FAIL  TEST-NET probe was ALLOWED — "
            "check your network_allowlist for an over-broad entry."
        )
        typer.echo("")
        typer.echo("[FAIL] Security regression: TEST-NET probe was ALLOWED.")
        raise typer.Exit(code=1)
    typer.echo("  network policy   : OK    TEST-NET probe denied as expected")

    typer.echo("")
    typer.echo("[OK] EURPE workspace is ready.")
    raise typer.Exit(code=0)


def _smoke_egress_probe(config: EurpeConfig) -> bool:
    """Return True iff the default-deny gate refuses a TEST-NET request.

    Splits out of :func:`smoke` so tests can pin the probe behaviour
    independently of stdout formatting. The probe call writes one line
    to the audit log either way (DENIED on success, ALLOWED on failure
    — the latter is what we surface as a hard exit code).
    """

    gate = make_network_policy(config)
    try:
        gate.check(
            host=_SMOKE_PROBE_HOST,
            port=_SMOKE_PROBE_PORT,
            scheme=_SMOKE_PROBE_SCHEME,
            path="/",
            source="smoke_probe",
        )
    except EgressDeniedError:
        # Expected outcome — gate denied, contract holds.
        return True
    # Returning at all (without an EgressDeniedError) means the gate
    # allowed the TEST-NET probe — a security regression.
    return False


if __name__ == "__main__":
    app()
