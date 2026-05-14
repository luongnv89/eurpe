"""Typer command implementation for ``eurpe ingest``.

Surfaces :class:`eurpe.ingestion.DoclingProposalParser` to the command
line. Behaviour:

* ``eurpe ingest <pdf>`` parses the file and prints a short summary
  (title, section count, table count, total text length) to stdout.
* ``eurpe ingest <pdf> --output <dir>`` additionally writes
  ``<dir>/<stem>.parsed.json`` containing the full :class:`ParsedProposal`
  serialized via Pydantic. The file is written only after a successful
  parse — see "Failure semantics" in the docling_parser docstring. The
  write is atomic (tmp + ``os.replace``) so a crash mid-write cannot
  leave a half-formed JSON behind.
* If the target ``.parsed.json`` already exists, the CLI refuses to
  overwrite it unless ``--overwrite``/``-f`` is passed. Silent
  clobbering of a previous parse is an easy data-loss footgun.
* On :class:`UnsupportedFormatError` or :class:`ParserError`, an error
  line is printed to stderr and the process exits with code 1. No JSON
  file is produced.
* The parser inherits :attr:`eurpe.config.EurpeConfig.offline_mode` from
  the loaded ``config.yaml``. With ``offline_mode: true`` (the default)
  Docling's OCR is disabled so no model weights are downloaded — see
  ``docling_parser`` "Offline-by-default contract".

The :func:`ingest` function is registered onto the top-level ``eurpe``
Typer in :mod:`eurpe.cli` as a flat command — see that module for the
wiring. Keeping the implementation here lets the ingestion package own
its own option signature and grow sub-actions later without touching the
top-level CLI bootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    load_config,
)
from eurpe.ingestion.docling_parser import DoclingProposalParser
from eurpe.ingestion.errors import IngestionError, ParserError, UnsupportedFormatError
from eurpe.ingestion.models import ParsedProposal


def _print_summary(parsed: ParsedProposal) -> None:
    """Print a concise human-readable summary of a parse result.

    Defined as a free function so tests can assert on the format
    independently of the Typer CliRunner. The caller (:func:`ingest`) has
    already received a ``ParsedProposal`` from the parser, so an
    ``isinstance`` guard here would be dead defensiveness — the type
    annotation is the contract.
    """

    table_count = sum(len(s.tables) for s in parsed.sections)
    typer.echo("Parsed proposal summary")
    typer.echo(f"  source        : {parsed.source_path}")
    typer.echo(f"  parser        : {parsed.parser}")
    typer.echo(f"  title         : {parsed.title or '(unknown)'}")
    typer.echo(f"  pages         : {parsed.page_count if parsed.page_count is not None else '?'}")
    typer.echo(f"  sections      : {len(parsed.sections)}")
    typer.echo(f"  tables        : {table_count}")
    typer.echo(f"  text length   : {parsed.total_text_length()} chars")


def ingest(
    pdf_path: Path = typer.Argument(
        ...,
        exists=False,
        # ``exists=False`` so we can produce our own ParserError for
        # "file not found" with the correct semantics rather than letting
        # Typer raise its own click error before we hit the parser.
        help="Path to the PDF to parse.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Directory to write <stem>.parsed.json into. Created if "
            "missing. Only written on successful parse."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help=(
            "Overwrite an existing <stem>.parsed.json. Without this "
            "flag the CLI errors out rather than silently clobbering a "
            "previous parse."
        ),
    ),
    metadata: Path | None = typer.Option(
        None,
        "--metadata",
        "-m",
        help=(
            "Optional YAML sidecar with ProposalMetadata. Currently "
            "validated only — the full join with parsed sections lands "
            "with chunking (issue #4)."
        ),
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (defaults to the repo-root config.yaml).",
    ),
) -> None:
    """Parse one PDF and print a structural summary.

    Returns exit code 0 on success, 1 on any ``IngestionError``.
    """

    # Load config so we can honour ``offline_mode`` (PRD §25 / §115). The
    # bootstrap pattern matches ``eurpe smoke``: copy the example into
    # place if no config.yaml exists yet so first-run users are not
    # punished with a ``FileNotFoundError``. Failure to load config is a
    # hard error — without it we cannot satisfy the offline contract.
    used_config_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_config_path)

    parser = DoclingProposalParser(offline=cfg.offline_mode)
    try:
        parsed = parser.parse(pdf_path)
    except UnsupportedFormatError as exc:
        # Distinct error type so the message can suggest a different
        # parser without sounding like Docling itself crashed.
        typer.echo(f"error: unsupported format: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ParserError as exc:
        # Print the formatted message (which already includes the source
        # path and underlying error). Keep stderr machine-friendly: one
        # line, leading "error: ".
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except IngestionError as exc:  # pragma: no cover - belt + suspenders
        typer.echo(f"error: ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_summary(parsed)

    # Only after a successful parse do we touch ``--output`` — this is
    # what enforces the "no partial state on failure" acceptance
    # criterion at the CLI layer.
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        out_path = output / f"{pdf_path.stem}.parsed.json"
        if out_path.exists() and not overwrite:
            typer.echo(
                f"error: output file already exists: {out_path} "
                "(pass --overwrite/-f to replace it)",
                err=True,
            )
            raise typer.Exit(code=1)
        # Atomic write: serialize to a sibling ``.tmp`` file then
        # ``Path.replace`` it onto the target. ``replace`` is an atomic
        # rename on POSIX/NTFS, so a crash mid-write cannot leave a
        # half-formed JSON visible to readers — they see either the old
        # file or the new one.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        typer.echo(f"  wrote         : {out_path}")

    if metadata is not None:
        # Read-only validation for now: confirms the sidecar parses as a
        # ProposalMetadata before chunking (issue #4) consumes it.
        # Imported lazily so the schema package isn't loaded for every
        # ``ingest`` call — keeps the CLI fast in the no-metadata case.
        import yaml

        from eurpe.schema import ProposalMetadata

        try:
            raw = yaml.safe_load(metadata.read_text(encoding="utf-8"))
            ProposalMetadata.model_validate(raw)
            typer.echo(f"  metadata      : {metadata} (validated)")
        except Exception as exc:  # noqa: BLE001 — surface any sidecar problem
            typer.echo(f"warning: metadata sidecar invalid: {exc}", err=True)
            # Sidecar errors do not fail the ingest itself — the parse
            # succeeded; we just couldn't validate the metadata. Users
            # can re-run with a fixed sidecar.

    sys.stdout.flush()
    raise typer.Exit(code=0)
