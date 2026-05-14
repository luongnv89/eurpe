"""End-to-end pipeline orchestrator for the E2E test suite.

Drives the five CLI invocations that mirror the happy-path operator
flow:

1. ``eurpe ingest <pdf> --output <run_dir> --metadata <yaml>`` —
   parse the PDF, write ``<stem>.parsed.json``, validate the sidecar.
2. ``eurpe index build <yaml> --collection <stem>`` — chunk, embed,
   upsert into a per-PDF collection. Per-PDF collection isolation
   keeps the E2E suite reproducible: a previous PDF's chunks never
   leak into the current PDF's retrieval pool.
3. ``eurpe index query <probe> --collection <stem> --threshold 0.0``
   — round-trip the retriever to prove the index is populated and
   queryable.
4. ``eurpe generate section --type methodology --output <base>``
   — generate a methodology draft using indexed evidence; emits both
   ``<base>.md`` and ``<base>.json``.
5. ``eurpe generate audit <base>.json`` — explicit, separate audit
   pass that satisfies AC4's "citations/audit output is produced"
   bullet alongside the in-band audit summary from step 4.

The orchestrator returns a structured artefact dict so the test can
make pointed assertions without re-parsing CLI output. Every CLI step's
combined stdout+stderr is also appended to ``run_dir/pipeline.log``
for human inspection when a CI run fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from eurpe.cli import app
from tests._helpers.metadata import write_metadata_yaml

# Regex for the ``index build`` summary line:
#   ``  ingested <name>: <N> chunks from <source>``
_CHUNK_COUNT_RE = re.compile(r"ingested\s+\S+\s*:\s*(\d+)\s+chunks", re.IGNORECASE)

# Regex for the ``Done. <N> chunks added; collection 'X' now holds <M>.``
_COLLECTION_COUNT_RE = re.compile(r"collection\s+\S+\s+now holds\s+(\d+)", re.IGNORECASE)


def _append_log(log_path: Path, header: str, output: str) -> None:
    """Append a step's combined stdout+stderr to ``pipeline.log``.

    Defensive: each section gets a delimiter line so the resulting log
    is greppable when CI logs the full file on failure.
    """

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {header} =====\n")
        fh.write(output or "(no output)")
        if not output.endswith("\n"):
            fh.write("\n")


def _count_query_hits(query_output: str) -> int:
    """Count result rows in ``eurpe index query`` output.

    Each result row starts with ``#N [score] ...``. The empty case
    prints ``(no results)`` instead — we treat that as 0.
    """

    if "(no results)" in query_output:
        return 0
    return sum(1 for line in query_output.splitlines() if line.startswith("#"))


def run_full_pipeline(
    runner: CliRunner,
    pdf_path: Path,
    run_dir: Path,
    config_path: Path,
    *,
    probe: str = "methodology approach",
    section_type: str = "methodology",
    intent: str = "Describe the proposed methodology for this work.",
    collection: str | None = None,
) -> dict[str, Any]:
    """Run ingest -> build -> query -> generate -> audit for one PDF.

    Returns a dict with the artefacts callers assert on:

    * ``chunk_count`` — chunks added by ``index build`` (parsed from
      stdout).
    * ``collection_count`` — total chunks now in the collection.
    * ``query_hits`` — number of result rows from ``index query``.
    * ``generated_md_path`` / ``generated_json_path`` — paths to the
      rendered Markdown + JSON drafts.
    * ``audit_summary`` — the full audit-subcommand output (stderr +
      stdout merged by CliRunner).
    * ``parsed_json_path`` — path to the ``<stem>.parsed.json`` file
      written by ``ingest``.
    * ``metadata_yaml_path`` — path to the ProposalMetadata YAML
      staged into the run dir.
    * ``ingest_output`` / ``build_output`` / ``query_output`` /
      ``generate_output`` / ``audit_output`` — raw CLI outputs for
      callers that want to assert on them. Also appended to
      ``run_dir/pipeline.log``.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    # Truncate the log up-front so re-running the same run_dir does not
    # confuse the reader with stale step output.
    log_path.write_text("", encoding="utf-8")

    # Stage the metadata YAML into the run dir. We do NOT copy the PDF;
    # ``source_path`` will be set to the absolute path of the original
    # so the build CLI can resolve it from any cwd.
    metadata_yaml_path = run_dir / f"{pdf_path.stem}.yml"
    write_metadata_yaml(metadata_yaml_path, pdf_path)

    # Per-PDF collection so previous iterations of the parametrised test
    # do not bleed chunks into the current query. Chroma collection
    # names must be 3-512 chars and ``[a-zA-Z0-9._-]``; the PDF stem can
    # contain spaces or commas (real-world filenames), so sanitise.
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", pdf_path.stem)
    if len(safe_stem) < 3:
        safe_stem = f"e2e_{safe_stem}"
    collection_name = collection or safe_stem[:512]

    # ----- 1. ingest -----
    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            str(pdf_path),
            "--output",
            str(run_dir),
            "--metadata",
            str(metadata_yaml_path),
            "--config",
            str(config_path),
        ],
    )
    _append_log(log_path, "ingest", ingest_result.output)
    if ingest_result.exit_code != 0:
        raise AssertionError(
            f"ingest failed (exit={ingest_result.exit_code}) for {pdf_path}:\n"
            f"{ingest_result.output}"
        )
    parsed_json_path = run_dir / f"{pdf_path.stem}.parsed.json"

    # ----- 2. index build -----
    build_result = runner.invoke(
        app,
        [
            "index",
            "build",
            str(metadata_yaml_path),
            "--collection",
            collection_name,
            "--config",
            str(config_path),
        ],
    )
    _append_log(log_path, "index build", build_result.output)
    if build_result.exit_code != 0:
        raise AssertionError(
            f"index build failed (exit={build_result.exit_code}) for {pdf_path}:\n"
            f"{build_result.output}"
        )

    chunk_count_match = _CHUNK_COUNT_RE.search(build_result.output)
    chunk_count = int(chunk_count_match.group(1)) if chunk_count_match else 0
    collection_count_match = _COLLECTION_COUNT_RE.search(build_result.output)
    collection_count = int(collection_count_match.group(1)) if collection_count_match else 0

    # ----- 3. index query -----
    query_result = runner.invoke(
        app,
        [
            "index",
            "query",
            probe,
            "--top-k",
            "5",
            "--threshold",
            "0.0",
            "--collection",
            collection_name,
            "--config",
            str(config_path),
        ],
    )
    _append_log(log_path, "index query", query_result.output)
    if query_result.exit_code != 0:
        raise AssertionError(
            f"index query failed (exit={query_result.exit_code}) for {pdf_path}:\n"
            f"{query_result.output}"
        )
    query_hits = _count_query_hits(query_result.output)

    # ----- 4. generate section -----
    draft_base = run_dir / pdf_path.stem
    generate_result = runner.invoke(
        app,
        [
            "generate",
            "section",
            "--type",
            section_type,
            "--intent",
            intent,
            "--threshold",
            "0.0",
            "--render",
            "both",
            "--output",
            str(draft_base),
            "--overwrite",
            "--collection",
            collection_name,
            "--config",
            str(config_path),
        ],
    )
    _append_log(log_path, "generate section", generate_result.output)
    if generate_result.exit_code != 0:
        raise AssertionError(
            f"generate section failed (exit={generate_result.exit_code}) "
            f"for {pdf_path}:\n{generate_result.output}"
        )
    generated_md_path = draft_base.with_suffix(".md")
    generated_json_path = draft_base.with_suffix(".json")

    # ----- 5. generate audit (explicit re-check) -----
    audit_result = runner.invoke(
        app,
        ["generate", "audit", str(generated_json_path)],
    )
    _append_log(log_path, "generate audit", audit_result.output)
    if audit_result.exit_code != 0:
        raise AssertionError(
            f"generate audit failed (exit={audit_result.exit_code}) "
            f"for {generated_json_path}:\n{audit_result.output}"
        )

    # Persist a JSON wrapper of the audit output so AC5 ("audit output
    # written to a deterministic location") is satisfied verbatim.
    audit_json_path = run_dir / "audit.json"
    audit_json_path.write_text(
        json.dumps({"output": audit_result.output}, indent=2),
        encoding="utf-8",
    )

    return {
        "chunk_count": chunk_count,
        "collection_count": collection_count,
        "query_hits": query_hits,
        "generated_md_path": generated_md_path,
        "generated_json_path": generated_json_path,
        "parsed_json_path": parsed_json_path,
        "metadata_yaml_path": metadata_yaml_path,
        "audit_json_path": audit_json_path,
        "audit_summary": audit_result.output,
        "ingest_output": ingest_result.output,
        "build_output": build_result.output,
        "query_output": query_result.output,
        "generate_output": generate_result.output,
        "audit_output": audit_result.output,
        "collection_name": collection_name,
    }
