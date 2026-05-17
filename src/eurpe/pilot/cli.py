"""Typer commands for ``eurpe pilot ...``.

Two sub-commands today, mirroring the operator-facing pattern used by
:mod:`eurpe.benchmarks.cli` and :mod:`eurpe.generation.cli`:

* ``eurpe pilot run`` — drive one MVP pilot validation end-to-end and
  optionally persist artefacts under ``--output-dir``.
* ``eurpe pilot rate`` — post-edit a saved pilot report to attach one
  coordinator's satisfaction rating; re-computes the verdict so the
  on-disk file stays internally consistent.

The default runtime is ``deterministic`` so a fresh clone produces a
complete smoke-mode pilot with zero network. Passing
``--runtime ollama`` swaps in the real backends (same convention as
the benchmark CLI). The runtime is recorded in the report's embedded
:class:`BenchmarkReport.runtime` fingerprint and in the rendered
Markdown, so a reviewer can always tell which path produced the
numbers.
"""

from __future__ import annotations

from pathlib import Path

import typer

from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.generation.llm import DeterministicLLMClient, LLMClient, make_llm_client
from eurpe.pilot.models import (
    GoNoGoVerdict,
    PilotMode,
    PilotReport,
    SatisfactionRating,
)
from eurpe.pilot.runner import (
    DEFAULT_SECTION_TYPES,
    PilotConfig,
    PilotRunError,
    attach_satisfaction,
    load_pilot_report,
    render_pilot_report_markdown,
    run_pilot,
)
from eurpe.retrieval import DeterministicHashEmbedder, Embedder
from eurpe.retrieval.embeddings import make_embedder
from eurpe.schema import SectionType

pilot_app = typer.Typer(
    name="pilot",
    help=(
        "Run MVP pilot validation against the indexed corpus. Defaults "
        "to a deterministic smoke-mode pilot; pass --runtime ollama "
        "for a real-LLM coordinator pilot."
    ),
    no_args_is_help=True,
    add_completion=False,
)


_RUNTIME_DETERMINISTIC = "deterministic"
_RUNTIME_OLLAMA = "ollama"


def _select_backends(
    runtime: str,
    config_path: Path,
) -> tuple[Embedder, LLMClient, str]:
    """Mirror of :func:`eurpe.benchmarks.cli._select_backends`.

    Duplicated rather than imported so the pilot module does not
    couple to the benchmark CLI's private surface (a future
    benchmark-CLI refactor should not break the pilot). The label is
    the *requested* runtime so the rendered report honestly says
    ``ollama`` even when the factory falls back to a deterministic
    stub (e.g., Ollama daemon unreachable in offline mode); the
    actual backend class is visible in the embedded benchmark
    report's runtime fingerprint.
    """

    if runtime == _RUNTIME_DETERMINISTIC:
        return (
            DeterministicHashEmbedder(dimension=128),
            DeterministicLLMClient(),
            _RUNTIME_DETERMINISTIC,
        )

    if runtime != _RUNTIME_OLLAMA:
        raise typer.BadParameter(
            f"Unknown runtime: {runtime!r}. Choose 'deterministic' or 'ollama'."
        )

    used_path = ensure_config_file(config_path, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(cfg)
    return make_embedder(cfg), make_llm_client(cfg), _RUNTIME_OLLAMA


def _write_pilot_artefacts(
    report: PilotReport,
    *,
    output_json: Path | None,
    output_markdown: Path | None,
) -> None:
    """Atomically write the pilot JSON / Markdown artefacts.

    Mirrors the atomic-write pattern from
    :mod:`eurpe.benchmarks.cli`: write to a sibling ``.tmp`` file and
    ``Path.replace`` it into place. ``None`` for either argument is a
    no-op so the operator can request only one format. The directories
    are created if missing.
    """

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_json.with_suffix(output_json.suffix + ".tmp")
        try:
            tmp.write_text(report.to_json() + "\n", encoding="utf-8")
            tmp.replace(output_json)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover - defensive
                    pass
            raise

    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_markdown.with_suffix(output_markdown.suffix + ".tmp")
        try:
            tmp.write_text(render_pilot_report_markdown(report), encoding="utf-8")
            tmp.replace(output_markdown)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover - defensive
                    pass
            raise


@pilot_app.command("run")
def pilot_run(
    call_id: str = typer.Option(
        "HORIZON-CL5-2024-D3-02",
        "--call-id",
        help=(
            "EU call identifier the pilot exercises (AC1 of issue #21). "
            "The default points at the Horizon Europe 2024 cybersecurity "
            "call used as the example across the E2E suite and the "
            "SANCUS fixture."
        ),
    ),
    proposal_title: str = typer.Option(
        "MVP Pilot Synthetic Corpus",
        "--proposal-title",
        help="Title stamped onto the synthetic indexed proposal.",
    ),
    section_types: list[str] = typer.Option(
        None,
        "--section-type",
        "-s",
        help=(
            "Section type to draft. Repeat for multiple. Must be one of "
            "the SectionType enum values (e.g., methodology, impact, "
            "implementation). Defaults to (methodology, impact, "
            "implementation) per AC1's 'at least three' requirement."
        ),
    ),
    mode: str = typer.Option(
        "smoke",
        "--mode",
        help=(
            "Pilot mode: 'smoke' (deterministic stubs, the default) or "
            "'coordinator' (real-LLM run, requires --runtime ollama)."
        ),
    ),
    runtime: str = typer.Option(
        _RUNTIME_DETERMINISTIC,
        "--runtime",
        help=(
            "Backend runtime: 'deterministic' (offline default) or "
            "'ollama' (real local LLM + embedder)."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help=(
            "Directory to persist per-section drafts (JSON + Markdown) "
            "plus the aggregate pilot-report.json. Required when the "
            "operator wants the release-notes-attachable artefacts."
        ),
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Path to write the aggregate pilot report as JSON.",
    ),
    output_markdown: Path | None = typer.Option(
        None,
        "--output-markdown",
        help="Path to write the aggregate pilot report as Markdown.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml (used only when --runtime ollama).",
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        help=(
            "Free-text operator notes captured on the report. Same "
            "privacy contract as analytics events — no proposal content."
        ),
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="Forwarded to GenerationRequest.top_k_examples.",
        min=1,
        max=20,
    ),
) -> None:
    """Drive one MVP pilot validation run end-to-end.

    Prints a human-readable summary to stdout (the same fields the
    Markdown render carries) and exits 0 unless the verdict is
    ``NO_GO`` (exit 1). ``CONDITIONAL`` is a clean exit because a
    smoke-mode pilot is not a release gate by itself — the operator
    is expected to follow up with a coordinator pilot.
    """

    parsed_mode = _parse_mode(mode)
    section_enums = _parse_section_types(section_types)
    embedder, llm, runtime_label = _select_backends(runtime, config_path)

    pilot_config = PilotConfig(
        mode=parsed_mode,
        call_id=call_id,
        proposal_title=proposal_title,
        section_types=section_enums,
        top_k_examples=top_k,
        notes=notes,
    )

    try:
        report = run_pilot(
            config=pilot_config,
            output_dir=output_dir,
            embedder=embedder,
            llm=llm,
        )
    except PilotRunError as exc:
        typer.echo(f"[FAIL] {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _write_pilot_artefacts(report, output_json=output_json, output_markdown=output_markdown)

    # Stdout summary: mirrors the benchmark CLI's pretty-print style.
    typer.echo(f"Pilot mode      : {report.mode.value}")
    typer.echo(f"Call ID         : {report.call_id}")
    typer.echo(f"Proposal        : {report.proposal_title}")
    typer.echo(f"Runtime         : {runtime_label}")
    typer.echo(f"Sections        : {len(report.section_results)}")
    typer.echo(
        f"Smoke probe     : {'PASS' if report.smoke.passed else 'FAIL'} "
        f"(exit {report.smoke.exit_code})"
    )
    audit_line = f"{report.audit.passed_drafts}/{report.audit.audited_drafts} drafts passed"
    typer.echo(f"Audit           : {audit_line}")
    typer.echo(f"Citation issues : {len(report.citation_issues)}")
    typer.echo(f"Verdict         : {report.verdict.value.upper()}")

    if report.verdict == GoNoGoVerdict.NO_GO:
        raise typer.Exit(code=1)


@pilot_app.command("rate")
def pilot_rate(
    report_path: Path = typer.Argument(
        ...,
        help="Path to the pilot-report.json produced by 'eurpe pilot run'.",
    ),
    section_type: str = typer.Option(
        ...,
        "--section-type",
        "-s",
        help="Section type being rated (must match one in the report).",
    ),
    coordinator_id: str = typer.Option(
        ...,
        "--coordinator-id",
        help=(
            "Anonymous coordinator identifier (e.g., 'coord-a'). MUST "
            "NOT be a real name, email, or other deanonymising id."
        ),
    ),
    rating: int = typer.Option(
        ...,
        "--rating",
        min=1,
        max=5,
        help="1-5 Likert satisfaction rating (PRD floor: ≥4 for GO).",
    ),
    time_saved_minutes: int = typer.Option(
        0,
        "--time-saved",
        min=0,
        help=("Approximate minutes saved against manual drafting (AC2 of issue #21)."),
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        help="Short coordinator note (content-safe — no proposal text).",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help=(
            "Path to write the updated report. Defaults to overwriting "
            "the input file in-place after a successful update."
        ),
    ),
) -> None:
    """Attach one coordinator's satisfaction rating to a saved pilot report.

    Reads the report, validates ``section_type`` is present, builds a
    :class:`SatisfactionRating`, attaches it via
    :func:`attach_satisfaction`, and writes the result back to disk.
    Re-computes the verdict so the updated file is internally
    consistent — a smoke-mode report whose verdict moves from
    ``CONDITIONAL`` to ``GO`` after the last rating lands here.
    """

    if not report_path.exists():
        typer.echo(f"[FAIL] pilot report not found: {report_path}", err=True)
        raise typer.Exit(code=1)

    report = load_pilot_report(report_path)
    rating_record = SatisfactionRating(
        coordinator_id=coordinator_id,
        rating=rating,
        time_saved_minutes=time_saved_minutes,
        notes=notes,
    )

    try:
        updated = attach_satisfaction(
            report=report,
            section_type=section_type,
            rating=rating_record,
        )
    except PilotRunError as exc:
        typer.echo(f"[FAIL] {exc}", err=True)
        raise typer.Exit(code=1) from exc

    target = output_path if output_path is not None else report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(updated.to_json() + "\n", encoding="utf-8")
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - defensive
                pass
        raise

    typer.echo(
        f"Rated {section_type} by {coordinator_id}: {rating}/5 — "
        f"new verdict: {updated.verdict.value.upper()}"
    )


def _parse_mode(value: str) -> PilotMode:
    """Parse the CLI ``--mode`` string into a :class:`PilotMode`.

    A small helper so the BadParameter message names the accepted set
    rather than letting StrEnum's KeyError leak out.
    """

    try:
        return PilotMode(value)
    except ValueError as exc:
        accepted = ", ".join(m.value for m in PilotMode)
        raise typer.BadParameter(f"--mode must be one of: {accepted} (got {value!r})") from exc


def _parse_section_types(values: list[str] | None) -> tuple[SectionType, ...]:
    """Parse repeated ``--section-type`` values into :class:`SectionType` tuple.

    ``None`` (no flag) falls back to :data:`DEFAULT_SECTION_TYPES`.
    Empty list (``--section-type ""`` quirks) is treated the same as
    ``None`` — the AC1 'at least three' requirement is enforced by
    the :class:`PilotConfig` validator, not here.
    """

    if not values:
        return DEFAULT_SECTION_TYPES
    parsed: list[SectionType] = []
    accepted = ", ".join(s.value for s in SectionType)
    for v in values:
        try:
            parsed.append(SectionType(v))
        except ValueError as exc:
            raise typer.BadParameter(
                f"--section-type must be one of: {accepted} (got {v!r})"
            ) from exc
    return tuple(parsed)
