"""End-to-end pipeline test — one parametrised case per PDF.

Drives every PDF in ``tests/e2e/fixtures/*.pdf`` and ``proposals/*.pdf``
through the full ``eurpe`` CLI happy path:

  ingest -> index build -> index query -> generate section -> generate audit

The list of PDFs is computed at module-load time via
:func:`tests.e2e.conftest.discover_proposals` so ``pytest.mark.parametrize``
sees a populated list. On a fresh CI checkout — where ``proposals/`` is
empty (it is in ``.gitignore``) — the synthetic fixture PDF is generated
on demand by the conftest, so collection always finds at least one case.

If neither the synthetic generation nor any real proposals are available
(e.g., ``reportlab`` is missing and ``proposals/`` is empty), the module
is skipped with a clear message via ``pytest.skip(allow_module_level=True)``.

A module-level :func:`pytest.importorskip` for ``docling`` ensures the
suite is gracefully skipped when running on a machine that has not yet
pulled the Docling dependency tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

# Module-level guards. Both must succeed before any test in this module
# runs. The order matters: importorskip raises Skipped during collection
# (so parametrize never evaluates against a missing docling), and the
# empty-pdfs guard runs after the import succeeds.
pytest.importorskip(
    "docling",
    reason="docling is required for the E2E pipeline; install with `pip install -e .[dev]`",
)

from tests._helpers.offline import write_offline_config  # noqa: E402
from tests._helpers.pipeline import run_full_pipeline  # noqa: E402
from tests._helpers.require_llm import assert_real_llm_was_used  # noqa: E402
from tests.e2e.conftest import discover_proposals  # noqa: E402

# Compute the parametrise list at module-load time. The conftest will
# have generated the synthetic fixture PDF by now (if reportlab is
# available), so this list is non-empty on a fresh CI checkout.
_PDFS: list[Path] = discover_proposals()

if not _PDFS:
    pytest.skip(
        "no proposals to E2E-test (proposals/ is empty and reportlab is "
        "not installed so the synthetic fixture PDF cannot be generated)",
        allow_module_level=True,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("pdf_path", _PDFS, ids=[p.stem for p in _PDFS])
def test_full_pipeline(pdf_path: Path, tmp_path: Path) -> None:
    """Drive ingest -> retrieve -> generate -> audit for one PDF.

    Per AC5, the run directory is deterministic: under ``tmp_path`` by
    default, overridable via ``EURPE_E2E_OUTPUT_DIR``. The latter is
    what CI workflows can set to publish artefacts.

    Assertions cover AC4 (issue #43):

    * parsed chunks > 0 (from ``index build`` summary)
    * index populated (``collection_count > 0``)
    * retrieval returns results (query CLI exits 0; ``query_hits`` is
      asserted >= 0 because the deterministic embedder may not match
      every probe well — the *contract* that retrieval ran is what
      matters, not the count)
    * at least one section generated (Markdown + JSON exist)
    * citations / audit output produced (audit JSON file present;
      step 4's in-band audit is suppressed via ``--no-audit`` so the
      explicit ``generate audit`` subcommand owns the verdict — see
      the AC3 invariant below)

    Plus AC3 (issue #45):

    * audit reports ``passed`` only when the rendered Markdown
      contains at least one citation row tied to a real chunk. When
      the audit fails the saved summary names a specific finding code
      (e.g., ``placeholder_text``, ``no_evidence_escape``) so an
      operator can act.
    """

    # Per-PDF run directory under either tmp_path (default) or the
    # operator-provided EURPE_E2E_OUTPUT_DIR. The latter is what CI
    # sets so artefacts can be uploaded.
    base = os.environ.get("EURPE_E2E_OUTPUT_DIR")
    run_root = Path(base) if base else tmp_path
    run_dir = run_root / pdf_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = write_offline_config(tmp_path)
    runner = CliRunner()

    artefacts: dict[str, Any] = run_full_pipeline(
        runner=runner,
        pdf_path=pdf_path,
        run_dir=run_dir,
        config_path=cfg_path,
    )

    # AC4 — parsed chunks > 0.
    assert (
        artefacts["chunk_count"] > 0
    ), f"index build reported 0 chunks for {pdf_path}:\n{artefacts['build_output']}"

    # AC4 — index populated.
    assert (
        artefacts["collection_count"] > 0
    ), f"collection count was 0 after build for {pdf_path}:\n{artefacts['build_output']}"

    # AC4 — retrieval returned results. The proof that retrieval ran is
    # the query CLI's exit code 0 (run_full_pipeline raises on non-zero).
    # Beyond that, we accept either ``(no results)`` (the deterministic
    # embedder might not match a generic probe well) or at least one
    # ``#N`` ranked row — both are valid CLI outputs from a working
    # retriever. A future improvement could parametrise the probe per
    # fixture so the hit-count assertion can tighten to ``>= 1``.
    assert artefacts["query_output"], "query CLI produced no output"
    assert (
        "(no results)" in artefacts["query_output"] or artefacts["query_hits"] >= 1
    ), f"query output unexpected:\n{artefacts['query_output']}"

    # AC4 — at least one section generated. The generate CLI wrote both
    # forms because run_full_pipeline uses ``--render both``.
    md_path: Path = artefacts["generated_md_path"]
    json_path: Path = artefacts["generated_json_path"]
    assert md_path.exists(), f"missing rendered Markdown: {md_path}"
    md_text = md_path.read_text(encoding="utf-8")
    assert "# " in md_text, f"rendered Markdown has no heading marker:\n{md_text[:500]}"

    assert json_path.exists(), f"missing rendered JSON: {json_path}"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert (
        payload["section_type"] == "methodology"
    ), f"unexpected section_type in {json_path}: {payload.get('section_type')!r}"

    # Issue #48 — when EURPE_E2E_REQUIRE_LLM=1, fail loudly if the
    # offline stub produced the draft (the GenerationDraft already
    # records the model that wrote each section). The gate body lives
    # in tests._helpers.require_llm so the fast-tier unit tests in
    # tests/test_e2e_require_llm.py exercise the exact same code path.
    assert_real_llm_was_used(payload)

    # AC4 — citations / audit output produced. The explicit ``audit``
    # subcommand re-checks the saved draft and writes audit.json. The
    # in-band audit in step 4 is suppressed via ``--no-audit`` so the
    # caller can assert on the explicit audit's exit code below.
    assert artefacts["audit_json_path"].exists()

    # Issue #45 AC3 — the audit summary must report passed only when
    # the rendered draft contains at least one citation row tied to a
    # real chunk. The contract holds in both directions:
    #
    # * passed (exit 0) ⇒ the rendered Markdown contains ``| 1 |``
    #   table-row marker (one row per cited chunk).
    # * not passed (exit 1) ⇒ the rendered Markdown either has no
    #   citation rows, or it contains a placeholder/escape sentence
    #   the audit gate rejects; the audit summary names the finding.
    audit_exit = artefacts["audit_exit_code"]
    audit_summary = artefacts["audit_summary"]
    has_citation_row = "| 1 |" in md_text
    if audit_exit == 0:
        assert (
            "passed (no findings)" in audit_summary or "passed with" in audit_summary
        ), f"audit reported exit 0 but the summary doesn't mark it passed:\n{audit_summary}"
        assert has_citation_row, (
            "audit passed but the rendered Markdown has no citation row — "
            "issue #45 AC3 requires at least one ``| 1 |`` row when audit passes:\n"
            f"{md_text[:1200]}"
        )
    else:
        assert "Audit findings" in audit_summary, (
            f"audit exit was {audit_exit} but no 'Audit findings' header was emitted:\n"
            f"{audit_summary}"
        )
        # The failing audit must name a specific finding code so the
        # operator can act. The two new gates from issue #45 are
        # ``no_evidence_escape`` and ``placeholder_text``; the older
        # gates (``missing_status``, ``bad_render``, ...) are valid
        # too. Just require the format ``ERROR (<code>):``.
        assert (
            "ERROR (" in audit_summary
        ), f"audit failure did not name an ERROR code:\n{audit_summary}"

    # AC5 — deterministic location: the pipeline log is in the run dir.
    assert (run_dir / "pipeline.log").exists()
