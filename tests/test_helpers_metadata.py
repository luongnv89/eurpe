"""Tests for the ``tests._helpers.metadata`` wiring.

The parser is tested in isolation in ``test_filename_parser``; here we
verify the merge contract documented in
:func:`synthesise_proposal_metadata`'s docstring:

* defaults → filename-parsed → overrides → derived (proposal_title /
  source_path).
* A sibling ``<stem>.yml`` always wins over filename-parsed values.
* A filename without a programme alias logs a warning and falls back
  to the default programme (issue #47 acceptance criteria).
"""

from __future__ import annotations

import logging

import yaml

from tests._helpers.metadata import (
    load_or_synthesise_metadata,
    synthesise_proposal_metadata,
)

SANCUS_NAME = "SANCUS_PROPOSAL_952672-SANCUS-H2020-SU-ICT-2019-PART_B_Section_1.pdf"
GEIGER_NAME = "GEIGER_883588--SEALED-PROPOSAL.pdf"


def test_sancus_filename_yields_h2020_metadata(tmp_path):
    pdf_path = tmp_path / SANCUS_NAME
    body = synthesise_proposal_metadata(pdf_path)
    assert body["programme"] == "horizon_2020"
    assert body["call_id"] == "H2020-SU-ICT-2019"
    assert body["topic_id"] == "952672"
    # Defaults still flow through for unspecified fields.
    assert body["outcome"] == "funded"
    assert body["year"] == 2024
    # Derived fields are present and correct.
    assert body["source_path"] == str(pdf_path)
    assert body["proposal_title"] == pdf_path.stem


def test_geiger_filename_logs_warning(tmp_path, caplog):
    pdf_path = tmp_path / GEIGER_NAME
    with caplog.at_level(logging.WARNING, logger="tests._helpers.metadata"):
        body = synthesise_proposal_metadata(pdf_path)
    # GEIGER has a topic_id but no programme alias → default programme
    # stays in place.
    assert body["programme"] == "horizon_europe"
    assert body["call_id"] == "UNSPECIFIED-CALL"
    assert body["topic_id"] == "883588"
    # Warning was emitted to flag the gap to the operator.
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Could not infer programme" in record.getMessage() for record in warnings)


def test_explicit_overrides_win_over_filename(tmp_path):
    pdf_path = tmp_path / SANCUS_NAME
    body = synthesise_proposal_metadata(
        pdf_path,
        programme="horizon_europe",
        call_id="HORIZON-CL3-2024-CS-01",
    )
    # The filename parsed ``horizon_2020`` but the caller's explicit
    # override must win.
    assert body["programme"] == "horizon_europe"
    assert body["call_id"] == "HORIZON-CL3-2024-CS-01"
    # The override does NOT touch topic_id, so the filename-derived
    # value is preserved.
    assert body["topic_id"] == "952672"


def test_sibling_yaml_wins_over_filename(tmp_path):
    pdf_path = tmp_path / SANCUS_NAME
    sibling = pdf_path.with_suffix(".yml")
    sibling.write_text(
        yaml.safe_dump(
            {
                "programme": "horizon_europe",
                "call_id": "HORIZON-CL3-2024-CS-01",
                "year": 2024,
                "outcome": "funded",
                "source_path": "ignored-by-helper",
            }
        ),
        encoding="utf-8",
    )
    body = load_or_synthesise_metadata(pdf_path)
    # YAML values win regardless of what the filename encodes.
    assert body["programme"] == "horizon_europe"
    assert body["call_id"] == "HORIZON-CL3-2024-CS-01"
    # source_path is rewritten to the absolute PDF path by the helper.
    assert body["source_path"] == str(pdf_path)


def test_unparseable_filename_retains_defaults(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    body = synthesise_proposal_metadata(pdf_path)
    assert body["programme"] == "horizon_europe"
    assert body["call_id"] == "UNSPECIFIED-CALL"
    assert "topic_id" not in body
