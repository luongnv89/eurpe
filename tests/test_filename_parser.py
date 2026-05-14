"""Tests for ``tests._helpers.filename_parser``.

The parser is a pure-string function — no filesystem access, no
side-effects — so the tests are simple input/output pairs that cover
the real filename shapes the project carries today (SANCUS H2020,
GEIGER topic-only) plus representative Horizon Europe call IDs.

Two invariants are guarded:

* The function never raises.
* It only emits keys that were parsed successfully — a missing programme
  alias must NOT silently invent a default; the caller decides.
"""

from __future__ import annotations

from tests._helpers.filename_parser import parse_proposal_filename


def test_sancus_filename_parses_h2020() -> None:
    result = parse_proposal_filename(
        "SANCUS_PROPOSAL_952672-SANCUS-H2020-SU-ICT-2019-PART_B_Section_1.pdf"
    )
    assert result["programme"] == "horizon_2020"
    assert result["call_id"] == "H2020-SU-ICT-2019"
    assert result["topic_id"] == "952672"
    # PART_B_Section_1 must not bleed into call_id.
    assert "PART" not in result["call_id"]
    assert "Section" not in result["call_id"]


def test_geiger_filename_only_topic_id() -> None:
    result = parse_proposal_filename("GEIGER_883588--SEALED-PROPOSAL.pdf")
    assert result == {"topic_id": "883588"}
    assert "programme" not in result
    assert "call_id" not in result


def test_horizon_europe_call_id() -> None:
    result = parse_proposal_filename("Foo_HORIZON-CL3-2024-CS-01_PartB.pdf")
    assert result["programme"] == "horizon_europe"
    assert result["call_id"] == "HORIZON-CL3-2024-CS-01"
    assert "topic_id" not in result


def test_horizon_europe_he_token() -> None:
    result = parse_proposal_filename("proposal_HE-2024-XYZ-01.pdf")
    assert result["programme"] == "horizon_europe"


def test_case_insensitive() -> None:
    result = parse_proposal_filename("foo_h2020-su-ict-2019_bar.pdf")
    assert result["programme"] == "horizon_2020"
    assert "call_id" in result
    # Case may be preserved from the input; assert content match
    # case-insensitively per the plan.
    assert "SU-ICT-2019" in result["call_id"].upper()


def test_unparseable_filename_returns_empty_dict() -> None:
    assert parse_proposal_filename("report.pdf") == {}


def test_year_not_matched_as_topic_id() -> None:
    # A bare four-digit year must NOT be picked up by the topic_id regex.
    result = parse_proposal_filename("annual_review_2019.pdf")
    assert "topic_id" not in result


def test_no_dash_no_call_id() -> None:
    # The programme token appears but there is no dash-separated suffix
    # → no call_id should be emitted (defensive guard from the plan).
    result = parse_proposal_filename("project_H2020.pdf")
    assert result.get("programme") == "horizon_2020"
    assert "call_id" not in result


def test_eight_digit_run_not_matched_as_topic_id() -> None:
    # The lookarounds reject sequences glued to other digits.
    result = parse_proposal_filename("dump_12345678.pdf")
    assert "topic_id" not in result


def test_seven_digit_topic_id_matches() -> None:
    result = parse_proposal_filename("foo_1234567_bar.pdf")
    assert result["topic_id"] == "1234567"


def test_does_not_raise_on_empty_string() -> None:
    # Defensive: the parser must never raise even on degenerate input.
    assert parse_proposal_filename("") == {}


def test_does_not_raise_on_extension_only() -> None:
    assert parse_proposal_filename(".pdf") == {}


def test_horizon_2020_underscore_alias() -> None:
    result = parse_proposal_filename("foo_HORIZON_2020-SU-ICT-2019.pdf")
    assert result["programme"] == "horizon_2020"


def test_horizon_europe_dash_alias() -> None:
    result = parse_proposal_filename("foo_HORIZON-EUROPE-CL5-2024-D3-02.pdf")
    assert result["programme"] == "horizon_europe"


def test_glued_alphanumeric_rejects_match() -> None:
    # ``XH2020`` should not match the H2020 alias because the lookbehind
    # rejects an adjacent alphanumeric.
    result = parse_proposal_filename("project_XH20203.pdf")
    assert "programme" not in result
