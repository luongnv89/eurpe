"""Pure-string parser for proposal PDF filenames.

Real proposal filenames typically encode the EU programme, the call ID,
and the Funding & Tenders topic ID. The synthesised metadata helper
(:mod:`tests._helpers.metadata`) used to ignore the filename entirely and
hard-code defaults, which produced citation labels that contradicted the
underlying PDF (issue #47).

This module exposes a single function, :func:`parse_proposal_filename`,
that takes the bare filename and returns a dict containing only the keys
that were successfully parsed. It never raises and never reads the
filesystem. The merge contract is intentionally narrow so the caller
keeps full control over precedence (defaults / parsed / overrides /
sibling YAML).
"""

from __future__ import annotations

import re
from typing import Any

# Programme aliases observed in filenames map to canonical ``Programme``
# enum values. Lookup is case-insensitive (the token is uppercased before
# the dict lookup); emission is always the canonical lowercase form.
_PROGRAMME_ALIASES: dict[str, str] = {
    "H2020": "horizon_2020",
    "HORIZON-2020": "horizon_2020",
    "HORIZON_2020": "horizon_2020",
    "HE": "horizon_europe",
    "HORIZON-EUROPE": "horizon_europe",
    "HORIZON_EUROPE": "horizon_europe",
    "HORIZON-CL0": "horizon_europe",
    "HORIZON-CL1": "horizon_europe",
    "HORIZON-CL2": "horizon_europe",
    "HORIZON-CL3": "horizon_europe",
    "HORIZON-CL4": "horizon_europe",
    "HORIZON-CL5": "horizon_europe",
    "HORIZON-CL6": "horizon_europe",
    "HORIZON-CL7": "horizon_europe",
    "HORIZON-CL8": "horizon_europe",
    "HORIZON-CL9": "horizon_europe",
}

# Pattern for the programme token. ``(?<![A-Za-z0-9])`` and
# ``(?![A-Za-z0-9])`` are character-class lookarounds that allow ``_``,
# ``-``, ``.`` and string edges as separators while rejecting glued
# alphanumerics (``XH2020``, ``H20203``). Python's ``\b`` treats ``_``
# as a word character so it would fail on ``proposal_HE-...``.
_PROGRAMME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(HORIZON[-_](?:EUROPE|2020|CL\d)|H2020|HE)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Six- or seven-digit topic IDs (e.g., 952672 / 883588). Surrounded by
# non-digit boundaries so a four-digit year like 2024 cannot match and
# an eight-digit run does not slice into a spurious match.
_TOPIC_ID_PATTERN = re.compile(r"(?<!\d)(\d{6,7})(?!\d)")

# Tokens that frequently follow a call_id in real filenames; we strip
# them off the trailing edge of the greedy capture so call_id stays
# clean. Comparison is case-insensitive.
_CALL_ID_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "PART",
        "PARTB",
        "PART_B",
        "SECTION",
        "SEALED",
        "PROPOSAL",
        "SUBMITTED",
        "FINAL",
        "DRAFT",
        "ANNEX",
    }
)


def _canonical_programme(token: str) -> str | None:
    """Map a programme alias token to its canonical enum value."""

    return _PROGRAMME_ALIASES.get(token.upper())


def _extract_call_id(stem: str, programme_match: re.Match[str]) -> str | None:
    """Capture a call ID starting at the programme token.

    The strategy is greedy then trim: we walk forward from the programme
    token while we still see ``[A-Za-z0-9-]``, stopping at any other
    character (notably ``_``, the most common stem-internal separator).
    We then split on ``-`` and pop trailing tokens that are obvious
    junk (``PART``, ``SECTION``, ...). Finally we require at least one
    surviving ``-`` because real call IDs are dash-separated; a single
    bare token like ``H2020`` is not a call ID.
    """

    start = programme_match.start()
    # Walk forward consuming [A-Za-z0-9-] runs.
    end = start
    while end < len(stem) and (stem[end].isalnum() or stem[end] == "-"):
        end += 1
    raw = stem[start:end]

    # Strip trailing junk tokens.
    parts = raw.split("-")
    while parts and parts[-1].upper() in _CALL_ID_STOP_TOKENS:
        parts.pop()
    cleaned = "-".join(parts)

    # A real call ID has at least one dash separator.
    if "-" not in cleaned:
        return None
    return cleaned


def parse_proposal_filename(filename: str) -> dict[str, Any]:
    """Return a dict of metadata fields parseable from ``filename``.

    Only the keys that were successfully parsed are present in the
    returned dict. Possible keys:

    * ``programme`` — canonical :class:`Programme` enum value
      (``"horizon_2020"`` or ``"horizon_europe"``).
    * ``call_id`` — the dash-separated call identifier, e.g.,
      ``"H2020-SU-ICT-2019"`` or ``"HORIZON-CL3-2024-CS-01"``.
    * ``topic_id`` — six- or seven-digit topic number from the
      Funding & Tenders portal.

    Returns ``{}`` when nothing was parsed. Never raises.
    """

    # Strip the extension so PDFs and other suffixes do not bleed into
    # the parsed fields. ``Path`` would do this too but the caller may
    # already have just the stem.
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename

    parsed: dict[str, Any] = {}

    programme_match = _PROGRAMME_PATTERN.search(stem)
    if programme_match is not None:
        canonical = _canonical_programme(programme_match.group(1))
        if canonical is not None:
            parsed["programme"] = canonical
            call_id = _extract_call_id(stem, programme_match)
            if call_id is not None:
                parsed["call_id"] = call_id

    topic_match = _TOPIC_ID_PATTERN.search(stem)
    if topic_match is not None:
        parsed["topic_id"] = topic_match.group(1)

    return parsed
