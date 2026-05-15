"""Heading-anchored extractors for Work Programme call/topic excerpts.

Two entry points:

* :func:`extract_topic_context_from_text` turns a pasted plaintext
  excerpt into a :class:`TopicContext` using cheap regex parsing — no
  network, no LLM, no third-party dep.
* :func:`extract_topic_context_from_pdf` runs Docling in offline mode
  against a topic-page PDF excerpt, concatenates headings + body into
  one flat text, and delegates to the text path.

Why regex rather than a structured parser?

The Work Programme topic pages we target are remarkably regular:
``Topic:`` / ``Expected Outcomes`` / ``Scope`` / ``Destination`` are
verbatim headings the EU re-uses across programmes. A grammar-based
parser would be overkill (and brittle when the EU adds a new heading
variant). Regex with anchored heading captures is enough for the
prototype and obvious to maintain.

The programme / call_id / topic_id regexes are deliberately copied
from :mod:`tests._helpers.filename_parser` rather than imported. The
test helper is a test-only path; reproducing the constants in this
runtime module keeps the dependency direction one-way (production
code never imports from ``tests/``). The two copies must stay in
sync — they are tested with overlapping fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eurpe.intake.models import TopicContext, TopicSource
from eurpe.schema import Programme, SectionType

if TYPE_CHECKING:
    from eurpe.ingestion.docling_parser import DoclingProposalParser

# Programme aliases observed in call IDs / topic text map to canonical
# ``Programme`` enum values. Lookup is case-insensitive. Copied (not
# imported) from ``tests._helpers.filename_parser`` to keep runtime
# code independent of the test tree.
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

# Programme token: same character-class lookarounds as the filename
# parser so ``_``, ``-`` and ``.`` work as separators but glued
# alphanumerics don't match.
_PROGRAMME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(HORIZON[-_](?:EUROPE|2020|CL\d)|H2020|HE)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Six- or seven-digit topic IDs.
_TOPIC_ID_PATTERN = re.compile(r"(?<!\d)(\d{6,7})(?!\d)")

# Tokens that frequently follow a call_id on the trailing edge — stripped
# off the greedy capture so call_id stays clean. Matches the filename
# parser convention; "Resilient" and similar topic-title words are not
# in this list because the trailing-edge stripper only fires on
# dash-separated tokens.
_CALL_ID_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "PART",
        "PARTB",
        "SECTION",
        "SEALED",
        "PROPOSAL",
        "SUBMITTED",
        "FINAL",
        "DRAFT",
        "ANNEX",
    }
)

# Heading names that terminate a heading-anchored block. The Expected
# Outcomes / Scope captures stop when any of these is seen as a
# heading-line prefix. ``re.IGNORECASE`` is applied at match time.
_BLOCK_TERMINATORS: tuple[str, ...] = (
    "Scope",
    "Objective",
    "Objectives",
    "Specific Conditions",
    "Type of action",
    "Eligibility",
    "Destination",
    "Cluster",
    "Expected Outcomes",
    "Methodology guidance",
    "Impact guidance",
    "Excellence guidance",
    "Implementation guidance",
)

# Map a heading prefix → SectionType for ``section_guidance`` extraction.
# Only fires on verbatim matches (no fuzzy / synonym guesses) so an
# operator can predict what gets surfaced.
_SECTION_GUIDANCE_HEADINGS: dict[str, SectionType] = {
    "methodology guidance": SectionType.METHODOLOGY,
    "impact guidance": SectionType.IMPACT,
    "impact pathway guidance": SectionType.IMPACT_PATHWAY,
    "excellence guidance": SectionType.EXCELLENCE,
    "implementation guidance": SectionType.IMPLEMENTATION,
    "work plan guidance": SectionType.WORK_PLAN,
    "consortium guidance": SectionType.CONSORTIUM,
    "budget guidance": SectionType.BUDGET,
    "ethics guidance": SectionType.ETHICS,
    "dissemination guidance": SectionType.DISSEMINATION,
}


def _normalise(text: str) -> str:
    """Strip BOM, normalise line endings, collapse runs of >2 blank lines.

    The collapse keeps the input visually compact for downstream regex
    work without altering paragraph structure — two consecutive blank
    lines remain (so paragraph boundaries are preserved).
    """

    if not text:
        return ""
    # Strip BOM if present.
    if text.startswith("﻿"):
        text = text[1:]
    # CRLF → LF first; CR-only (old Mac) → LF.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines down to exactly 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _canonical_programme(token: str) -> Programme | None:
    """Map a programme alias token to its :class:`Programme` enum value."""

    canonical = _PROGRAMME_ALIASES.get(token.upper())
    if canonical is None:
        return None
    return Programme(canonical)


def _extract_call_id(text: str, programme_match: re.Match[str]) -> str | None:
    """Capture a call ID starting at the programme token.

    Walks forward consuming ``[A-Za-z0-9-]`` until the run ends, then
    strips trailing junk tokens (PART, SECTION, ...) from the
    dash-separated parts. A real call ID has at least one ``-``.
    Same shape as the filename parser's helper.
    """

    start = programme_match.start()
    end = start
    while end < len(text) and (text[end].isalnum() or text[end] == "-"):
        end += 1
    raw = text[start:end]

    parts = raw.split("-")
    while parts and parts[-1].upper() in _CALL_ID_STOP_TOKENS:
        parts.pop()
    cleaned = "-".join(parts)

    if "-" not in cleaned:
        return None
    return cleaned


def _parse_programme_call_topic(text: str) -> tuple[Programme | None, str | None, str | None]:
    """Recover ``(programme, call_id, topic_id)`` from raw text.

    Returns ``(None, None, None)`` when nothing matches. Each field is
    independent: a topic ID can be recovered even without a programme
    match.
    """

    programme: Programme | None = None
    call_id: str | None = None

    programme_match = _PROGRAMME_PATTERN.search(text)
    if programme_match is not None:
        programme = _canonical_programme(programme_match.group(1))
        if programme is not None:
            call_id = _extract_call_id(text, programme_match)

    topic_id: str | None = None
    topic_match = _TOPIC_ID_PATTERN.search(text)
    if topic_match is not None:
        topic_id = topic_match.group(1)

    return programme, call_id, topic_id


def _extract_labelled_line(text: str, labels: tuple[str, ...]) -> str | None:
    """Return the value following ``Label:`` on a single line, if any.

    Case-insensitive match on the label. The captured value is whatever
    follows the colon on the same line, stripped. Returns the first
    match for any label in ``labels`` (search order matters when two
    labels appear).
    """

    for label in labels:
        pattern = re.compile(
            rf"^[ \t]*{re.escape(label)}[ \t]*:[ \t]*(.+?)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        )
        m = pattern.search(text)
        if m is not None:
            value = m.group(1).strip()
            if value:
                return value
    return None


def _extract_block_after_heading(text: str, heading: str) -> str | None:
    """Return the block of text following a heading until the next heading.

    ``heading`` is matched case-insensitively on its own line, with an
    optional trailing colon. The block ends at the first occurrence of
    a heading line listed in :data:`_BLOCK_TERMINATORS` (also matched
    on its own line, optional trailing colon) or at end-of-text.

    Returns ``None`` if the heading was not found, or an empty string
    when the block is genuinely empty.
    """

    head_pattern = re.compile(
        rf"^[ \t]*{re.escape(heading)}[ \t]*:?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    head_match = head_pattern.search(text)
    if head_match is None:
        return None
    start = head_match.end()

    # Find the next heading-line terminator after ``start``.
    terminator_pattern = re.compile(
        r"^[ \t]*(?:" + "|".join(re.escape(t) for t in _BLOCK_TERMINATORS) + r")[ \t]*:?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    end = len(text)
    for m in terminator_pattern.finditer(text, pos=start):
        # Skip the heading we just matched (the terminator list includes
        # the heading itself).
        if m.start() == head_match.start():
            continue
        end = m.start()
        break

    return text[start:end].strip()


def _split_bullets(block: str) -> list[str]:
    """Split a block into bullets.

    Lines starting with ``-``, ``*``, ``•``, or ``\\d+\\.`` are treated
    as individual bullets. If no bullet markers are present, the block
    is split on blank lines (one bullet per paragraph). Empty bullets
    are dropped. Each returned bullet has its leading marker stripped
    and surrounding whitespace trimmed.
    """

    if not block:
        return []

    lines = block.splitlines()
    bullet_marker = re.compile(r"^[ \t]*(?:[-*•]|\d+\.)\s+")
    bullets: list[str] = []
    current: list[str] = []

    has_marker = any(bullet_marker.match(line) for line in lines)
    if has_marker:
        for line in lines:
            m = bullet_marker.match(line)
            if m is not None:
                # Flush the previous bullet, if any, before starting a new one.
                if current:
                    joined = " ".join(part.strip() for part in current).strip()
                    if joined:
                        bullets.append(joined)
                    current = []
                current.append(line[m.end():])
            elif line.strip():
                # Continuation of the current bullet.
                if current:
                    current.append(line)
        # Flush the trailing bullet.
        if current:
            joined = " ".join(part.strip() for part in current).strip()
            if joined:
                bullets.append(joined)
        return [b for b in bullets if b]

    # No marker — split on blank lines into paragraphs.
    paragraphs = re.split(r"\n\s*\n", block)
    return [p.strip() for p in paragraphs if p.strip()]


def _extract_section_guidance(text: str) -> dict[SectionType, str]:
    """Capture ``<Section> guidance:`` blocks keyed by :class:`SectionType`.

    Only the headings explicitly listed in
    :data:`_SECTION_GUIDANCE_HEADINGS` are recognised. The captured
    block stops at the next heading in :data:`_BLOCK_TERMINATORS`.
    """

    out: dict[SectionType, str] = {}
    for heading, section_type in _SECTION_GUIDANCE_HEADINGS.items():
        block = _extract_block_after_heading(text, heading)
        if block:
            out[section_type] = block
    return out


def extract_topic_context_from_text(
    text: str,
    *,
    source_path: str | None = None,
) -> TopicContext:
    """Parse a pasted Work Programme topic excerpt into :class:`TopicContext`.

    Best-effort: any field the regexes cannot recover stays at its
    default. Empty input yields a :class:`TopicContext` with all-empty
    fields and ``source=TopicSource.PASTED_TEXT``.

    When ``source_path`` is provided the record is tagged with
    ``source=TopicSource.PDF_EXCERPT`` and ``source_path`` is recorded
    verbatim — this lets :func:`extract_topic_context_from_pdf` reuse
    this function without a separate model-construction path.
    """

    normalised = _normalise(text)

    programme, call_id, topic_id = _parse_programme_call_topic(normalised)

    topic_title = _extract_labelled_line(normalised, ("Topic title", "Topic"))

    outcomes_block = _extract_block_after_heading(normalised, "Expected Outcomes")
    expected_outcomes = _split_bullets(outcomes_block) if outcomes_block else []

    scope_block = _extract_block_after_heading(normalised, "Scope")
    scope: str | None = None
    if scope_block:
        # Collapse internal whitespace runs so the rendered prompt stays
        # tidy, but preserve paragraph breaks.
        scope_paragraphs = [
            " ".join(p.split()) for p in re.split(r"\n\s*\n", scope_block) if p.strip()
        ]
        scope = "\n\n".join(scope_paragraphs) or None

    destination = _extract_labelled_line(normalised, ("Destination", "Cluster"))

    section_guidance = _extract_section_guidance(normalised)

    if source_path is None:
        source = TopicSource.PASTED_TEXT
        path_field: str | None = None
    else:
        source = TopicSource.PDF_EXCERPT
        path_field = str(source_path)

    return TopicContext(
        programme=programme,
        call_id=call_id,
        topic_id=topic_id,
        topic_title=topic_title,
        expected_outcomes=expected_outcomes,
        scope=scope,
        destination=destination,
        section_guidance=section_guidance,
        raw_text=normalised,
        source=source,
        source_path=path_field,
    )


def extract_topic_context_from_pdf(
    pdf_path: Path,
    *,
    parser: Any = None,
) -> TopicContext:
    """Parse a topic-page PDF excerpt into :class:`TopicContext`.

    Uses :class:`~eurpe.ingestion.docling_parser.DoclingProposalParser`
    in offline mode by default. The parsed sections are concatenated
    into a flat text blob (``"<heading>\\n<text>"`` per section, joined
    by blank lines, with the title prepended if present) and handed to
    :func:`extract_topic_context_from_text` for the actual field
    extraction.

    ``parser`` accepts any object exposing a ``parse(Path) -> ParsedProposal``
    method so tests can stub the Docling round-trip with a fake.

    The lazy import means a fast test (one that exercises the
    text-only extractor) does not pull Docling into the import graph.
    """

    if parser is None:
        from eurpe.ingestion.docling_parser import DoclingProposalParser

        parser = DoclingProposalParser(offline=True)

    parsed = parser.parse(pdf_path)

    text_parts: list[str] = []
    if parsed.title:
        text_parts.append(parsed.title)
    for section in parsed.sections:
        text_parts.append(f"{section.heading}\n{section.text}")
    flat_text = "\n\n".join(text_parts)

    return extract_topic_context_from_text(flat_text, source_path=str(pdf_path))
