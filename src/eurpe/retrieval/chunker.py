"""Hierarchical chunker that turns a parsed proposal into retrieval-ready chunks.

The chunker is the join point between two layers:

* **Structural** — :class:`~eurpe.ingestion.models.ParsedProposal`
  (headings, body text, tables, page spans) produced by the Docling
  parser. Knows nothing about programme/call/outcome.
* **Provenance** — :class:`~eurpe.schema.ProposalMetadata` from the
  YAML sidecar (programme, call_id, outcome, source_path).

The chunker walks the structural tree section-by-section and emits
:class:`~eurpe.retrieval.models.Chunk` records that carry both layers
joined together via :class:`~eurpe.schema.ChunkMetadata`. The drift
validator on ``ChunkMetadata`` enforces that every chunk's
``source_status`` equals its proposal's ``outcome`` — we set both from
the same value so the invariant holds by construction.

Splitting strategy
------------------
Within a section the splitter is sentence-aware: it prefers to break
at ``. ``, ``! ``, ``? ``, or a blank line, falling back to a hard
character cut at ``target_chars + tolerance`` if no boundary appears
in the search window. Consecutive chunks share an ``overlap_chars``
suffix/prefix so a query that straddles a boundary can still match.
A trailing chunk shorter than ``min_chunk_chars`` is merged into its
predecessor so the index does not fill with stub fragments — those
hurt retrieval quality and waste vector slots.

Tables are emitted as their own chunks (one chunk per
:class:`~eurpe.ingestion.models.ParsedTable`). Their text is the cell
grid joined with ``" | "`` per row and ``"\\n"`` between rows. The
parent section's heading and section-type tag travel with the table
chunk so a query that hits the table still resolves a meaningful
location for the citation footer.

Section-type inference
----------------------
:func:`infer_section_type` does a lowercase substring match against
the section heading. ``IMPACT_PATHWAY`` is checked before ``IMPACT``
because every "impact pathway" heading also contains "impact" — order
matters in the lookup table.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from eurpe.ingestion.models import ParsedProposal, ParsedSection, ParsedTable
from eurpe.retrieval.models import Chunk
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    ProposalMetadata,
    SectionType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section-type inference
# ---------------------------------------------------------------------------

# Order matters: more-specific patterns must come BEFORE less-specific
# ones because the lookup uses ``in`` (substring) — "impact pathway"
# also contains "impact". A list of (needle, section_type) pairs keeps
# the precedence explicit, which a dict cannot do.
_SECTION_TYPE_RULES: list[tuple[str, SectionType]] = [
    ("impact pathway", SectionType.IMPACT_PATHWAY),
    ("excellence", SectionType.EXCELLENCE),
    ("methodology", SectionType.METHODOLOGY),
    ("work plan", SectionType.WORK_PLAN),
    ("workplan", SectionType.WORK_PLAN),
    ("implementation", SectionType.IMPLEMENTATION),
    ("consortium", SectionType.CONSORTIUM),
    ("partners", SectionType.CONSORTIUM),
    ("budget", SectionType.BUDGET),
    ("resources", SectionType.BUDGET),
    ("ethics", SectionType.ETHICS),
    ("dissemination", SectionType.DISSEMINATION),
    ("exploitation", SectionType.DISSEMINATION),
    # IMPACT comes last among the impact-family patterns so
    # "impact pathway" wins above. Pure "impact" still resolves
    # correctly because all the more-specific rules will have
    # missed first.
    ("impact", SectionType.IMPACT),
]


def infer_section_type(heading: str) -> SectionType:
    """Best-effort map a section heading to a :class:`SectionType`.

    Case-insensitive substring match. Returns :attr:`SectionType.OTHER`
    for headings that match no rule (annexes, appendices, prefatory
    material). Pure function — no side effects, deterministic — so it
    is trivially testable with parametrize.
    """

    if not heading:
        return SectionType.OTHER
    needle_text = heading.lower()
    for needle, section_type in _SECTION_TYPE_RULES:
        if needle in needle_text:
            return section_type
    return SectionType.OTHER


# ---------------------------------------------------------------------------
# Splitter helpers
# ---------------------------------------------------------------------------

# Sentence-end markers we look for, in priority order. ``"\n\n"`` (a
# blank line / paragraph break) is preferred because it is unambiguously
# a sentence boundary; the others are weaker signals.
_BREAK_TOKENS: tuple[str, ...] = ("\n\n", ". ", "? ", "! ")


def _find_split(text: str, target: int, tolerance: int) -> int:
    """Return an index near ``target`` that lands on a sentence boundary.

    Searches the window ``[target - tolerance, target + tolerance]`` for
    any of :data:`_BREAK_TOKENS` (in priority order) and returns the
    *end* of the matched token (so the chunk includes the period or the
    blank line). Falls back to ``target`` if no boundary appears in
    the window — a hard char-cut is preferable to letting chunks grow
    unbounded.
    """

    if target >= len(text):
        return len(text)
    lo = max(0, target - tolerance)
    hi = min(len(text), target + tolerance)
    window = text[lo:hi]
    for marker in _BREAK_TOKENS:
        # Prefer a break that is closest to ``target``. ``rfind`` from the
        # end of the window gives us the latest opportunity inside the
        # tolerance window, which keeps chunks closer to ``target`` size.
        rel = window.rfind(marker)
        if rel >= 0:
            return lo + rel + len(marker)
    return target


def _split_text(
    text: str,
    *,
    target_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
    tolerance: int = 100,
) -> list[tuple[int, int]]:
    """Return a list of ``(start, end)`` character offsets into ``text``.

    The returned spans cover the whole text in order with the
    configured overlap. The final span may be merged with its
    predecessor if it would otherwise be shorter than
    ``min_chunk_chars`` — that's the rule that keeps a 50-char tail
    from polluting the index with a stub chunk.
    """

    n = len(text)
    if n == 0:
        return []
    if n <= target_chars:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < n:
        # If the remaining text fits in one chunk, take it whole.
        if n - cursor <= target_chars:
            spans.append((cursor, n))
            break
        proposed_end = cursor + target_chars
        end = _find_split(text, proposed_end, tolerance)
        # Guard against pathological input that would emit a zero-length
        # chunk (e.g., the splitter returned ``cursor`` because nothing
        # changed). Force progress.
        if end <= cursor:
            end = min(cursor + target_chars, n)
        spans.append((cursor, end))
        # Advance with overlap; ``max(... , cursor + 1)`` is the
        # belt-and-suspenders progress guarantee.
        cursor = max(end - overlap_chars, cursor + 1)

    # Merge a too-small tail into its predecessor so we don't emit
    # stubby chunks. ``min_chunk_chars`` is the only knob that controls
    # this behaviour; setting it to 0 would skip the merge entirely.
    if len(spans) >= 2:
        last_start, last_end = spans[-1]
        if last_end - last_start < min_chunk_chars:
            prev_start, _prev_end = spans[-2]
            spans[-2] = (prev_start, last_end)
            spans.pop()
    return spans


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class HierarchicalChunker:
    """Split a :class:`ParsedProposal` into :class:`Chunk` records.

    Per-section pass:

    1. Infer the section's :class:`SectionType` from its heading.
    2. Split the section body via :func:`_split_text` and emit one
       :class:`Chunk` per span. Char offsets are relative to the
       section's text — the chunker does not currently reconstruct
       offsets relative to the whole document because the parser does
       not expose them. (See ``ParsedSection`` docstring: "char offsets
       are intentionally not modelled here. They will be added when
       chunking lands (Issue #4)" — at the section level we DO expose
       them on each chunk via :class:`CitationAnchor`.)
    3. Emit one :class:`Chunk` per :class:`ParsedTable` in the section.
       The table chunk's ``parent_section_heading`` mirrors the section
       it sits in so a query that hits the table can cite its origin.

    The ``chunk_index`` is a global counter across the whole
    proposal, not per-section, so :attr:`Chunk.chunk_id` produces
    distinct ids for two chunks that happen to share a section
    heading hash.
    """

    def __init__(
        self,
        *,
        target_chars: int = 1200,
        overlap_chars: int = 200,
        min_chunk_chars: int = 200,
    ) -> None:
        if target_chars <= 0:
            raise ValueError(f"target_chars must be positive, got {target_chars}")
        if overlap_chars < 0:
            raise ValueError(f"overlap_chars must be non-negative, got {overlap_chars}")
        if overlap_chars >= target_chars:
            # Would never advance the cursor: each chunk would start
            # before its predecessor ended.
            raise ValueError(
                f"overlap_chars ({overlap_chars}) must be < target_chars ({target_chars})"
            )
        if min_chunk_chars < 0:
            raise ValueError(f"min_chunk_chars must be non-negative, got {min_chunk_chars}")
        self._target_chars = target_chars
        self._overlap_chars = overlap_chars
        self._min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def chunk(self, parsed: ParsedProposal, proposal: ProposalMetadata) -> list[Chunk]:
        """Return all chunks for ``parsed``, joined with ``proposal``.

        The ``proposal.outcome`` value drives the chunk's
        ``source_status`` — the drift validator on
        :class:`ChunkMetadata` enforces equality. Setting both fields
        from the same source guarantees the invariant.
        """

        document_id = _document_id_for(parsed)
        out: list[Chunk] = []
        chunk_index = 0
        for section in parsed.sections:
            for chunk in self._chunk_section(
                section,
                proposal=proposal,
                document_id=document_id,
                start_index=chunk_index,
            ):
                out.append(chunk)
                chunk_index += 1
        return out

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _chunk_section(
        self,
        section: ParsedSection,
        *,
        proposal: ProposalMetadata,
        document_id: str,
        start_index: int,
    ) -> Iterable[Chunk]:
        """Yield text chunks then table chunks for one section."""

        section_type = infer_section_type(section.heading)
        running_index = start_index

        # Body text first.
        spans = _split_text(
            section.text,
            target_chars=self._target_chars,
            overlap_chars=self._overlap_chars,
            min_chunk_chars=self._min_chunk_chars,
        )
        for char_start, char_end in spans:
            text_slice = section.text[char_start:char_end]
            # ``str.strip`` would change the offsets we record; instead
            # we skip empties (a space-only chunk would be useless).
            if not text_slice.strip():
                continue
            anchor = CitationAnchor(
                document_id=document_id,
                section_heading=section.heading,
                page=section.page_start,
                char_start=char_start,
                char_end=char_end,
            )
            yield Chunk(
                text=text_slice,
                metadata=ChunkMetadata(
                    proposal=proposal,
                    section_type=section_type,
                    parent_section_heading=section.heading,
                    chunk_index=running_index,
                    anchor=anchor,
                    source_status=proposal.outcome,
                ),
            )
            running_index += 1

        # Tables next, one chunk each.
        for table in section.tables:
            table_text = _format_table(table)
            if not table_text.strip():
                continue
            anchor = CitationAnchor(
                document_id=document_id,
                section_heading=section.heading,
                page=table.page or section.page_start,
                # Table cells live outside the body text stream, so
                # char offsets aren't meaningful for them — leave None
                # rather than fabricate a span.
                char_start=None,
                char_end=None,
            )
            yield Chunk(
                text=table_text,
                metadata=ChunkMetadata(
                    proposal=proposal,
                    section_type=section_type,
                    parent_section_heading=section.heading,
                    chunk_index=running_index,
                    anchor=anchor,
                    source_status=proposal.outcome,
                ),
            )
            running_index += 1


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------


def _format_table(table: ParsedTable) -> str:
    """Render a :class:`ParsedTable` as plain text for embedding.

    Rows joined with newlines, cells with ``" | "``. Empty cells are
    preserved as empty strings between separators so a downstream
    reader can spot the column structure even after the chunk has
    been embedded and retrieved.
    """

    return "\n".join(" | ".join(row) for row in table.rows)


def _document_id_for(parsed: ParsedProposal) -> str:
    """Pick a stable document id for a parsed proposal.

    Prefers the file *stem* (``my_proposal`` from
    ``/abs/path/my_proposal.pdf``) because that's the human-friendly
    handle most operators recognise from on-disk file listings.
    Falls back to the literal ``source_path`` when the stem is empty
    (e.g., a path of ``"."`` or ``"/"``).
    """

    from pathlib import Path

    stem = Path(parsed.source_path).stem
    return stem or parsed.source_path
