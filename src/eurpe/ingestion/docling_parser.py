"""Docling-based proposal PDF parser.

Wraps :class:`docling.document_converter.DocumentConverter` to produce a
strongly-typed :class:`~eurpe.ingestion.models.ParsedProposal` from a single
PDF on disk.

Why this layer exists
---------------------
Docling's own document tree changes shape between minor releases (2.x has
already shifted ``iterate_items`` semantics and where ``num_pages`` lives).
EURPE freezes a *minimal contract* (title, ordered sections with depth,
tables as cell-text grids, page count) so the rest of the pipeline does not
break every time we bump Docling. If Docling's API moves again, only this
file changes.

Lazy import policy
------------------
``docling`` itself is heavy: pulling it in pulls accelerate / torch /
huggingface-hub / lxml / pillow / numpy and (on first ``parse()``) downloads
~40 MB of OCR model weights. We import it inside :meth:`parse` rather than
at module top so

* ``import eurpe.ingestion`` stays cheap (the CLI smoke command does not
  pay the cost),
* a developer running the model + error tests does not need the docling
  install at all (test_ingestion_models.py / test_ingestion_errors.py
  cover those cases without touching this parser).

Failure semantics
-----------------
Any exception raised by Docling is wrapped in
:class:`~eurpe.ingestion.errors.ParserError` carrying the source path and
the original exception via the ``cause`` attribute and the ``__cause__``
chain (raise … from …). The acceptance criterion "parser failures produce
clear errors without corrupting existing indexed data" is satisfied here
because :meth:`parse` is purely functional — it never writes to disk, so a
failure cannot leave partial state behind. Callers that DO write to disk
(see :mod:`eurpe.ingestion.cli`) are required to gate their writes on a
successful return from :meth:`parse`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from eurpe.ingestion.errors import ParserError, UnsupportedFormatError
from eurpe.ingestion.models import ParsedProposal, ParsedSection, ParsedTable

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from docling.document_converter import DocumentConverter


class DoclingProposalParser:
    """Parse proposal PDFs to :class:`ParsedProposal` using Docling.

    Acceptance-criteria coverage (issue #3):

    * Accepts a PDF and produces section-level output via :meth:`parse`.
    * Preserves title, headings, section boundaries, and table cell text.
    * Failures raise :class:`ParserError` and the parser writes nothing to
      disk, so partial state cannot leak into a downstream index.
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

    def __init__(self, *, ocr_enabled: bool = False) -> None:
        # ``ocr_enabled`` is recorded for future use; Docling's
        # ``DocumentConverter`` reads its own pipeline options from the
        # registered format options, so this prototype does not yet wire it
        # through. Kept on the constructor so the public API does not
        # change when the OCR toggle lands.
        self._ocr_enabled = ocr_enabled
        # Cache the converter across calls because constructing one is
        # surprisingly expensive (it imports torch and may trigger a model
        # download). Tests that need a fresh instance simply construct a
        # new ``DoclingProposalParser``.
        self._converter: DocumentConverter | None = None

    @property
    def ocr_enabled(self) -> bool:
        """Whether OCR was requested at construction time."""
        return self._ocr_enabled

    def supports(self, path: Path) -> bool:
        """Return ``True`` if ``path``'s extension is parseable.

        Matches case-insensitively so ``Report.PDF`` and ``report.pdf`` are
        treated identically (real-world proposals come with both casings).
        """

        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, path: Path) -> ParsedProposal:
        """Parse ``path`` and return a :class:`ParsedProposal`.

        Order of validation matters:

        1. ``UnsupportedFormatError`` if the extension is not in
           :attr:`SUPPORTED_EXTENSIONS`. Raised before touching the disk
           so callers can fan out across many files cheaply.
        2. ``ParserError`` if the file does not exist on disk.
        3. ``ParserError`` (with chained cause) if Docling itself raises
           or returns an empty document.
        """

        if not self.supports(path):
            raise UnsupportedFormatError(
                f"Unsupported file extension {path.suffix!r}; "
                f"supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )
        if not path.exists():
            raise ParserError(str(path), "file not found")

        # Lazy import — see module docstring "Lazy import policy". Wrapped
        # in its own try so an install-time problem (e.g., a torch wheel
        # mismatch) surfaces as a ``ParserError`` rather than an
        # unhandled ``ImportError`` deep in the CLI.
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ParserError(
                str(path),
                "docling is not installed; run `pip install -e .` to add it",
                cause=exc,
            ) from exc

        if self._converter is None:
            self._converter = DocumentConverter()

        try:
            result = self._converter.convert(str(path))
            doc = result.document
        except Exception as exc:  # noqa: BLE001 — Docling exposes many failure types
            # Wrap *any* Docling failure so the caller sees a single,
            # predictable exception type. Keep the original via __cause__
            # (raise … from …) plus the explicit ``cause`` attribute on
            # ParserError so programmatic handlers don't need to walk
            # __cause__.
            raise ParserError(str(path), f"docling.convert failed: {exc}", cause=exc) from exc

        try:
            return self._build_parsed_proposal(doc, path)
        except ParserError:
            raise
        except Exception as exc:  # noqa: BLE001 — defensive
            # Walking the document tree itself can raise if Docling's data
            # model shifts. Treat that as a parser failure rather than
            # crashing the caller.
            raise ParserError(
                str(path),
                f"failed to walk Docling document: {exc}",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_parsed_proposal(self, doc: Any, path: Path) -> ParsedProposal:
        """Convert a ``DoclingDocument`` to :class:`ParsedProposal`.

        The walk is deliberately *flat*: every ``SectionHeaderItem`` opens
        a new :class:`ParsedSection`, and every following ``TextItem`` at
        body tree-depth (1) appends to the current section's text. Items
        nested inside tables (tree-depth >= 2) are skipped here because
        they are recovered separately via ``doc.tables``. A hierarchical
        section tree is intentionally deferred to issue #4 (chunking).
        """

        sections: list[ParsedSection] = []
        current: ParsedSection | None = None
        # Buffer text lines for the current section so we can join with
        # newlines once at flush time (avoids repeated string concat).
        current_lines: list[str] = []

        title: str | None = None

        for item, tree_level in doc.iterate_items():
            type_name = type(item).__name__
            label = getattr(item, "label", None)
            label_value = getattr(label, "value", label)

            # The first TitleItem (if any) wins the title slot; otherwise
            # the first H1 SectionHeaderItem is a reasonable fallback.
            if title is None and type_name == "TitleItem":
                title = (getattr(item, "text", "") or "").strip() or None

            if type_name == "SectionHeaderItem":
                if title is None:
                    title = (getattr(item, "text", "") or "").strip() or None
                # Flush the previous section before opening a new one.
                if current is not None:
                    self._finalize_section(current, current_lines, sections)
                heading_text = (getattr(item, "text", "") or "").strip()
                if not heading_text:
                    # Skip empty headings rather than emit an invalid section.
                    current = None
                    current_lines = []
                    continue
                # Docling's ``SectionHeaderItem.level`` is 1-based and may
                # exceed our cap (6, matching HTML H1..H6). Clamp instead of
                # rejecting so an unusually-deep heading doesn't lose a
                # whole section worth of text.
                raw_level = int(getattr(item, "level", 1) or 1)
                clamped_level = max(1, min(6, raw_level))
                current = ParsedSection(
                    heading=heading_text,
                    level=clamped_level,
                    text="",
                    page_start=_first_page(item),
                    page_end=_first_page(item),
                )
                current_lines = []
                continue

            # Body text: append to the current section's buffer. Items
            # without a current section (i.e., body text before any
            # heading) are dropped on the floor for the prototype — that
            # leading content typically belongs to a cover page, not a
            # numbered section.
            if type_name == "TextItem" and tree_level == 1 and current is not None:
                text = (getattr(item, "text", "") or "").strip()
                if text:
                    current_lines.append(text)
                page = _first_page(item)
                if page is not None:
                    if current.page_start is None:
                        current.page_start = page
                    current.page_end = page

            # Suppress unused-variable warnings in branches that intentionally
            # ignore some labels (e.g., picture/list-group items).
            del label_value

        # Final flush.
        if current is not None:
            self._finalize_section(current, current_lines, sections)

        tables = self._extract_tables(doc)

        # Re-attach tables to their nearest preceding section (best-effort
        # via page number). If we cannot attach (no section on that page),
        # leave the table standalone in section ``__document__`` — but for
        # now we only attach to a known section to keep the model strict.
        if sections and tables:
            self._attach_tables_to_sections(sections, tables)

        page_count = _safe_num_pages(doc)
        # ``doc.name`` is Docling's document name, typically the file stem
        # — better than nothing if we never saw an explicit title item.
        if not title:
            doc_name = getattr(doc, "name", None)
            if isinstance(doc_name, str) and doc_name.strip():
                title = doc_name.strip()

        return ParsedProposal(
            source_path=str(path.resolve()),
            title=title,
            sections=sections,
            page_count=page_count,
            parser="docling",
        )

    def _finalize_section(
        self,
        section: ParsedSection,
        buffered_lines: list[str],
        sink: list[ParsedSection],
    ) -> None:
        """Join buffered text lines into the section and append to ``sink``.

        Mutating the section in-place is safe because ``ParsedSection``
        does not enable ``frozen=True`` and ``validate_assignment`` is off
        for this model — the join here is the single mutation we make.
        """

        section.text = "\n".join(buffered_lines).strip()
        sink.append(section)

    def _extract_tables(self, doc: Any) -> list[ParsedTable]:
        """Return one :class:`ParsedTable` per table in ``doc``.

        Uses ``TableItem.data.table_cells`` directly so we do not depend
        on pandas / dataframe export — Docling's dataframe path can fail
        if a table has merged cells. Cells are placed by their
        ``start_row_offset_idx`` / ``start_col_offset_idx`` into a 2D
        grid sized by ``num_rows`` / ``num_cols``.
        """

        out: list[ParsedTable] = []
        tables = getattr(doc, "tables", None) or []
        for tbl in tables:
            data = getattr(tbl, "data", None)
            if data is None:
                continue
            num_rows = int(getattr(data, "num_rows", 0) or 0)
            num_cols = int(getattr(data, "num_cols", 0) or 0)
            if num_rows <= 0 or num_cols <= 0:
                continue
            grid: list[list[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            for cell in getattr(data, "table_cells", []) or []:
                r = int(getattr(cell, "start_row_offset_idx", 0) or 0)
                c = int(getattr(cell, "start_col_offset_idx", 0) or 0)
                if 0 <= r < num_rows and 0 <= c < num_cols:
                    grid[r][c] = (getattr(cell, "text", "") or "").strip()
            out.append(
                ParsedTable(
                    section_heading=None,  # filled in by _attach_tables_to_sections
                    rows=grid,
                    page=_first_page(tbl),
                )
            )
        return out

    def _attach_tables_to_sections(
        self,
        sections: list[ParsedSection],
        tables: list[ParsedTable],
    ) -> None:
        """Attach each table to the closest preceding section by page.

        Best-effort: if a table's page falls within a section's
        [page_start, page_end] window, that section gets it; otherwise the
        table goes into the last section that started on or before the
        table's page. Tables with no page are appended to the last
        section.
        """

        for table in tables:
            target: ParsedSection | None = None
            if table.page is None:
                target = sections[-1]
            else:
                for sec in sections:
                    if (
                        sec.page_start is not None
                        and sec.page_end is not None
                        and sec.page_start <= table.page <= sec.page_end
                    ):
                        target = sec
                        break
                if target is None:
                    # Fall back to the last section whose page_start is on
                    # or before this table's page.
                    for sec in sections:
                        if sec.page_start is not None and sec.page_start <= table.page:
                            target = sec
                if target is None:
                    target = sections[-1]
            # Mutate the table's section_heading for traceability and
            # append. The Pydantic model has ``extra="forbid"`` and no
            # ``frozen=True``, so direct attribute assignment is fine.
            table.section_heading = target.heading
            target.tables.append(table)


def _first_page(item: Any) -> int | None:
    """Return the 1-indexed page of an item's first provenance record.

    Docling items carry a ``prov`` list of one or more ``ProvenanceItem``
    objects each with a ``page_no``. We take the first because EURPE only
    needs an approximate anchor at the prototype stage; full
    cross-page-span handling lands with chunking in issue #4.
    """

    prov = getattr(item, "prov", None) or []
    for p in prov:
        page_no = getattr(p, "page_no", None)
        if isinstance(page_no, int) and page_no >= 1:
            return page_no
    return None


def _safe_num_pages(doc: Any) -> int | None:
    """Return ``doc.num_pages()`` if callable, falling back to ``len(doc.pages)``.

    Docling has shipped both shapes across 2.x; rather than pin to one,
    we probe and return ``None`` if neither is available.
    """

    fn = getattr(doc, "num_pages", None)
    if callable(fn):
        try:
            value = fn()
        except Exception:  # noqa: BLE001 - probing
            value = None
        if isinstance(value, int):
            return value
    pages = getattr(doc, "pages", None)
    if pages is not None:
        try:
            return len(pages)
        except TypeError:
            return None
    return None
