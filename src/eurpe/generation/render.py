"""Markdown rendering for generated drafts with visible source-status labels.

The renderer is the bridge between
:class:`~eurpe.generation.GenerationDraft` (the workflow's structured
output) and the human-readable Markdown a user pastes into a proposal
or shares with a co-author. Every citation in the draft must visibly
carry its :class:`~eurpe.schema.SourceStatus` so the reader can tell
funded patterns from rejected patterns from ESR advisory commentary
without having to chase footnotes.

What this module guarantees (Issue #7 acceptance criteria)
----------------------------------------------------------
* **AC1** — every ``[N]`` marker in the draft text gets a
  ``## References`` row that exposes the source document, section
  heading, and source-status label. Two complementary forms are
  produced for every render: a Markdown table (at-a-glance scan) and
  a per-citation notes block (carries the cautionary framing).
* **AC3** — :class:`~eurpe.schema.SourceStatus.FUNDED` and
  :class:`~eurpe.schema.SourceStatus.REJECTED` render with distinct
  visual badges *and* distinct textual caveats, so a reader can never
  mistake a cautionary lesson for a positive pattern.

Why two label dicts (LABEL + BADGE)
-----------------------------------
``STATUS_LABEL`` is the plain text label used inside table cells where
the column is already named ``Status``. ``STATUS_BADGE`` includes a
prefix glyph (``✓``, ``✗``, ``ⓘ``, ``?``) used in headings, references
list, and the per-citation notes — places where the label appears
inline with prose and benefits from a visual cue. Keeping them as
two named constants (rather than slicing a single combined string)
makes intent obvious at every call site and lets the audit module
look for the badge specifically when verifying rendered output.

PRD invariant: ESR notes
------------------------
ESR notes are external-reviewer commentary, NOT ground truth. The
``STATUS_CAVEAT`` for ESR explicitly says "advisory only" so the
reader cannot accidentally treat them as evidence. This is the same
framing the prompt builder already uses in ``eurpe.generation.prompt``.
"""

from __future__ import annotations

from eurpe.generation.models import CitationRef, GenerationDraft
from eurpe.schema import SectionType, SourceStatus

#: Plain text label per :class:`SourceStatus`. Used inside the Markdown
#: table where the column header already says "Status" so a glyph would
#: be visual noise. Centralised so a label change propagates to every
#: render path.
STATUS_LABEL: dict[SourceStatus, str] = {
    SourceStatus.FUNDED: "FUNDED",
    SourceStatus.REJECTED: "REJECTED",
    SourceStatus.ESR_NOTE: "ESR NOTE",
    SourceStatus.UNKNOWN: "UNKNOWN STATUS",
}

#: Visual badge per :class:`SourceStatus`. Used everywhere the label
#: appears inline with prose (per-citation notes, references list).
#: The leading glyph renders distinctly even in plain Markdown viewers
#: that strip color.
STATUS_BADGE: dict[SourceStatus, str] = {
    SourceStatus.FUNDED: "✓ FUNDED",
    SourceStatus.REJECTED: "✗ REJECTED",
    SourceStatus.ESR_NOTE: "ⓘ ESR ADVISORY",
    SourceStatus.UNKNOWN: "? UNKNOWN",
}

#: One-line caveat appended underneath the per-citation note for any
#: non-funded status. Funded gets an empty string so the renderer can
#: skip it without a special case. The caveats are intentionally
#: prose-style (italicised) so they read as commentary rather than
#: data; the table row already carries the machine-readable status.
STATUS_CAVEAT: dict[SourceStatus, str] = {
    SourceStatus.FUNDED: "",
    SourceStatus.REJECTED: "_Cautionary lesson — this proposal was NOT funded._",
    SourceStatus.ESR_NOTE: "_Reviewer commentary — advisory only, not ground truth._",
    SourceStatus.UNKNOWN: "_Source status unverified — treat with caution._",
}


def _section_title(section_type: SectionType) -> str:
    """``METHODOLOGY`` → ``"Methodology"``; ``IMPACT_PATHWAY`` → ``"Impact Pathway"``.

    Mirrors :meth:`eurpe.generation.prompt.SectionPromptBuilder._humanize_section_type`
    so the rendered draft heading matches the section title used in the
    prompt. Duplicated rather than imported to avoid coupling render
    output to the prompt builder's private helper.
    """

    return section_type.value.replace("_", " ").title()


def _humanize_programme(programme) -> str:  # type: ignore[no-untyped-def]
    """``HORIZON_EUROPE`` → ``"Horizon Europe"``.

    Same shape as the prompt builder's helper. Typed loosely so callers
    can hand a :class:`~eurpe.schema.Programme` member without an extra
    cast — the only operation we call is ``.value``.
    """

    return programme.value.replace("_", " ").title()


def _format_page(page: int | None) -> str:
    """Page placeholder so missing pagination doesn't render as ``None``."""

    return f"p. {page}" if page is not None else "p. n/a"


def _format_section(section: str | None) -> str:
    """Section heading placeholder."""

    return section if section else "n/a"


def _format_title(title: str | None) -> str:
    """Proposal title placeholder."""

    return title if title else "untitled"


def _escape_pipe(text: str) -> str:
    """Escape ``|`` so a snippet with a pipe doesn't break the Markdown table.

    Markdown tables use ``|`` as the column separator; an unescaped
    pipe in a cell silently splits it into two columns and the audit's
    column-width check would fail. Backslash-escaping is the convention
    most renderers honour.
    """

    return text.replace("|", "\\|")


class MarkdownCitationRenderer:
    """Render a :class:`GenerationDraft` to Markdown with source-status labels.

    Output structure:

    .. code-block:: markdown

        # {Section Title}

        {draft.text — inline [N] markers preserved verbatim}

        ## References

        | # | Status | Programme | Call | Section | Page | Source |
        |---|--------|-----------|------|---------|------|--------|
        | 1 | FUNDED | Horizon Europe | HORIZON-... | Methodology | 12 | (title) |

        ### Notes

        - [1] ✓ FUNDED — Horizon Europe call HORIZON-..., p. 12, §Methodology
        - [2] ✗ REJECTED — Horizon 2020 call ..., p. 8, §Implementation
              _Cautionary lesson — this proposal was NOT funded._

    The two-form rendering (table + per-citation notes) is intentional:
    the table is for at-a-glance scanning; the notes block carries the
    cautionary framing for non-funded sources. Both are mandatory in
    the output — :class:`~eurpe.generation.audit.CitationAudit` checks
    for the badge in the rendered Markdown to confirm the renderer
    didn't accidentally drop one.

    The renderer is stateless and deterministic: same draft → byte-equal
    output. Tests rely on that; do not introduce timestamp or order
    randomness in the rendered string.
    """

    def render(self, draft: GenerationDraft) -> str:
        """Render the draft as a complete Markdown document."""

        section_title = _section_title(draft.section_type)
        body = draft.text.rstrip()
        references_block = self._render_references(draft.citations)

        return (
            f"# {section_title}\n"
            "\n"
            f"{body}\n"
            "\n"
            "## References\n"
            "\n"
            f"{references_block}\n"
        )

    # ------------------------------------------------------------------
    # internal helpers — small, named, testable indirectly via render()
    # ------------------------------------------------------------------

    def _render_references(self, citations: list[CitationRef]) -> str:
        """Emit the ``## References`` body — table + ``### Notes`` block.

        Empty citation list renders as a single italicised line
        (``_No citations._``) instead of an empty table; this is the
        readable signal that the workflow ran but found no evidence
        worth quoting. The audit treats the empty case as legal.
        """

        if not citations:
            return "_No citations._"

        table = self._render_table(citations)
        notes = self._render_notes(citations)
        return f"{table}\n\n### Notes\n\n{notes}"

    @staticmethod
    def _render_table(citations: list[CitationRef]) -> str:
        """Render the at-a-glance Markdown table.

        Every row has the same number of columns (7) — the audit
        verifies this so a future refactor that drops a column doesn't
        ship silently.
        """

        header = "| # | Status | Programme | Call | Section | Page | Source |"
        separator = "|---|--------|-----------|------|---------|------|--------|"
        rows = [header, separator]
        for citation in citations:
            status_label = STATUS_LABEL[citation.source_status]
            programme = _humanize_programme(citation.programme)
            section = _format_section(citation.section_heading)
            page = (
                str(citation.page) if citation.page is not None else "n/a"
            )  # bare number in the table, "p." prefix is a Notes-block convention
            source = _format_title(citation.proposal_title)
            row = (
                f"| {citation.citation_id} "
                f"| {_escape_pipe(status_label)} "
                f"| {_escape_pipe(programme)} "
                f"| {_escape_pipe(citation.call_id)} "
                f"| {_escape_pipe(section)} "
                f"| {_escape_pipe(page)} "
                f"| {_escape_pipe(source)} |"
            )
            rows.append(row)
        return "\n".join(rows)

    @staticmethod
    def _render_notes(citations: list[CitationRef]) -> str:
        """Render the per-citation notes block with badges and caveats.

        Every citation gets one bullet:

            - [N] {BADGE} — Programme call CALL-ID, p. P, §Section

        Non-funded citations get an additional indented italicised line
        with the :data:`STATUS_CAVEAT` text so the cautionary framing
        cannot be missed even if the reader skims past the table.
        """

        bullets: list[str] = []
        for citation in citations:
            badge = STATUS_BADGE[citation.source_status]
            programme = _humanize_programme(citation.programme)
            section = _format_section(citation.section_heading)
            page = _format_page(citation.page)
            line = (
                f"- [{citation.citation_id}] {badge} — {programme} call "
                f"{citation.call_id}, {page}, §{section}"
            )
            bullets.append(line)
            caveat = STATUS_CAVEAT[citation.source_status]
            if caveat:
                # Two-space indent so the caveat visually attaches to the
                # bullet above without becoming a nested list item.
                bullets.append(f"  {caveat}")
        return "\n".join(bullets)
