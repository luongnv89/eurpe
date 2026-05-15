"""Prompt construction for the section-generation workflow.

Turns a :class:`~eurpe.generation.GenerationRequest` plus a list of
:class:`~eurpe.retrieval.RetrievalResult` records into a single string
prompt for the LLM, alongside the structured
:class:`~eurpe.generation.CitationRef` list that the workflow attaches
to the resulting draft.

Why all the citation work happens here
--------------------------------------
The 1:1 mapping between ``[N]`` markers in the prompt and entries in
the citation list is what lets the workflow validate the LLM's output
("did the model hallucinate citation [99]?"). Building both in lockstep
in one place keeps the contract auditable: if you read the prompt, you
know exactly which citations the workflow will accept in the response.

Source-status framing
---------------------
The prompt explicitly distinguishes ``**FUNDED**`` (positive patterns),
``**REJECTED**`` (cautionary), ``**ESR notes**`` (advisory commentary),
and ``**UNKNOWN**`` evidence. This is not cosmetic — it is the
mechanism by which the system delivers PRD § "Source-status labelling"
end-to-end. The "Output instructions" section also tells the model
how to treat each: cite funded as supporting evidence, frame rejected
as cautionary lessons, and never cite ESR notes as ground truth.

Stable citation-line format
---------------------------
Each evidence entry is rendered on a single line that starts with
``[N] **STATUS**`` so :class:`~eurpe.generation.llm.DeterministicLLMClient`
can locate the markers with a stable regex
(:data:`eurpe.generation.llm._PROMPT_CITATION_LINE`). Changing the
format here without updating the regex will break the deterministic
test path immediately — that breakage is intentional, the format is
contract.
"""

from __future__ import annotations

from eurpe.generation.models import CitationRef, GenerationRequest
from eurpe.generation.profiles import DraftingProfile
from eurpe.intake.models import TopicContext
from eurpe.retrieval import RetrievalResult
from eurpe.schema import Programme, SectionType, SourceStatus

#: Cap on the snippet length included in the prompt's evidence list and in the
#: :class:`CitationRef.snippet` field. 300 chars is enough to convey the gist
#: of a passage without bloating the prompt or the citation list.
_SNIPPET_MAX_CHARS = 300


#: Section-type-specific guidance injected into the prompt's "## Section
#: guidance" block. Each value names the structural elements a Horizon Europe
#: / Horizon 2020 section is expected to cover. Sourced from the EU's own
#: standard application form templates and the PRD's section-type vocabulary.
SECTION_GUIDANCE: dict[SectionType, str] = {
    SectionType.METHODOLOGY: (
        "Describe the technical approach: methods, models, datasets, "
        "validation strategy, and risk mitigation."
    ),
    SectionType.IMPACT_PATHWAY: (
        "Articulate the impact pathway: from outputs to outcomes to wider "
        "impacts. Include scientific, economic, and societal dimensions, "
        "with measurable indicators."
    ),
    SectionType.IMPACT: (
        "Describe the expected impacts: scientific, economic, societal, "
        "and policy. Identify target stakeholders and quantify expected "
        "benefits where possible."
    ),
    SectionType.EXCELLENCE: (
        "Frame ambition and credibility: state of the art, research "
        "questions, novelty, soundness of methodology, interdisciplinary "
        "fit."
    ),
    SectionType.IMPLEMENTATION: (
        "Describe the work plan structure: WPs, tasks, milestones, "
        "deliverables, risk register, governance, and consortium fit."
    ),
    SectionType.WORK_PLAN: (
        "Lay out the WP/task structure with effort allocation, milestones, "
        "and a brief Gantt summary. Include dependencies between tasks."
    ),
    SectionType.CONSORTIUM: (
        "Describe the consortium composition: each partner's role, "
        "complementarity, and added value."
    ),
    SectionType.BUDGET: (
        "Justify the resource allocation: personnel costs, equipment, "
        "subcontracting, travel, with rationale per WP."
    ),
    SectionType.ETHICS: (
        "Address ethical considerations: data protection, informed "
        "consent, dual use, environmental impact."
    ),
    SectionType.DISSEMINATION: (
        "Plan dissemination, exploitation, and communication: target "
        "audiences, channels, IPR strategy."
    ),
    SectionType.OTHER: ("Provide a clear, well-structured contribution to this section."),
}


#: Human-readable labels for each :class:`SourceStatus`, used in the prompt's
#: evidence list. Centralised so changing one label updates every render path.
_STATUS_LABEL: dict[SourceStatus, str] = {
    SourceStatus.FUNDED: "FUNDED",
    SourceStatus.REJECTED: "REJECTED",
    SourceStatus.ESR_NOTE: "ESR NOTE",
    SourceStatus.UNKNOWN: "UNKNOWN",
}


def _truncate_snippet(text: str, *, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    """Single-line snippet with whitespace collapsed and an ellipsis if cut.

    Mirrors the formatting convention of
    :func:`eurpe.retrieval.cli._format_snippet` so a snippet looks the
    same in the prompt and in CLI query output.
    """

    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


class SectionPromptBuilder:
    """Build an LLM prompt + structured citation list from a request and retrieval results.

    The resulting prompt has a stable, documented structure (see the
    module docstring) so:

    * :class:`~eurpe.generation.llm.DeterministicLLMClient` can parse
      the citation lines deterministically.
    * The :class:`~eurpe.generation.workflow.SectionGenerationWorkflow`
      can validate the LLM's response against the same citation list.
    * Tests can assert on specific phrases (e.g., "Do not invent",
      "**FUNDED**") without depending on the LLM's behaviour.

    The class is stateless — same inputs → same outputs — so a single
    instance can be shared across calls and threads.

    Drafting profiles
    -----------------
    If a :class:`~eurpe.generation.profiles.DraftingProfile` is provided,
    the builder uses programme-specific section guidance and expected
    outputs. The profile is passed per-call rather than at construction
    time so a single builder instance can serve multiple programmes.
    """

    def build(
        self,
        request: GenerationRequest,
        results: list[RetrievalResult],
        *,
        profile: DraftingProfile | None = None,
    ) -> tuple[str, list[CitationRef]]:
        """Return ``(prompt_text, citations)`` for the given request and results.

        Citations are 1-indexed in retrieval order: the first
        :class:`RetrievalResult` becomes ``[1]``, the second ``[2]``,
        and so on. Order is preserved through the prompt and the
        returned list, so a citation's ``citation_id`` directly indexes
        the list with ``citations[id - 1]``.

        If ``profile`` is provided, programme-specific section guidance
        and expected outputs are used. Otherwise, the default
        :data:`SECTION_GUIDANCE` is used.
        """

        citations = self._build_citations(results)

        # Use profile-specific guidance if available, otherwise fall back to default.
        if profile is not None:
            profile_guidance = profile.get_section_guidance(request.section_type)
            guidance = profile_guidance or SECTION_GUIDANCE.get(
                request.section_type, SECTION_GUIDANCE[SectionType.OTHER]
            )
            expected_outputs = profile.get_expected_outputs(request.section_type)
        else:
            guidance = SECTION_GUIDANCE.get(
                request.section_type, SECTION_GUIDANCE[SectionType.OTHER]
            )
            expected_outputs = []

        # When the supplied TopicContext carries a per-section guidance
        # entry for this request's section, append it to the existing
        # (profile / default) guidance under a labelled paragraph. The
        # prefix is part of the prompt contract — pinned by the
        # ``test_prompt_renders_topic_section_guidance_under_section_guidance_block``
        # test.
        topic_context = request.topic_context
        topic_section_guidance: str | None = None
        if topic_context is not None:
            topic_section_guidance = topic_context.section_guidance.get(request.section_type)

        section_title = self._humanize_section_type(request.section_type)
        programme_label = (
            self._humanize_programme(request.target_programme)
            if request.target_programme is not None
            else "Any (no programme filter)"
        )
        evidence_block = self._format_evidence(results, citations)
        topic_context_block = self._format_topic_context_block(topic_context, request.call_context)

        # Build the section-guidance block, appending the per-section
        # topic guidance paragraph when present.
        guidance_block = f"{guidance}\n"
        if topic_section_guidance:
            guidance_block += (
                f"\n**Topic requirements for this section:** {topic_section_guidance}\n"
            )

        prompt_parts = [
            "# EU Proposal Section Drafting Task\n",
            "\n",
            f"**Section type:** {section_title}\n",
            f"**Programme:** {programme_label}\n",
            f"**User intent:** {request.user_intent}\n",
            "\n",
            "## Section guidance\n",
            guidance_block,
        ]

        # Add expected outputs if the profile defines them.
        if expected_outputs:
            outputs_list = "\n".join(f"* {output}" for output in expected_outputs)
            prompt_parts.extend(
                [
                    "\n",
                    "## Expected outputs\n",
                    "Consider including the following elements where relevant:\n",
                    f"{outputs_list}\n",
                ]
            )

        prompt_parts.extend(
            [
                "\n",
                "## Call / topic context\n",
                f"{topic_context_block}\n",
                "\n",
                "## Retrieved evidence\n",
                "Use the following examples from past proposals as inspiration. "
                "Each example is labeled with its source status. **Funded** examples "
                "represent successful patterns; **Rejected** examples are cautionary; "
                "**ESR notes** are advisory commentary, not ground truth. Cite each "
                "example you use as [N].\n",
                "\n",
                f"{evidence_block}\n",
                "\n",
                "## Output instructions\n",
                "Write a concise, technically precise draft of the section in markdown. "
                "Cite supporting evidence inline using [N] markers matching the numbered "
                "list above. Do not invent information not supported by the retrieved "
                "evidence. If a citation is from a REJECTED example, frame it as a "
                "cautionary lesson. Do not cite ESR notes as fact.",
            ]
        )

        # AC #3 of issue #9: when a TopicContext carries expected
        # outcomes, instruct the LLM to reference them in the draft.
        # The exact wording is pinned by the
        # ``test_prompt_includes_expected_outcomes_instruction_when_topic_supplied``
        # test.
        if topic_context is not None and topic_context.expected_outcomes:
            prompt_parts.append(
                " Reference the supplied Expected Outcomes from the call / topic "
                "context where appropriate, framing the draft so it explicitly "
                "addresses the topic's intended outcomes."
            )
        prompt_parts.append("\n")

        prompt = "".join(prompt_parts)

        return prompt, citations

    # ------------------------------------------------------------------
    # internal helpers — small, named, testable indirectly via build()
    # ------------------------------------------------------------------

    @staticmethod
    def _build_citations(results: list[RetrievalResult]) -> list[CitationRef]:
        """Turn retrieval results into 1-indexed :class:`CitationRef` records."""

        citations: list[CitationRef] = []
        for idx, result in enumerate(results, start=1):
            chunk = result.chunk
            meta = chunk.metadata
            proposal = meta.proposal
            citations.append(
                CitationRef(
                    citation_id=idx,
                    source_status=meta.source_status,
                    programme=proposal.programme,
                    call_id=proposal.call_id,
                    proposal_title=proposal.proposal_title,
                    section_heading=meta.parent_section_heading or meta.anchor.section_heading,
                    page=meta.anchor.page,
                    chunk_id=chunk.chunk_id,
                    snippet=_truncate_snippet(chunk.text),
                )
            )
        return citations

    @staticmethod
    def _format_evidence(
        results: list[RetrievalResult],
        citations: list[CitationRef],
    ) -> str:
        """Render the retrieved evidence as a numbered list block.

        Each entry uses the stable ``[N] **STATUS** — call_id, p. P, §heading``
        format so :data:`eurpe.generation.llm._PROMPT_CITATION_LINE`
        can locate every marker. The blockquote on the second line is
        the snippet excerpt the model can quote.
        """

        if not results:
            return "(no examples retrieved)"

        # ``citations`` is built in retrieval order from ``results`` so
        # we only iterate the citation list here — every field the
        # evidence header needs is on the citation already. ``results``
        # is kept in the signature so a future renderer that wants the
        # raw chunk text or score can be added without breaking callers.
        lines: list[str] = []
        for citation in citations:
            label = _STATUS_LABEL[citation.source_status]
            programme = SectionPromptBuilder._humanize_programme(citation.programme)
            section = citation.section_heading or "(no section heading)"
            page = f"p. {citation.page}" if citation.page is not None else "p. ?"
            header = (
                f"[{citation.citation_id}] **{label}** — {programme} call "
                f"{citation.call_id}, {page}, §{section}"
            )
            lines.append(header)
            # Indented blockquote keeps the snippet visually distinct from
            # the header without forcing the LLM to parse a separate JSON
            # structure.
            lines.append(f"    > {citation.snippet}")
            lines.append("")  # blank line between entries

        # Drop the trailing blank line so the prompt doesn't have a double
        # newline before "## Output instructions".
        return "\n".join(lines).rstrip()

    @staticmethod
    def _format_topic_context_block(
        topic_context: TopicContext | None,
        call_context: str,
    ) -> str:
        """Render the ``## Call / topic context`` block body.

        Three rendering paths, picked by inputs:

        1. ``topic_context is None`` — fall back to the legacy free-text
           rendering: ``call_context`` verbatim, or ``(none provided)``
           when empty. This preserves the pre-issue-#9 behaviour.
        2. ``topic_context`` set — render programme / call id / topic id
           / title / destination / expected-outcomes / scope as a
           series of labelled lines. Any unset field is omitted. When
           ``call_context`` is also non-empty, its text appears
           appended under a ``**Free-text notes:**`` sub-block so
           operators can keep pasting additional notes alongside the
           structured context.
        3. Edge case: a TopicContext with every parsed field empty
           still renders to (effectively) the legacy path because no
           label lines will be emitted.

        The ``**Programme:**`` / ``**Call ID:**`` / ``**Topic ID:**`` /
        ``**Topic title:**`` / ``**Expected outcomes:**`` / ``**Scope:**``
        / ``**Destination:**`` labels are part of the prompt contract;
        they are pinned by the prompt-builder tests.
        """

        call_context_stripped = call_context.strip()

        if topic_context is None:
            return call_context_stripped or "(none provided)"

        lines: list[str] = []
        if topic_context.programme is not None:
            programme_label = SectionPromptBuilder._humanize_programme(topic_context.programme)
            lines.append(f"**Programme:** {programme_label}")
        if topic_context.call_id:
            lines.append(f"**Call ID:** {topic_context.call_id}")
        if topic_context.topic_id:
            lines.append(f"**Topic ID:** {topic_context.topic_id}")
        if topic_context.topic_title:
            lines.append(f"**Topic title:** {topic_context.topic_title}")
        if topic_context.destination:
            lines.append(f"**Destination:** {topic_context.destination}")

        if topic_context.expected_outcomes:
            lines.append("")
            lines.append("**Expected outcomes:**")
            for outcome in topic_context.expected_outcomes:
                lines.append(f"* {outcome}")

        if topic_context.scope:
            lines.append("")
            lines.append(f"**Scope:** {topic_context.scope}")

        if call_context_stripped:
            lines.append("")
            lines.append("**Free-text notes:**")
            lines.append(call_context_stripped)

        rendered = "\n".join(lines).strip()
        # Defensive: if the TopicContext was entirely empty and no
        # call_context was supplied, fall back to the legacy marker so
        # the prompt structure is preserved.
        return rendered or "(none provided)"

    @staticmethod
    def _humanize_section_type(section_type: SectionType) -> str:
        """``METHODOLOGY`` → ``"Methodology"``; ``IMPACT_PATHWAY`` → ``"Impact Pathway"``."""

        return section_type.value.replace("_", " ").title()

    @staticmethod
    def _humanize_programme(programme: Programme) -> str:
        """``HORIZON_EUROPE`` → ``"Horizon Europe"``; ``CEF`` → ``"Cef"``.

        Callers are responsible for null-checking; the
        ``request.target_programme is not None`` guard at the call site
        in :meth:`build` is the contract.
        """

        return programme.value.replace("_", " ").title()
