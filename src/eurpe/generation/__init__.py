"""Generation package for EURPE.

Hosts the section-generation workflow: retrieve relevant past-proposal
evidence, build a source-status-aware prompt, call a local LLM
(Ollama/local OpenAI-compatible engines) or explicitly configured cloud
provider, validate citation markers, and return a structured
:class:`GenerationDraft`.

Public surface kept narrow on purpose so the rest of the codebase
imports from ``eurpe.generation`` rather than internal modules:

* :class:`GenerationRequest`, :class:`GenerationDraft`,
  :class:`CitationRef` — Pydantic models that flow into / out of
  the workflow.
* :class:`SectionPromptBuilder` — turns request + retrieval results
  into an LLM prompt and a structured citation list.
* :class:`LLMClient` (protocol), :class:`DeterministicLLMClient`,
  :func:`make_llm_client` — LLM client backends and factory.
* :class:`SectionGenerationWorkflow` — orchestrates the full
  retrieve → prompt → generate → validate → assemble pipeline.
* :class:`MarkdownCitationRenderer`, :data:`STATUS_LABEL`,
  :data:`STATUS_BADGE`, :data:`STATUS_CAVEAT` — Markdown rendering
  with visible source-status labels (Issue #7).
* :class:`CitationAudit`, :class:`AuditFinding`, :class:`AuditResult`,
  :class:`AuditSeverity` — release-blocking checks for source-status
  compliance (Issue #7).
* Three exception types: :class:`GenerationError`,
  :class:`LLMUnavailableError`, :class:`OfflineLLMError`.

LangGraph is intentionally NOT used yet; see the docstring of
:mod:`eurpe.generation.workflow` for the rationale.
"""

from __future__ import annotations

from eurpe.generation.audit import (
    AuditFinding,
    AuditResult,
    AuditSeverity,
    CitationAudit,
)
from eurpe.generation.audit_harness import (
    CitationAuditRow,
    DraftAuditResult,
    ReleaseAuditHarness,
    ReleaseAuditHarnessError,
    ReleaseAuditReport,
    has_release_blocking_findings,
)
from eurpe.generation.critic import CriticAgent, build_requirements_checked
from eurpe.generation.critic_loop import (
    DEFAULT_MAX_ITERATIONS,
    MAX_ITERATIONS_CEILING,
    CriticLoopWorkflow,
    IterationResult,
)
from eurpe.generation.errors import (
    GenerationError,
    LLMUnavailableError,
    OfflineLLMError,
)
from eurpe.generation.llm import DeterministicLLMClient, LLMClient, OllamaLLMClient, make_llm_client
from eurpe.generation.models import (
    CitationRef,
    GenerationDraft,
    GenerationRequest,
    IterationRecord,
)
from eurpe.generation.profiles import (
    DraftingProfile,
    list_available_profiles,
    load_profile,
)
from eurpe.generation.prompt import SECTION_GUIDANCE, SectionPromptBuilder
from eurpe.generation.render import (
    STATUS_BADGE,
    STATUS_CAVEAT,
    STATUS_LABEL,
    MarkdownCitationRenderer,
)
from eurpe.generation.service import (
    GenerationService,
    SectionGenerationRequest,
    SectionIterationRequest,
)
from eurpe.generation.workflow import SectionGenerationWorkflow

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "MAX_ITERATIONS_CEILING",
    "SECTION_GUIDANCE",
    "STATUS_BADGE",
    "STATUS_CAVEAT",
    "STATUS_LABEL",
    "AuditFinding",
    "AuditResult",
    "AuditSeverity",
    "CitationAudit",
    "CitationAuditRow",
    "CitationRef",
    "CriticAgent",
    "CriticLoopWorkflow",
    "DeterministicLLMClient",
    "DraftAuditResult",
    "DraftingProfile",
    "GenerationDraft",
    "GenerationError",
    "GenerationRequest",
    "GenerationService",
    "IterationRecord",
    "IterationResult",
    "LLMClient",
    "LLMUnavailableError",
    "MarkdownCitationRenderer",
    "OfflineLLMError",
    "OllamaLLMClient",
    "ReleaseAuditHarness",
    "ReleaseAuditHarnessError",
    "ReleaseAuditReport",
    "SectionGenerationRequest",
    "SectionGenerationWorkflow",
    "SectionIterationRequest",
    "SectionPromptBuilder",
    "build_requirements_checked",
    "has_release_blocking_findings",
    "list_available_profiles",
    "load_profile",
    "make_llm_client",
]
