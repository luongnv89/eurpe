"""Generation package for EURPE.

Hosts the section-generation workflow: retrieve relevant past-proposal
evidence, build a source-status-aware prompt, call a local LLM
(Ollama or a deterministic stub), validate citation markers, and
return a structured :class:`GenerationDraft`.

Public surface kept narrow on purpose so the rest of the codebase
imports from ``eurpe.generation`` rather than internal modules:

* :class:`GenerationRequest`, :class:`GenerationDraft`,
  :class:`CitationRef` — Pydantic models that flow into / out of
  the workflow.
* :class:`SectionPromptBuilder` — turns request + retrieval results
  into an LLM prompt and a structured citation list.
* :class:`LLMClient` (protocol), :class:`OllamaLLMClient`,
  :class:`DeterministicLLMClient`, :func:`make_llm_client` — LLM
  client backends and factory.
* :class:`SectionGenerationWorkflow` — orchestrates the full
  retrieve → prompt → generate → validate → assemble pipeline.
* Three exception types: :class:`GenerationError`,
  :class:`LLMUnavailableError`, :class:`OfflineLLMError`.

LangGraph is intentionally NOT used yet; see the docstring of
:mod:`eurpe.generation.workflow` for the rationale.
"""

from __future__ import annotations

from eurpe.generation.errors import (
    GenerationError,
    LLMUnavailableError,
    OfflineLLMError,
)
from eurpe.generation.llm import (
    DeterministicLLMClient,
    LLMClient,
    OllamaLLMClient,
    make_llm_client,
)
from eurpe.generation.models import CitationRef, GenerationDraft, GenerationRequest
from eurpe.generation.prompt import SECTION_GUIDANCE, SectionPromptBuilder
from eurpe.generation.workflow import SectionGenerationWorkflow

__all__ = [
    "SECTION_GUIDANCE",
    "CitationRef",
    "DeterministicLLMClient",
    "GenerationDraft",
    "GenerationError",
    "GenerationRequest",
    "LLMClient",
    "LLMUnavailableError",
    "OfflineLLMError",
    "OllamaLLMClient",
    "SectionGenerationWorkflow",
    "SectionPromptBuilder",
    "make_llm_client",
]
