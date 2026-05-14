"""Exception hierarchy for the generation package.

A single base (:class:`GenerationError`) so callers — most importantly
:mod:`eurpe.generation.cli` and the future FastAPI surface — can write
``except GenerationError`` once and surface a clean ``error: ...`` line on
every drafting-layer failure. Mirrors the ``IndexingError`` /
``IngestionError`` patterns in the sibling packages so the three feel
idiomatic side-by-side.

Subclasses identify *category* rather than *cause*:

* :class:`LLMUnavailableError` — the LLM endpoint cannot be reached
  (e.g., Ollama is not running on ``localhost:11434``). Distinct from a
  generic :class:`GenerationError` so the CLI can suggest a recovery
  action ("start ``ollama serve`` and pull the model") rather than a
  generic "request failed" message.
* :class:`OfflineLLMError` — the caller asked for a real LLM under
  ``offline_mode=True`` and no offline fallback was available. Reserved
  for a future "strict offline" mode; the current
  :func:`~eurpe.generation.llm.make_llm_client` factory always falls
  back to :class:`~eurpe.generation.llm.DeterministicLLMClient` so this
  error is not raised today.
"""

from __future__ import annotations


class GenerationError(Exception):
    """Base class for any failure inside :mod:`eurpe.generation`.

    Catch this in callers (CLI, API) when you want one branch for "the
    drafting layer broke". Subclasses carry more specific intent for
    handlers that want to differentiate.
    """


class LLMUnavailableError(GenerationError):
    """Raised when the LLM endpoint cannot be reached.

    Typical cause: Ollama is not running, or the configured
    ``ollama_base_url`` is wrong. The error message should be
    actionable — name the URL that was probed and suggest the recovery
    command.
    """


class OfflineLLMError(GenerationError):
    """Offline mode requested but no offline LLM client is available.

    Reserved for a future "strict offline" mode. The current
    :func:`~eurpe.generation.llm.make_llm_client` falls back to
    :class:`~eurpe.generation.llm.DeterministicLLMClient` whenever the
    real LLM is unreachable under offline mode, so this error is never
    raised today. Kept here so a stricter future mode does not need to
    grow a new exception type.
    """
