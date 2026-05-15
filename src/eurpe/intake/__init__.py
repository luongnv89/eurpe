"""Call/topic intake package — turn pasted text or PDF excerpts into context.

The intake layer is the boundary between user-supplied Work Programme
material and the section-generation prompt. It produces a single
:class:`TopicContext` record from one of two input forms:

* :func:`extract_topic_context_from_text` — operator pastes the topic
  page as plaintext (e.g., copy-paste from the Funding & Tenders portal).
* :func:`extract_topic_context_from_pdf` — operator uploads a PDF
  excerpt of the topic page; Docling parses it offline and the result
  is fed into the text extractor.

The intake layer does NOT touch the LLM or the retriever — it is a
pure parsing module. The generation layer reads the resulting
:class:`TopicContext` off the :class:`~eurpe.generation.GenerationRequest`
and renders it into the prompt (issue #9 AC #2 + #3).

Public surface kept narrow so callers import from ``eurpe.intake``
rather than the internal modules.
"""

from __future__ import annotations

from eurpe.intake.extractor import (
    extract_topic_context_from_pdf,
    extract_topic_context_from_text,
)
from eurpe.intake.models import TopicContext, TopicSource

__all__ = [
    "TopicContext",
    "TopicSource",
    "extract_topic_context_from_pdf",
    "extract_topic_context_from_text",
]
