"""Test-only helpers that build :class:`Chunk` records from the existing
``tests/fixtures/metadata/*.yaml`` files.

The metadata fixtures already encode every interesting source-status /
programme combination, so re-using them keeps the test surface honest:
if the schema gains a new required field, the fixtures fail to parse
and the chunks fail to build, which is the loud signal we want.

The chunks here carry synthetic *text* designed to be easy to query
deterministically:

* Every chunk's text contains a unique ``marker_token`` derived from
  the fixture filename. A query containing that token has overwhelming
  hash overlap with that one chunk under
  :class:`~eurpe.retrieval.embeddings.DeterministicHashEmbedder` and
  thus comes back at rank 1.
* The remaining text is filler that names the source-status and
  programme so a reader scanning failed test output can immediately
  see what went wrong.

Functions are grouped here (not in ``conftest.py``) because they are
needed both by ``test_index.py`` (to populate the index) and by
``test_retrieval_cli.py`` (when fabricating CLI inputs).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from eurpe.retrieval.models import Chunk
from eurpe.schema import ChunkMetadata

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "metadata"


def _marker_for(fixture_name: str) -> str:
    """Return the unique token embedded in the chunk text for a fixture.

    The token combines the fixture stem (e.g. ``funded_horizon_europe``)
    with a fixed ``-marker`` suffix. Concatenating both halves makes the
    token unlikely to collide with a real proposal vocabulary, and the
    fixed suffix gives query strings a stable handle to reference.
    """

    return f"{Path(fixture_name).stem}-marker"


def _load_metadata(yaml_path: Path) -> ChunkMetadata:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return ChunkMetadata.model_validate(raw)


def build_fixture_chunks() -> list[Chunk]:
    """Build one :class:`Chunk` per fixture YAML, in sorted filename order.

    Sorting on filename gives a stable order so a test can assert on
    "the funded chunk is at index 1" without depending on filesystem
    enumeration order.
    """

    out: list[Chunk] = []
    for yaml_path in sorted(FIXTURES_DIR.glob("*.yaml")):
        meta = _load_metadata(yaml_path)
        marker = _marker_for(yaml_path.name)
        text = (
            f"This chunk is the {marker}. "
            f"Source status is {meta.source_status.value}. "
            f"Programme is {meta.proposal.programme.value}. "
            f"Call id is {meta.proposal.call_id}. "
            f"Section heading is {meta.parent_section_heading or 'unnamed'}."
        )
        out.append(Chunk(text=text, metadata=meta))
    return out


def query_text_for(fixture_name: str) -> str:
    """Return a query string that should match the named fixture's chunk.

    Uses the marker token plus a couple of distinctive non-stopword
    tokens so the deterministic-hash embedder has more than one bucket
    of overlap. The advisor's note: avoid stopwords; pick distinctive
    multi-token queries.
    """

    return f"{_marker_for(fixture_name)} distinctive marker query"
