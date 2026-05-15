"""FastAPI dependency providers backed by per-config singletons.

Why singletons?
---------------
``DoclingProposalParser`` caches its underlying ``DocumentConverter`` and
``ChromaIndex`` opens a ``chromadb.PersistentClient`` (which itself
holds OS-level file locks). Re-instantiating either on every request would
either pay the construction cost on every parse / upsert (Docling) or
collide with itself on the same on-disk path (Chroma).

Why a hand-rolled cache instead of ``functools.lru_cache``?
-----------------------------------------------------------
:class:`EurpeConfig` is a Pydantic ``BaseModel`` and is not hashable by
default. ``lru_cache`` requires hashable arguments, so we key the cache
on the configuration *path* string instead of the ``EurpeConfig`` itself.
This also matches the natural use case: an operator might switch
configuration files in a long-lived process (rare in practice, but the
test suite does it constantly), and we want the new config's index path
to win without leaking the old one.

Tests override providers via ``app.dependency_overrides`` so the cache
is never exercised in the fast tier — every test gets a fresh stub.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends

from eurpe.api.storage import ParseTokenStore
from eurpe.config import (
    DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    EurpeConfig,
    ensure_config_file,
    ensure_runtime_dirs,
    load_config,
)
from eurpe.ingestion.docling_parser import DoclingProposalParser
from eurpe.retrieval import (
    ChromaIndex,
    HierarchicalChunker,
    make_embedder,
)

# Module-level singleton caches, keyed on a string path so Pydantic
# models (unhashable) never become cache keys directly.
_config_cache: dict[str, EurpeConfig] = {}
_parser_cache: dict[str, DoclingProposalParser] = {}
_chunker_cache: dict[str, HierarchicalChunker] = {}
_index_cache: dict[tuple[str, str], ChromaIndex] = {}
_token_store_cache: dict[str, ParseTokenStore] = {}

# The configuration path used by every provider. Tests can override this
# by clearing the caches and writing their own config — see
# ``reset_dependency_caches`` below.
_CONFIG_PATH: Path = DEFAULT_CONFIG_PATH


def set_config_path(config_path: Path) -> None:
    """Override the config path used by the providers and reset all caches.

    Tests use this to point the API at a temp-dir config without touching
    the repo's ``config.yaml``. Production code never calls it — the
    default is the repo-root ``config.yaml`` and that is what
    ``uvicorn eurpe.api.main:app`` will pick up.
    """

    global _CONFIG_PATH
    _CONFIG_PATH = config_path
    reset_dependency_caches()


def reset_dependency_caches() -> None:
    """Drop every cached singleton so the next request rebuilds from disk.

    Tests call this in their teardown so cross-test state cannot leak
    (especially the open Chroma client, which holds a file lock).
    """

    _config_cache.clear()
    _parser_cache.clear()
    _chunker_cache.clear()
    _index_cache.clear()
    _token_store_cache.clear()


def get_config() -> EurpeConfig:
    """Return the per-process :class:`EurpeConfig` singleton.

    Loads the file the first time it is requested, then memoises the
    parsed model. ``ensure_runtime_dirs`` runs once on first load so a
    fresh checkout has the directories in place before the first request
    tries to write to them.
    """

    key = str(_CONFIG_PATH)
    cached = _config_cache.get(key)
    if cached is not None:
        return cached
    used_path = ensure_config_file(_CONFIG_PATH, EXAMPLE_CONFIG_PATH)
    cfg = load_config(used_path).resolve_paths()
    ensure_runtime_dirs(cfg)
    _config_cache[key] = cfg
    return cfg


def get_parser(cfg: EurpeConfig = Depends(get_config)) -> DoclingProposalParser:
    """Return the cached :class:`DoclingProposalParser` for this config.

    Honours ``cfg.offline_mode``: in offline mode (the production default)
    Docling is constructed with ``do_ocr=False`` so no model weights are
    downloaded. The parser's underlying ``DocumentConverter`` is itself
    cached on the parser instance, so re-using one parser across many
    requests is cheap.
    """

    key = str(_CONFIG_PATH)
    cached = _parser_cache.get(key)
    if cached is not None:
        return cached
    parser = DoclingProposalParser(offline=cfg.offline_mode)
    _parser_cache[key] = parser
    return parser


def get_chunker(cfg: EurpeConfig = Depends(get_config)) -> HierarchicalChunker:
    """Return the shared :class:`HierarchicalChunker` instance.

    The chunker is stateless aside from its tuning knobs (``target_chars``,
    ``overlap_chars``, ``min_chunk_chars``), so a single shared instance
    serves every request.
    """

    # ``cfg`` is unused today but kept in the signature so a future
    # configuration knob (e.g., ``chunker.target_chars``) can be plumbed
    # through without changing every caller.
    del cfg
    key = str(_CONFIG_PATH)
    cached = _chunker_cache.get(key)
    if cached is not None:
        return cached
    chunker = HierarchicalChunker()
    _chunker_cache[key] = chunker
    return chunker


def get_index(
    cfg: EurpeConfig = Depends(get_config),
    *,
    collection: str = "default",
) -> ChromaIndex:
    """Return a cached :class:`ChromaIndex` for ``cfg.index_path`` + collection.

    Two different collections (e.g., ``default`` and ``synthetic``) live
    side-by-side under the same ``index_path`` and need separate
    :class:`ChromaIndex` instances, so the cache key includes the
    collection name.

    The embedder is built fresh each cache miss because
    :func:`make_embedder` is cheap (it only opens a network connection on
    first use) and we want the cache key to track config changes.
    """

    key = (str(_CONFIG_PATH), collection)
    cached = _index_cache.get(key)
    if cached is not None:
        return cached
    embedder = make_embedder(cfg)
    index = ChromaIndex(
        index_path=cfg.index_path,
        embedder=embedder,
        collection_name=collection,
    )
    _index_cache[key] = index
    return index


def get_token_store(cfg: EurpeConfig = Depends(get_config)) -> ParseTokenStore:
    """Return the cached :class:`ParseTokenStore` rooted at ``cfg.runtime_dir``."""

    key = str(_CONFIG_PATH)
    cached = _token_store_cache.get(key)
    if cached is not None:
        return cached
    store = ParseTokenStore(cfg.runtime_dir)
    _token_store_cache[key] = store
    return store
