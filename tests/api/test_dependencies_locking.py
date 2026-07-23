"""Concurrency regression tests for the ``_cache_lock`` double-checked
locking in :mod:`eurpe.api.dependencies` (perf/race-fix PR #89).

Each provider does an unlocked fast-path read, then re-checks under
``_cache_lock`` before constructing the singleton. These tests make the
construction step slow (``time.sleep``) and hammer the provider with
concurrent threads; if the lock were missing or misapplied, several
threads would race past the unlocked check simultaneously and each
construct their own instance.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from eurpe.api import dependencies as deps
from tests._helpers.offline import write_offline_config


@pytest.fixture
def offline_config(tmp_path: Path) -> Path:
    cfg_path = write_offline_config(tmp_path)
    deps.set_config_path(cfg_path)
    yield cfg_path
    deps.reset_dependency_caches()


def test_concurrent_get_config_constructs_singleton_once(
    offline_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_load_config = deps.load_config
    construction_count = 0
    count_lock = threading.Lock()

    def slow_load_config(path: Path):
        nonlocal construction_count
        with count_lock:
            construction_count += 1
        time.sleep(0.05)
        return real_load_config(path)

    monkeypatch.setattr(deps, "load_config", slow_load_config)

    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        cfg = deps.get_config()
        with results_lock:
            results.append(cfg)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert construction_count == 1
    assert len(results) == 8
    assert len({id(r) for r in results}) == 1


def test_concurrent_get_generation_service_constructs_singleton_once(
    offline_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_build = deps._build_generation_service
    construction_count = 0
    count_lock = threading.Lock()

    def slow_build(cfg, *, collection: str, key: tuple[str, str]):
        nonlocal construction_count
        with count_lock:
            construction_count += 1
        time.sleep(0.05)
        return real_build(cfg, collection=collection, key=key)

    monkeypatch.setattr(deps, "_build_generation_service", slow_build)

    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        service = deps.get_generation_service(deps.get_config())
        with results_lock:
            results.append(service)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert construction_count == 1
    assert len(results) == 6
    assert len({id(r) for r in results}) == 1
