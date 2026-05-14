"""Shared pytest fixtures for EURPE.

Provides a ``no_network`` fixture that monkeypatches :func:`socket.socket.connect`
so that any code path attempting to reach the network during a test causes a
hard failure. Use it on tests that must prove offline-mode behaviour.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test the moment any code tries to open a TCP connection."""

    def _blocked(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - intentional
        raise pytest.fail.Exception(
            "Network access attempted during a test marked offline-only."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
