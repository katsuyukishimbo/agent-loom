"""Shared pytest fixtures and global fake-mode setup.

Why force fake mode here: every test in this suite should run offline. If a
test wants to assert real-mode behaviour it should monkeypatch the env var
back, never the other way round.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("AGENT_LOOM_FAKE_LLM", "1")


@pytest.fixture
def run_id():
    return uuid4()


@pytest.fixture
def tmp_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point TRACE_DIR at a per-test tmp directory.

    Why monkeypatch the env: Settings is read via get_settings() at call time,
    so a fresh env var picks up the redirect without reloading the module.
    """
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    return tmp_path
