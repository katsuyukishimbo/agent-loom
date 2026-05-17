"""Phase 1b DoD: cross-process recall through pgvector.

Two subprocesses run hello_harness with `--store pg`. After the second run we
assert that:

  (a) at least two episodes live in the DB,
  (b) the first run's episode has references_count >= 1 (the second run's
      Planner recalled it).

If Postgres+pgvector is not reachable the test skips.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_loom.memory.store_pg import (
    PgvectorEpisodicStore,
    _default_database_url,
    reachable,
)


def _db_available() -> bool:
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        reachable(_default_database_url())
    )


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="Postgres+pgvector not reachable; run ./scripts/dev_db_up.sh && alembic upgrade head",
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _child_env() -> dict[str, str]:
    """Env for the spawned hello_harness process.

    - AGENT_LOOM_FAKE_LLM=1 keeps the run offline (no API key required).
    - AGENT_LOOM_USE_PG=1 routes episodes through the pgvector store.
    - PYTHONPATH= explicit so `python -m` finds the in-tree package even
      when the test runner started from a different cwd.
    """
    env = os.environ.copy()
    env["AGENT_LOOM_FAKE_LLM"] = "1"
    env["AGENT_LOOM_USE_PG"] = "1"
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    # Trace files would otherwise pile up in the repo's runs/ dir; redirect.
    env.setdefault("TRACE_DIR", str(REPO_ROOT / "runs"))
    return env


async def test_cross_process_recall(tmp_path: Path) -> None:
    # Clean slate. The test owns the table for its lifetime.
    store = PgvectorEpisodicStore()
    await store.truncate()
    assert await store.count() == 0

    env = _child_env()

    # First child writes one episode (success path).
    result1 = subprocess.run(
        [sys.executable, "-m", "agent_loom.examples.hello_harness", "--store", "pg"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result1.returncode == 0, (
        f"first child failed: stdout={result1.stdout!r} stderr={result1.stderr!r}"
    )
    after_first = await store.list_all()
    assert len(after_first) == 1, (
        f"expected 1 episode after first run, got {len(after_first)}; stdout={result1.stdout!r}"
    )
    seed_id = after_first[0].episode_id

    # Second child writes its own episode AND recalls the first one through
    # the Planner. The `references_count` bump is what proves recall fired.
    result2 = subprocess.run(
        [sys.executable, "-m", "agent_loom.examples.hello_harness", "--store", "pg"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result2.returncode == 0, (
        f"second child failed: stdout={result2.stdout!r} stderr={result2.stderr!r}"
    )

    after_second = await store.list_all()
    assert len(after_second) == 2, (
        f"expected 2 episodes after second run, got {len(after_second)}"
    )

    seed = next(e for e in after_second if e.episode_id == seed_id)
    assert seed.references_count >= 1, (
        "second run's Planner did not recall the seed episode"
    )
