"""Executor-side reflection wiring (Phase 2).

We assert:
  - On Verifier failure, the Executor writes a SECOND memory_write event
    tagged kind="reflection".
  - The reflection Episode lands in the store as a separate row from the
    raw failure Episode.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.types import FailureCategory, RunStatus
from agent_loom.core.verifier import Verifier
from agent_loom.memory.hub import MemoryHub


class _AlwaysFail(Verifier):
    async def verify(self, *, contract, artifact, trace):  # type: ignore[override]
        judgement, event = await super().verify(
            contract=contract, artifact=artifact, trace=trace
        )
        return (
            judgement.model_copy(
                update={
                    "passed": False,
                    "score": 0.1,
                    "failure_category": FailureCategory.HALLUCINATED_ARTIFACT,
                    "reflection": "References function nope() which does not exist.",
                }
            ),
            event,
        )


async def test_failure_run_writes_reflection_episode(tmp_trace_dir: Path) -> None:
    hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=hub),
        generator=Generator(),
        verifier=_AlwaysFail(),
        memory_hub=hub,
        max_iterations=1,
    )
    run = await executor.run("Write nope().")
    assert run.status == RunStatus.FAILED

    all_eps = await hub.store.list_all()
    sources = {e.source for e in all_eps}
    assert "executor" in sources  # the raw failure
    assert "reflection" in sources  # the compaction


async def test_reflection_trace_event_is_emitted(tmp_trace_dir: Path) -> None:
    hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=hub),
        generator=Generator(),
        verifier=_AlwaysFail(),
        memory_hub=hub,
        max_iterations=1,
    )
    run = await executor.run("Write nope().")
    trace_path = tmp_trace_dir / str(run.run_id) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    reflection_writes = [
        e
        for e in events
        if e.get("kind") == "memory_write" and e.get("inputs", {}).get("kind") == "reflection"
    ]
    assert reflection_writes, (
        f"Expected a memory_write event tagged reflection; got kinds: "
        f"{[e.get('inputs', {}).get('kind') for e in events if e.get('kind') == 'memory_write']}"
    )


async def test_passing_run_does_not_trigger_reflection(tmp_trace_dir: Path) -> None:
    """The reflection step must only fire on failure."""
    hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=hub),
        generator=Generator(),
        verifier=Verifier(),  # fake mode: pass
        memory_hub=hub,
        max_iterations=1,
    )
    run = await executor.run("Write fib.")
    assert run.status == RunStatus.COMPLETED

    all_eps = await hub.store.list_all()
    sources = {e.source for e in all_eps}
    assert "reflection" not in sources


async def test_failure_recall_includes_both_reflection_and_raw_failure(
    tmp_trace_dir: Path,
) -> None:
    """After a failure the recall must surface BOTH the raw failure and its reflection.

    Why this isn't a strict ordering test: the fake EmbeddingService is
    SHA-256-based, so two distinct content strings produce uncorrelated unit
    vectors. The R × I × R product can end up dominated by relevance noise,
    leaving the importance gap (raw=5 vs reflection≥8) unable to reliably
    decide the order. Real-mode integration (with semantic embeddings) is the
    right place to assert "reflection outranks raw"; here we only check that
    both signals are present in the top-K so the Planner has the constraint
    available downstream.
    """
    hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=hub),
        generator=Generator(),
        verifier=_AlwaysFail(),
        memory_hub=hub,
        max_iterations=1,
    )
    await executor.run("Write nope().")

    hits = await hub.recall("nope", top_k=5)
    sources = {ep.source for ep in hits}
    assert "reflection" in sources, "reflection episode missing from top-K recall"
    assert "executor" in sources, "raw failure episode missing from top-K recall"
