"""End-to-end memory integration tests.

The headline test (`test_second_run_recalls_first_run_episode`) is the Phase 1a
Definition of Done in code form: two sequential runs in the same process, with
the second run reading back the first run's episode through R × I × R recall.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.types import RunStatus
from agent_loom.core.verifier import Verifier
from agent_loom.memory.hub import MemoryHub


def _build_executor_with_memory() -> tuple[Executor, MemoryHub]:
    hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=hub),
        generator=Generator(),
        verifier=Verifier(),
        memory_hub=hub,
    )
    return executor, hub


async def test_first_run_writes_one_episode(tmp_trace_dir: Path) -> None:
    """A passing run must persist exactly one episode (success path)."""
    executor, hub = _build_executor_with_memory()
    run = await executor.run("Write a Python function fib(n).")
    assert run.status == RunStatus.COMPLETED

    episodes = await hub.store.list_all()
    assert len(episodes) == 1
    only = episodes[0]
    assert only.embedding is not None
    assert only.metadata["passed"] == "true"


async def test_second_run_recalls_first_run_episode(tmp_trace_dir: Path) -> None:
    """Headline DoD: 2-runs-in-process, second run reads the first's episode.

    What this proves:
    - The Planner's recall hook fires on the second run.
    - The shared MemoryHub bridges runs.
    - references_count actually increments when recall returns a hit.
    """
    executor, hub = _build_executor_with_memory()

    # Run 1: writes the seed episode.
    run1 = await executor.run("Write a Python function fib(n).")
    assert run1.status == RunStatus.COMPLETED
    assert len(await hub.store.list_all()) == 1

    seed_id = (await hub.store.list_all())[0].episode_id

    # Run 2: similar prompt; second episode lands AND the first is recalled.
    run2 = await executor.run("Implement a Python fibonacci function.")
    assert run2.status == RunStatus.COMPLETED

    episodes = await hub.store.list_all()
    assert len(episodes) == 2

    # The first episode must have been referenced at least once by run 2's
    # recall before its own episode was written.
    seed = next(e for e in episodes if e.episode_id == seed_id)
    assert seed.references_count >= 1


async def test_second_run_trace_contains_memory_read(tmp_trace_dir: Path) -> None:
    """The trace JSONL of the second run must include a memory_read event."""
    executor, _hub = _build_executor_with_memory()

    await executor.run("Write a Python function fib(n).")
    run2 = await executor.run("Implement a Python fibonacci function.")

    trace_path = tmp_trace_dir / str(run2.run_id) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    kinds = {(e["module"], e["kind"]) for e in events}
    assert ("planner", "memory_read") in kinds


async def test_run_trace_contains_memory_write(tmp_trace_dir: Path) -> None:
    """Every run with memory_hub enabled emits a memory_write event for the episode."""
    executor, _hub = _build_executor_with_memory()
    run = await executor.run("Write a Python function fib(n).")

    trace_path = tmp_trace_dir / str(run.run_id) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]

    # Two memory_write events expected: one from _write_memory_episode (our
    # new code), one from the executor's finalize block. Find the episode one.
    ep_writes = [
        e
        for e in events
        if e["kind"] == "memory_write" and "episode_id" in e.get("outputs", {})
    ]
    assert len(ep_writes) == 1


async def test_recall_summary_lands_in_generator_persona(
    tmp_trace_dir: Path,
) -> None:
    """The recall preface must be woven into the contract persona on run 2.

    We inspect the planner's TraceEvent on the second run via the trace JSONL —
    `outputs.goal` reflects the SprintContract goal. We can't read the persona
    directly from trace (it's not serialised there for privacy), so this test
    asserts the recall-prefix code path runs by checking the memory_read event
    exists and lists the seed episode's id.
    """
    executor, hub = _build_executor_with_memory()

    await executor.run("Write a Python function fib(n).")
    seed_id = (await hub.store.list_all())[0].episode_id

    run2 = await executor.run("Implement a Python fibonacci function.")
    trace_path = tmp_trace_dir / str(run2.run_id) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    reads = [e for e in events if e["kind"] == "memory_read"]
    assert reads, "second run must emit a memory_read event"
    recalled_ids = reads[0]["outputs"]["episode_ids"]
    assert str(seed_id) in recalled_ids


async def test_no_memory_hub_falls_back_to_phase0_behaviour(
    tmp_trace_dir: Path,
) -> None:
    """Without a MemoryHub the harness behaves like Phase 0: no recall, no writes."""
    executor = Executor(
        planner=Planner(),  # no memory_hub
        generator=Generator(),
        verifier=Verifier(),
        # memory_hub omitted on purpose
    )
    run = await executor.run("Write a Python function fib(n).")
    assert run.status == RunStatus.COMPLETED

    trace_path = tmp_trace_dir / str(run.run_id) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    kinds = {(e["module"], e["kind"]) for e in events}
    assert ("planner", "memory_read") not in kinds
    # The one memory_write we DO see is the executor's finalize event, not an
    # episode write — distinguishable by the absence of `episode_id` in
    # outputs.
    ep_writes = [
        e
        for e in events
        if e["kind"] == "memory_write" and "episode_id" in e.get("outputs", {})
    ]
    assert ep_writes == []


async def test_clean_context_invariant_still_holds(tmp_trace_dir: Path) -> None:
    """Phase 1a must NOT relax the Verifier's Clean Context invariant.

    The Verifier still receives only contract + artifact + trace=[]. We assert
    this by inspecting Executor source for the locked call shape.
    """
    import inspect

    src = inspect.getsource(Executor._generate_verify_loop)
    assert "trace=[]" in src, (
        "Phase 1a must not pass trace events into Verifier. "
        "Span-level analysis is Phase 2."
    )
