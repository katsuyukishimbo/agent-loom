"""End-to-end Executor tests (fake mode).

These cover the Phase 0 DoD:
- ≥4 trace events on disk (planner + generator + verifier + executor summary)
- artifact.txt persisted
- summary.json persisted
- budget gate halts the loop
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.types import RunStatus, SprintContract
from agent_loom.core.verifier import Verifier


def _build_executor() -> Executor:
    return Executor(planner=Planner(), generator=Generator(), verifier=Verifier())


async def test_run_completes_in_fake_mode(tmp_trace_dir: Path) -> None:
    executor = _build_executor()
    run = await executor.run("Write a Python function fib(n).")
    assert run.status == RunStatus.COMPLETED
    assert run.iterations == 1
    assert run.final_artifact_id is not None


async def test_run_writes_trace_artifact_and_summary(tmp_trace_dir: Path) -> None:
    executor = _build_executor()
    run = await executor.run("Write a Python function fib(n).")
    run_dir = tmp_trace_dir / str(run.run_id)
    trace_path = run_dir / "trace.jsonl"
    artifact_path = run_dir / "artifact.txt"
    summary_path = run_dir / "summary.json"

    assert trace_path.exists()
    assert artifact_path.exists()
    assert summary_path.exists()

    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    # Phase 0 DoD: ≥ 4 events. We expect planner, generator, verifier, executor.
    assert len(events) >= 4
    modules_seen = {e["module"] for e in events}
    assert {"planner", "generator", "verifier", "executor"} <= modules_seen


async def test_artifact_text_contains_fib_definition(tmp_trace_dir: Path) -> None:
    executor = _build_executor()
    run = await executor.run("Write a Python function fib(n).")
    artifact_path = tmp_trace_dir / str(run.run_id) / "artifact.txt"
    assert "def fib" in artifact_path.read_text(encoding="utf-8")


async def test_budget_exceeded_escalates(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-USD budget should always escalate, even in fake mode (cost is $0)."""

    class ZeroBudgetPlanner(Planner):
        async def plan(self, *, run_id, user_goal):  # type: ignore[override]
            contract, event = await super().plan(run_id=run_id, user_goal=user_goal)
            tight = contract.model_copy(update={"max_cost_usd": 0.0})
            return tight, event

    # Force at least a sub-cent cost so the budget gate actually trips.
    # Why: fake mode has cost_usd = 0.0 by design. We patch the event after
    # planning to simulate a real provider's first call burning the budget.
    real_plan = ZeroBudgetPlanner().plan

    async def _expensive_plan(*, run_id, user_goal):
        contract, event = await real_plan(run_id=run_id, user_goal=user_goal)
        event_with_cost = event.model_copy(update={"cost_usd": 0.01})
        return contract, event_with_cost

    planner = ZeroBudgetPlanner()
    planner.plan = _expensive_plan  # type: ignore[method-assign]

    executor = Executor(planner=planner, generator=Generator(), verifier=Verifier())
    run = await executor.run("Write fib.")
    assert run.status == RunStatus.ESCALATED


async def test_loop_exhausts_when_verifier_keeps_failing(tmp_trace_dir: Path) -> None:
    """If the Verifier never passes, the Executor returns the last attempt with FAILED status.

    Why: this exercises the "loop exhaustion" branch — the alternative escape
    hatch to BudgetExceeded. Real-world Phase 0 use cases hit it when the
    Generator can't satisfy the criteria within max_iterations.
    """

    class AlwaysFailVerifier(Verifier):
        async def verify(self, *, contract, artifact, trace):  # type: ignore[override]
            judgement, event = await super().verify(
                contract=contract, artifact=artifact, trace=trace
            )
            failing = judgement.model_copy(update={"passed": False, "score": 0.2})
            return failing, event

    executor = Executor(
        planner=Planner(),
        generator=Generator(),
        verifier=AlwaysFailVerifier(),
        max_iterations=2,
    )
    run = await executor.run("Write fib.")
    assert run.status == RunStatus.FAILED
    assert run.iterations == 2


async def test_summary_json_round_trips(tmp_trace_dir: Path) -> None:
    executor = _build_executor()
    run = await executor.run("Write a Python function fib(n).")
    summary_path = tmp_trace_dir / str(run.run_id) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["iterations"] == 1


async def test_executor_writes_summary_event_last(tmp_trace_dir: Path) -> None:
    """The executor's finalize event should be the last line in trace.jsonl."""
    executor = _build_executor()
    run = await executor.run("Write a Python function fib(n).")
    trace_path = tmp_trace_dir / str(run.run_id) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert events[-1]["module"] == "executor"


# Sanity: Phase 0 promises a single iteration in the happy path.
async def test_single_iteration_in_fake_happy_path(tmp_trace_dir: Path) -> None:
    executor = _build_executor()
    run = await executor.run("Write a Python function fib(n).")
    assert run.iterations == 1


# Sanity: Verifier got an empty trace slice in Phase 0 (span analysis is Phase 2).
async def test_clean_context_invariant_holds_at_runtime(tmp_trace_dir: Path) -> None:
    """The harness must never pass Generator trace events into Verifier in Phase 0.

    We assert this indirectly by checking the executor's call site signature
    via inspection. If a future change adds a 'history=' arg, the test fails.
    """
    import inspect

    src = inspect.getsource(Executor._generate_verify_loop)
    assert "trace=[]" in src, (
        "Phase 0 must pass an empty trace slice to Verifier. "
        "Span-level analysis is Phase 2."
    )


# Coverage helper: SprintContract is imported for type clarity above; ensure it's usable.
def test_sprint_contract_is_importable() -> None:
    assert SprintContract is not None
