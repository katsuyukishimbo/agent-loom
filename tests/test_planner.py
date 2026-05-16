"""Planner unit tests (fake mode)."""

from __future__ import annotations

from uuid import uuid4

from agent_loom.core.planner import Planner
from agent_loom.core.types import SprintContract, TraceEvent


async def test_plan_returns_contract_and_event() -> None:
    planner = Planner(model="claude-opus-4-7")
    run_id = uuid4()
    contract, event = await planner.plan(
        run_id=run_id,
        user_goal="Write a Python function fib(n).",
    )
    assert isinstance(contract, SprintContract)
    assert isinstance(event, TraceEvent)
    assert contract.run_id == run_id
    assert contract.acceptance_criteria, "Planner must produce ≥1 acceptance criterion"


async def test_event_metadata_is_populated() -> None:
    planner = Planner(model="claude-opus-4-7")
    _, event = await planner.plan(run_id=uuid4(), user_goal="x")
    assert event.module == "planner"
    assert event.kind == "llm_call"
    assert event.provider == "fake"
    assert event.ended_at >= event.started_at
