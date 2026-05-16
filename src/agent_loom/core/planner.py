"""Planner — decomposes a user goal into a SprintContract.

Phase 0: one LLM call. No memory recall (that arrives in Phase 1), no failure
constraint injection (Phase 2). The output is a SprintContract instance plus the
single TraceEvent describing the call, both returned to the Executor which owns
trace persistence.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from agent_loom.config import get_settings
from agent_loom.core.trace import make_event
from agent_loom.core.types import SprintContract, TraceEvent
from agent_loom.llm import complete
from agent_loom.prompts import load_prompt


class Planner:
    def __init__(self, model: str | None = None) -> None:
        # Why nullable + late-bind: tests can pass model="fake" explicitly; real
        # callers let env-vars decide via Settings.
        self.model = model or get_settings().planner_model

    async def plan(self, *, run_id: UUID, user_goal: str) -> tuple[SprintContract, TraceEvent]:
        system = load_prompt("planner")
        started_at = datetime.utcnow()
        resp = await complete(model=self.model, system=system, user=user_goal, role="planner")
        ended_at = datetime.utcnow()

        # The LLM returns goal/non_goals/etc. but not run_id, so we splice it in.
        # Why splice rather than ask the LLM: run_id is harness-owned, not
        # LLM-owned. Avoid letting the model invent UUIDs.
        contract_data = dict(resp.parsed)
        contract_data["run_id"] = run_id
        contract = SprintContract(**contract_data)

        event = make_event(
            run_id=run_id,
            module="planner",
            kind="llm_call",
            started_at=started_at,
            ended_at=ended_at,
            inputs={"user_goal": user_goal, "model": self.model},
            outputs={"contract_id": str(contract.contract_id), "goal": contract.goal},
            cost_usd=resp.cost_usd,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            provider=resp.provider,
            model=resp.model,
        )
        return contract, event
