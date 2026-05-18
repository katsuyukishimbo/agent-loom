"""Planner — decomposes a user goal into a SprintContract.

Phase 1a wired optional MemoryHub for recall and threaded a one-paragraph
recall preface through the persona field. Phase 2 adds:

- **Failure constraint injection**: a second recall call narrowed to
  failure-tagged Episodes (`recall_failures`) produces strings that go
  straight into `SprintContract.forbidden`. The Generator sees them; the
  Verifier reads `forbidden` and grades accordingly.

Public API:
    plan(run_id, user_goal) -> (SprintContract, TraceEvent)

Side-channel for the Executor:
    self.drain_side_events() -> list[TraceEvent]   # memory_read events
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from agent_loom.config import get_settings
from agent_loom.core.trace import make_event
from agent_loom.core.types import SprintContract, TraceEvent
from agent_loom.llm import complete
from agent_loom.memory.hub import MemoryHub
from agent_loom.prompts import load_prompt


class Planner:
    def __init__(
        self,
        model: str | None = None,
        memory_hub: MemoryHub | None = None,
        recall_top_k: int = 3,
        forbidden_top_k: int = 3,
    ) -> None:
        # Why nullable + late-bind: tests can pass model="fake" explicitly; real
        # callers let env-vars decide via Settings.
        self.model = model or get_settings().planner_model
        self.memory_hub = memory_hub
        self.recall_top_k = recall_top_k
        # Failure constraint injection — pulled with a separate, narrower
        # recall so the persona preface and the forbidden list don't fight
        # for the same ranked slots.
        self.forbidden_top_k = forbidden_top_k
        # Side-channel for non-LLM TraceEvents the Executor must persist. We
        # keep the public plan() return shape as `(contract, event)` so all
        # existing tests and the executor subclass-override pattern keep
        # working unchanged.
        self._side_events: list[TraceEvent] = []

    def drain_side_events(self) -> list[TraceEvent]:
        """Return and clear buffered TraceEvents from the most recent plan().

        Called by the Executor after each plan() to persist memory_read events.
        Idempotent: calling twice yields an empty list the second time.
        """
        events, self._side_events = self._side_events, []
        return events

    async def plan(
        self, *, run_id: UUID, user_goal: str
    ) -> tuple[SprintContract, TraceEvent]:
        # Reset the side channel at the top of each call. If a prior plan()
        # left events un-drained, dropping them here is the right call —
        # otherwise we'd attribute them to the wrong run.
        self._side_events = []

        recall_preface = ""
        forbidden_injections: list[str] = []
        if self.memory_hub is not None:
            recall_started = datetime.utcnow()
            episodes = await self.memory_hub.recall(user_goal, top_k=self.recall_top_k)
            recall_ended = datetime.utcnow()
            recall_preface = MemoryHub.format_recall_for_persona(episodes)
            self._side_events.append(
                make_event(
                    run_id=run_id,
                    module="planner",
                    kind="memory_read",
                    started_at=recall_started,
                    ended_at=recall_ended,
                    inputs={"query": user_goal, "top_k": self.recall_top_k},
                    outputs={
                        "episode_ids": [str(e.episode_id) for e in episodes],
                        "count": len(episodes),
                    },
                )
            )

            # Failure constraint injection (Phase 2 core). We deliberately
            # query the store twice — once for general recall, once for
            # failure-only — rather than filtering the first result list:
            # general recall over-fetches strong successes which crowd out
            # the smaller failure population.
            forbid_started = datetime.utcnow()
            failures = await self.memory_hub.recall_failures(
                user_goal, top_k=self.forbidden_top_k
            )
            forbid_ended = datetime.utcnow()
            forbidden_injections = MemoryHub.format_failures_for_forbidden(failures)
            self._side_events.append(
                make_event(
                    run_id=run_id,
                    module="planner",
                    kind="memory_read",
                    started_at=forbid_started,
                    ended_at=forbid_ended,
                    inputs={
                        "query": user_goal,
                        "top_k": self.forbidden_top_k,
                        "filter": "failures",
                    },
                    outputs={
                        "episode_ids": [str(e.episode_id) for e in failures],
                        "count": len(failures),
                    },
                )
            )

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

        if recall_preface:
            # Prepend recall summary to the persona so the Generator inherits
            # the context without us touching types.py. The default persona
            # ("pragmatic senior engineer...") stays at the end.
            contract = contract.model_copy(
                update={"persona": f"{recall_preface}\n\n{contract.persona}"}
            )

        if forbidden_injections:
            # Extend, don't overwrite: the LLM may already have emitted its
            # own forbidden entries based on the goal text. De-duplicate the
            # union so repeated runs don't accumulate identical strings.
            merged: list[str] = list(contract.forbidden)
            for item in forbidden_injections:
                if item not in merged:
                    merged.append(item)
            contract = contract.model_copy(update={"forbidden": merged})

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
