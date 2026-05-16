"""Executor — the only stateful module.

Owns:
- The trace file (TraceWriter). All filesystem writes pass through here.
- The run-level budget (max_cost_usd, total_cost_usd).
- The retry loop: plan once, then generate/verify until passed OR
  max_iterations reached OR budget blown.

Why a single owner for writes: it keeps the "writes single-threaded"
invariant trivial — the other modules return TraceEvents as values, this one
appends them to disk.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from agent_loom.config import get_settings
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.trace import TraceWriter, make_event
from agent_loom.core.types import (
    Artifact,
    Run,
    RunStatus,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.core.verifier import Verifier


class BudgetExceeded(RuntimeError):
    """Raised internally to short-circuit the loop when over budget."""


class Executor:
    def __init__(
        self,
        planner: Planner,
        generator: Generator,
        verifier: Verifier,
        max_iterations: int = 3,
    ) -> None:
        self.planner = planner
        self.generator = generator
        self.verifier = verifier
        self.max_iterations = max_iterations

    async def run(self, user_goal: str) -> Run:
        settings = get_settings()
        run = Run(run_id=uuid4(), user_goal=user_goal, status=RunStatus.PLANNING)
        trace_writer = TraceWriter(run.run_id, base_dir=settings.trace_dir)

        try:
            contract, plan_event = await self.planner.plan(
                run_id=run.run_id, user_goal=user_goal
            )
            trace_writer.write(plan_event)
            run.total_cost_usd += plan_event.cost_usd
            self._check_budget(run, contract)

            artifact, judgement = await self._generate_verify_loop(
                run=run, contract=contract, trace_writer=trace_writer
            )

            run.final_artifact_id = artifact.artifact_id
            run.status = RunStatus.COMPLETED if judgement.passed else RunStatus.FAILED
            trace_writer.save_artifact(artifact.content)
        except BudgetExceeded:
            run.status = RunStatus.ESCALATED
        finally:
            run.ended_at = datetime.utcnow()
            trace_writer.write(
                make_event(
                    run_id=run.run_id,
                    module="executor",
                    kind="memory_write",
                    started_at=run.started_at,
                    ended_at=run.ended_at,
                    inputs={"user_goal": user_goal},
                    outputs={
                        "status": run.status.value,
                        "iterations": run.iterations,
                        "total_cost_usd": run.total_cost_usd,
                    },
                )
            )
            trace_writer.save_summary(run.model_dump(mode="json"))

        return run

    async def _generate_verify_loop(
        self,
        *,
        run: Run,
        contract: SprintContract,
        trace_writer: TraceWriter,
    ) -> tuple[Artifact, VerifierJudgement]:
        """Re-try generate→verify until passed, exhausted, or over budget.

        Why re-using the same SprintContract across iterations: Phase 0 keeps
        the loop minimal. Reflective re-planning (a new contract with
        `forbidden` entries derived from failure) is a Phase 2 feature.
        """
        artifact: Artifact | None = None
        judgement: VerifierJudgement | None = None
        for _ in range(self.max_iterations):
            run.iterations += 1
            run.status = RunStatus.GENERATING
            artifact, gen_event = await self.generator.generate(contract=contract)
            trace_writer.write(gen_event)
            run.total_cost_usd += gen_event.cost_usd
            self._check_budget(run, contract)

            run.status = RunStatus.VERIFYING
            judgement, ver_event = await self.verifier.verify(
                contract=contract,
                artifact=artifact,
                # Pass an empty trace slice in Phase 0; span-level analysis is Phase 2.
                trace=[],
            )
            trace_writer.write(ver_event)
            run.total_cost_usd += ver_event.cost_usd
            self._check_budget(run, contract)

            if judgement.passed:
                return artifact, judgement

        # If we got here, the loop exhausted without passing. Return the last
        # pair so callers can inspect the failed attempt.
        assert artifact is not None and judgement is not None  # for type checker
        return artifact, judgement

    def _check_budget(self, run: Run, contract: SprintContract) -> None:
        if run.total_cost_usd >= contract.max_cost_usd:
            raise BudgetExceeded(
                f"Run {run.run_id} exceeded budget: "
                f"${run.total_cost_usd:.4f} >= ${contract.max_cost_usd:.4f}"
            )
