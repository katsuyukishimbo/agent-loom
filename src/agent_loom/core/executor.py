"""Executor — the only stateful module.

Owns:
- The trace file (TraceWriter). All filesystem writes pass through here.
- The run-level budget (max_cost_usd, total_cost_usd).
- The retry loop: plan once, then generate/verify until passed OR
  max_iterations reached OR budget blown.
- (Phase 1a) The memory write call after the Verifier judges each attempt.
  The Executor is the only module allowed to write to MemoryHub. Planner is
  read-only; Generator and Verifier never touch memory. This mirrors
  "writes single-threaded" at the memory layer.

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
from agent_loom.memory.hub import MemoryHub


class BudgetExceeded(RuntimeError):
    """Raised internally to short-circuit the loop when over budget."""


class Executor:
    def __init__(
        self,
        planner: Planner,
        generator: Generator,
        verifier: Verifier,
        max_iterations: int = 3,
        memory_hub: MemoryHub | None = None,
    ) -> None:
        self.planner = planner
        self.generator = generator
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.memory_hub = memory_hub

    async def run(self, user_goal: str) -> Run:
        settings = get_settings()
        run = Run(run_id=uuid4(), user_goal=user_goal, status=RunStatus.PLANNING)
        trace_writer = TraceWriter(run.run_id, base_dir=settings.trace_dir)

        try:
            contract, plan_event = await self.planner.plan(
                run_id=run.run_id, user_goal=user_goal
            )
            # Drain memory_read events (Phase 1a) that the Planner buffered
            # during recall. Writing them here keeps trace persistence single-
            # threaded inside the Executor.
            for side_event in getattr(self.planner, "drain_side_events", lambda: [])():
                trace_writer.write(side_event)
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

        Why re-using the same SprintContract across iterations: Phase 0/1a keep
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

            # Memory write happens AFTER the Verifier's judgement is on disk so
            # a crash mid-write doesn't lose the judgement, only the memory
            # record. Done before the budget check so we still capture the
            # episode even if the next iteration would push us over budget.
            written_episode = None
            if self.memory_hub is not None:
                written_episode = await self._write_memory_episode(
                    contract=contract,
                    artifact=artifact,
                    judgement=judgement,
                    trace_writer=trace_writer,
                    run=run,
                )

            # Reflective Compaction (Phase 2). Failures trigger a separate LLM
            # call that distills "what went wrong" into a high-importance
            # Episode the next Planner can read. We do this BEFORE the budget
            # check so a tight budget still gets the reflection on disk.
            if (
                not judgement.passed
                and self.memory_hub is not None
                and written_episode is not None
            ):
                run.status = RunStatus.REFLECTING
                await self._write_reflection_episode(
                    contract=contract,
                    artifact=artifact,
                    judgement=judgement,
                    source_episode=written_episode,
                    trace_writer=trace_writer,
                    run=run,
                )

            self._check_budget(run, contract)

            if judgement.passed:
                return artifact, judgement

        # If we got here, the loop exhausted without passing. Return the last
        # pair so callers can inspect the failed attempt.
        assert artifact is not None and judgement is not None  # for type checker
        return artifact, judgement

    async def _write_memory_episode(
        self,
        *,
        contract: SprintContract,
        artifact: Artifact,
        judgement: VerifierJudgement,
        trace_writer: TraceWriter,
        run: Run,
    ):
        """Write a single Episode for this attempt and record a trace event.

        Why a private helper: the call has three side effects (LLM call for
        importance, store insert, trace write) and bundling them keeps the
        loop body readable.

        Returns the written `Episode` so the caller can attach a derived
        reflection edge in the same iteration.
        """
        assert self.memory_hub is not None  # for type checker
        started_at = datetime.utcnow()
        episode = await self.memory_hub.write_from_judgement(
            contract=contract, artifact=artifact, judgement=judgement
        )
        ended_at = datetime.utcnow()
        trace_writer.write(
            make_event(
                run_id=run.run_id,
                module="executor",
                kind="memory_write",
                started_at=started_at,
                ended_at=ended_at,
                inputs={
                    "contract_id": str(contract.contract_id),
                    "artifact_id": str(artifact.artifact_id),
                },
                outputs={
                    "episode_id": str(episode.episode_id),
                    "importance": episode.importance,
                    "passed": judgement.passed,
                },
            )
        )
        return episode

    async def _write_reflection_episode(
        self,
        *,
        contract: SprintContract,
        artifact: Artifact,
        judgement: VerifierJudgement,
        source_episode,
        trace_writer: TraceWriter,
        run: Run,
    ) -> None:
        """Run Reflective Compaction and persist the resulting Episode.

        Why a second helper rather than folding into `_write_memory_episode`:
        the reflection call is a distinct LLM call (different role, different
        cost) and we want a separate `memory_write` trace event for it so the
        replay UI can show the failure → reflection pairing.
        """
        assert self.memory_hub is not None  # for type checker
        started_at = datetime.utcnow()
        reflection_ep = await self.memory_hub.write_reflection(
            contract=contract,
            artifact=artifact,
            judgement=judgement,
            source_episode=source_episode,
        )
        ended_at = datetime.utcnow()
        trace_writer.write(
            make_event(
                run_id=run.run_id,
                module="executor",
                kind="memory_write",
                started_at=started_at,
                ended_at=ended_at,
                inputs={
                    "contract_id": str(contract.contract_id),
                    "source_episode_id": str(source_episode.episode_id),
                    "kind": "reflection",
                },
                outputs={
                    "episode_id": str(reflection_ep.episode_id),
                    "importance": reflection_ep.importance,
                    "failure_category": (
                        judgement.failure_category.value
                        if judgement.failure_category
                        else None
                    ),
                },
            )
        )

    def _check_budget(self, run: Run, contract: SprintContract) -> None:
        if run.total_cost_usd >= contract.max_cost_usd:
            raise BudgetExceeded(
                f"Run {run.run_id} exceeded budget: "
                f"${run.total_cost_usd:.4f} >= ${contract.max_cost_usd:.4f}"
            )
