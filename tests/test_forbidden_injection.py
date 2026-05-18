"""Phase 2 DoD — failure episodes from a prior run inject into next contract.

The end-to-end story:
    Run 1: forced failure (SPEC_MISREAD)
    Run 2: similar goal → Planner pulls the failure → SprintContract.forbidden
           carries a "Past failure" entry the Generator/Verifier must respect.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.core.verifier import Verifier
from agent_loom.memory.hub import MemoryHub


class _SpecMisreadVerifier(Verifier):
    """Verifier that always reports SPEC_MISREAD.

    Why a subclass rather than monkeypatch: Verifier's class hierarchy is
    locked by `_assert_clean_context`, so subclassing is the canonical way to
    inject behaviour. The signature stays clean-context-compliant.
    """

    async def verify(self, *, contract, artifact, trace):  # type: ignore[override]
        judgement, event = await super().verify(
            contract=contract, artifact=artifact, trace=trace
        )
        failing = judgement.model_copy(
            update={
                "passed": False,
                "score": 0.2,
                "failure_category": FailureCategory.SPEC_MISREAD,
                "reflection": (
                    "Generator solved the wrong problem; the produced "
                    "function does not implement fib(n)."
                ),
            }
        )
        return failing, event


async def test_planner_injects_past_failures_into_forbidden(tmp_trace_dir: Path) -> None:
    """The headline DoD test for Phase 2's forbidden injection."""
    memory_hub = MemoryHub.fake()

    # Run 1 — forced failure plants a failure Episode and a reflection.
    executor1 = Executor(
        planner=Planner(memory_hub=memory_hub),
        generator=Generator(),
        verifier=_SpecMisreadVerifier(),
        memory_hub=memory_hub,
        max_iterations=1,
    )
    run1 = await executor1.run("Implement broken fib(n).")
    assert not run1.status.value == "completed", "Run 1 was supposed to fail"

    # Run 2 — Planner.plan() must surface the prior failure in `forbidden`.
    planner2 = Planner(memory_hub=memory_hub)
    contract, _ = await planner2.plan(
        run_id=uuid4(),
        user_goal="Implement broken fib(n) variation.",
    )
    assert any(
        f.startswith("Past failure") for f in contract.forbidden
    ), f"Forbidden list missing Past failure entry: {contract.forbidden}"


async def test_forbidden_entries_carry_failure_category_tag(tmp_trace_dir: Path) -> None:
    memory_hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=memory_hub),
        generator=Generator(),
        verifier=_SpecMisreadVerifier(),
        memory_hub=memory_hub,
        max_iterations=1,
    )
    await executor.run("Implement broken fib(n).")

    planner2 = Planner(memory_hub=memory_hub)
    contract, _ = await planner2.plan(run_id=uuid4(), user_goal="Implement fib(n).")

    # At least one forbidden entry must mention SPEC_MISREAD.
    matched = [f for f in contract.forbidden if "spec_misread" in f]
    assert matched, f"Expected spec_misread tag in forbidden, got {contract.forbidden}"


async def test_forbidden_injection_writes_two_memory_read_events(
    tmp_trace_dir: Path,
) -> None:
    """Planner now reads memory twice: general recall + failure-only recall."""
    memory_hub = MemoryHub.fake()
    executor = Executor(
        planner=Planner(memory_hub=memory_hub),
        generator=Generator(),
        verifier=_SpecMisreadVerifier(),
        memory_hub=memory_hub,
        max_iterations=1,
    )
    await executor.run("Implement broken fib(n).")

    planner2 = Planner(memory_hub=memory_hub)
    await planner2.plan(run_id=uuid4(), user_goal="x")
    side_events = planner2.drain_side_events()
    reads = [e for e in side_events if e.kind == "memory_read"]
    assert len(reads) == 2, f"Expected 2 memory_read events (general + failures), got {len(reads)}"
    filter_outputs = [
        e for e in reads if e.inputs.get("filter") == "failures"
    ]
    assert filter_outputs, "Missing the failure-filtered recall side event"


async def test_no_injection_when_memory_is_empty() -> None:
    """A fresh MemoryHub must produce contracts whose forbidden stays empty."""
    memory_hub = MemoryHub.fake()
    planner = Planner(memory_hub=memory_hub)
    contract, _ = await planner.plan(
        run_id=uuid4(), user_goal="Write a brand new function."
    )
    # Default LLM payload returns forbidden=[]; the injector adds nothing.
    assert contract.forbidden == []


async def test_no_injection_when_only_successes_recalled() -> None:
    """Past successes must not produce forbidden entries."""
    memory_hub = MemoryHub.fake()
    # Plant a successful episode directly.
    contract = SprintContract(
        run_id=uuid4(),
        goal="Compute squares of integers.",
        acceptance_criteria=["sq(3) == 9"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id,
        kind="text",
        content="def sq(n): return n*n",
    )
    judgement = VerifierJudgement(
        artifact_id=artifact.artifact_id,
        passed=True,
        score=0.95,
        failure_category=None,
        reflection="Looks fine.",
    )
    await memory_hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )

    planner = Planner(memory_hub=memory_hub)
    contract2, _ = await planner.plan(
        run_id=uuid4(), user_goal="Compute squares of integers."
    )
    # All recalled episodes are passes; failure-recall returns empty;
    # forbidden therefore stays empty.
    assert all(
        not f.startswith("Past failure") for f in contract2.forbidden
    ), contract2.forbidden
