"""Phase 2 benchmark: repeat-failure-rate before/after memory.

Definition: for two batches of identical failing tasks, the rate of attempts
that fail with the same failure mode that was already in memory. Phase 2
should pull this rate down meaningfully on the second batch because the
Planner now reads past failures as `forbidden` constraints.

Usage:
    AGENT_LOOM_FAKE_LLM=1 python -m benchmarks.repeat_failure_rate --tasks 10

Output:
    Batch 1: forced failures (no memory leverage yet)
    Batch 2: same tasks; ideally fewer match Batch 1's failure category

This benchmark runs in fake mode by design: real LLM cost would dominate the
signal. The fake mode here switches Verifier behaviour at the controller
level rather than via env vars so the metric is reproducible.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.types import FailureCategory, RunStatus, VerifierJudgement
from agent_loom.core.verifier import Verifier
from agent_loom.memory.hub import MemoryHub


@dataclass
class BatchResult:
    """Aggregate metrics over a single batch of runs."""

    label: str
    total: int
    failures: int
    repeat_failures: int  # failures whose category matched a known prior failure

    @property
    def failure_rate(self) -> float:
        return 0.0 if self.total == 0 else self.failures / self.total

    @property
    def repeat_rate(self) -> float:
        return 0.0 if self.total == 0 else self.repeat_failures / self.total


class _SeedFailVerifier(Verifier):
    """Verifier that fails the first N attempts with a fixed failure category.

    Why a class and not a monkeypatch: the Executor / Generator have nothing
    to override; only the Verifier produces the failure verdict. Keeping
    state on the instance avoids a global counter.
    """

    def __init__(
        self, fail_count: int, category: FailureCategory = FailureCategory.SPEC_MISREAD
    ) -> None:
        super().__init__()
        self._fail_count = fail_count
        self._category = category
        self._seen = 0

    async def verify(self, *, contract, artifact, trace):  # type: ignore[override]
        judgement, event = await super().verify(
            contract=contract, artifact=artifact, trace=trace
        )
        self._seen += 1
        if self._seen <= self._fail_count:
            return (
                judgement.model_copy(
                    update={
                        "passed": False,
                        "score": 0.2,
                        "failure_category": self._category,
                        "reflection": (
                            "Forced failure for benchmark — would not satisfy "
                            "acceptance_criteria in the real Verifier."
                        ),
                    }
                ),
                event,
            )
        return judgement, event


def _classify_repeat(judgement: VerifierJudgement, known: set[str]) -> bool:
    """Did this judgement repeat a failure category we've already observed?"""
    if judgement.passed:
        return False
    if judgement.failure_category is None:
        return False
    return judgement.failure_category.value in known


async def _run_batch(
    *,
    label: str,
    n_tasks: int,
    memory_hub: MemoryHub,
    known_categories: set[str],
    fail_first_n: int,
) -> BatchResult:
    """Drive `n_tasks` identical failures and aggregate results.

    "Repeat" means the failure category was already in `known_categories`
    *before* this batch started. Within-batch self-collisions don't count —
    otherwise the very first batch would always report repeats once two tasks
    share the same forced category, which defeats the metric's purpose.
    """
    verifier = _SeedFailVerifier(fail_count=fail_first_n)
    planner = Planner(memory_hub=memory_hub)
    executor = Executor(
        planner=planner,
        generator=Generator(),
        verifier=verifier,
        max_iterations=1,
        memory_hub=memory_hub,
    )

    # Snapshot the prior-knowledge set at batch start. Mutations to
    # `known_categories` during this loop seed the next batch but must not
    # change what "repeat" means inside this one.
    pre_batch_known = frozenset(known_categories)

    failures = 0
    repeats = 0
    for i in range(n_tasks):
        run = await executor.run(f"benchmark task #{i} — write fib(n)")
        if run.status != RunStatus.COMPLETED:
            failures += 1
            # Use the last failure on this run to decide repeat. Sufficient
            # for fake-mode metrics; real-mode runs collect per-iteration.
            # We look up via the most recent failure episode in memory.
            failure_eps = await memory_hub.recall_failures(
                f"benchmark task #{i}", top_k=1
            )
            if failure_eps:
                cat = failure_eps[0].metadata.get("failure_category", "")
                if cat in pre_batch_known:
                    repeats += 1
                if cat:
                    known_categories.add(cat)

    return BatchResult(
        label=label, total=n_tasks, failures=failures, repeat_failures=repeats
    )


async def _main_async(tasks: int) -> None:
    memory_hub = MemoryHub.fake()
    known: set[str] = set()

    print("[bench] Batch 1 — all forced failures, no prior knowledge")
    b1 = await _run_batch(
        label="batch1",
        n_tasks=tasks,
        memory_hub=memory_hub,
        known_categories=known,
        fail_first_n=tasks,
    )
    print(
        f"  total={b1.total} failures={b1.failures} "
        f"failure_rate={b1.failure_rate:.2f} "
        f"repeat_rate={b1.repeat_rate:.2f}"
    )

    print("[bench] Batch 2 — same forced failures, memory now seeded")
    b2 = await _run_batch(
        label="batch2",
        n_tasks=tasks,
        memory_hub=memory_hub,
        known_categories=known,
        fail_first_n=tasks,
    )
    print(
        f"  total={b2.total} failures={b2.failures} "
        f"failure_rate={b2.failure_rate:.2f} "
        f"repeat_rate={b2.repeat_rate:.2f}"
    )

    delta = b1.repeat_rate - b2.repeat_rate
    print(f"[bench] repeat-rate delta (B1 - B2): {delta:+.2f}")


def main() -> None:
    # Force fake mode so the benchmark doesn't touch real LLM providers.
    os.environ.setdefault("AGENT_LOOM_FAKE_LLM", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_main_async(args.tasks))


if __name__ == "__main__":
    main()
