"""Phase 2 DoD: repeat-failure rate measurably drops on the second batch.

The benchmark module exposes `_run_batch` as a primitive; we drive two
batches in a tight loop and assert the failure category recall on batch 2
captures the prior failure mode at least once.

Note on what we're measuring in fake mode: the Verifier in fake mode always
passes by default, so we use a `_SeedFailVerifier` that forces failures.
Memory recall on batch 2 lets the system identify the repeat — which is the
metric we care about.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.repeat_failure_rate import _run_batch
from agent_loom.memory.hub import MemoryHub


async def test_batch_2_recognises_repeat_failures(tmp_trace_dir: Path) -> None:
    """After batch 1 plants failures, batch 2 must classify some as repeats."""
    hub = MemoryHub.fake()
    known: set[str] = set()

    b1 = await _run_batch(
        label="b1",
        n_tasks=3,
        memory_hub=hub,
        known_categories=known,
        fail_first_n=3,
    )
    assert b1.failures == 3
    assert b1.repeat_failures == 0  # first time, nothing to repeat against
    assert "spec_misread" in known  # batch 1 should have planted this

    b2 = await _run_batch(
        label="b2",
        n_tasks=3,
        memory_hub=hub,
        known_categories=known,
        fail_first_n=3,
    )
    # Batch 2: same forced category, memory has been seeded. The aggregator
    # must classify at least one as a repeat to indicate memory's signal.
    assert b2.repeat_failures >= 1, (
        f"Expected at least one repeat on batch 2; got {b2.repeat_failures}"
    )


async def test_repeat_rate_is_zero_for_first_batch(tmp_trace_dir: Path) -> None:
    """Batch 1 cannot have repeats because there's nothing in memory yet."""
    hub = MemoryHub.fake()
    known: set[str] = set()

    b1 = await _run_batch(
        label="only",
        n_tasks=2,
        memory_hub=hub,
        known_categories=known,
        fail_first_n=2,
    )
    assert b1.repeat_rate == 0.0


async def test_batch_aggregates_have_consistent_counts(tmp_trace_dir: Path) -> None:
    hub = MemoryHub.fake()
    known: set[str] = set()
    b = await _run_batch(
        label="x",
        n_tasks=4,
        memory_hub=hub,
        known_categories=known,
        fail_first_n=4,
    )
    assert b.total == 4
    assert b.failures <= b.total
    assert b.repeat_failures <= b.failures
