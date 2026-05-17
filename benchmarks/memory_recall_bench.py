"""Phase 1b benchmark: PgvectorEpisodicStore.recall() latency on 1k episodes.

Usage:
    ./scripts/dev_db_up.sh && alembic upgrade head
    AGENT_LOOM_FAKE_LLM=1 python -m benchmarks.memory_recall_bench

Output:
    seed:    wall time to insert N episodes
    recall:  mean, p50, p95 over R iterations of recall(top_k=5)

Phase 1b DoD: mean recall < 200ms on the dev laptop.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from datetime import datetime, timedelta

from agent_loom.memory.embeddings import FakeEmbeddingService
from agent_loom.memory.store import Episode
from agent_loom.memory.store_pg import (
    PgvectorEpisodicStore,
    _default_database_url,
    reachable,
)


async def _seed(store: PgvectorEpisodicStore, n: int) -> None:
    """Insert `n` synthetic episodes.

    Each episode gets a deterministic FakeEmbeddingService vector for its
    content, plus a backdated `last_referenced_at` distributed over the past
    60 days so recency scoring has something to spread.
    """
    svc = FakeEmbeddingService()
    now = datetime.utcnow()

    # Batch the inserts to avoid 1k round-trips; psycopg's execute_many would
    # be ideal but we keep this simple and rely on the connection-per-call
    # cost being negligible vs total wall time at N=1000.
    for i in range(n):
        content = f"synthetic episode #{i}: fib variant {i % 17}"
        emb = await svc.embed(content)
        age = timedelta(days=(i % 60))
        await store.write(
            Episode(
                content=content,
                importance=1.0 + (i % 10),
                embedding=emb,
                last_referenced_at=now - age,
            )
        )


async def _measure(
    store: PgvectorEpisodicStore, query: list[float], iterations: int
) -> dict[str, float]:
    """Time `iterations` calls to recall(top_k=5)."""
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await store.recall(query, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    return {
        "n": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "max_ms": latencies[-1],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reset", action="store_true", help="TRUNCATE before seeding")
    args = parser.parse_args()

    if not await reachable(_default_database_url()):
        raise SystemExit(
            "DATABASE_URL is not reachable. Run ./scripts/dev_db_up.sh && "
            "alembic upgrade head first."
        )

    store = PgvectorEpisodicStore()
    if args.reset:
        await store.truncate()

    existing = await store.count()
    needed = max(0, args.episodes - existing)
    if needed:
        print(f"[bench] seeding {needed} episodes (existing={existing})...")
        t0 = time.perf_counter()
        await _seed(store, needed)
        print(f"[bench] seed wall: {(time.perf_counter() - t0):.2f}s")

    svc = FakeEmbeddingService()
    query = await svc.embed("synthetic episode #42: fib variant 8")

    print(f"[bench] timing recall over {args.iterations} iterations...")
    stats = await _measure(store, query, args.iterations)
    print(
        f"[bench] recall  n={stats['n']:>4}  "
        f"mean={stats['mean_ms']:>6.2f}ms  "
        f"p50={stats['p50_ms']:>6.2f}ms  "
        f"p95={stats['p95_ms']:>6.2f}ms  "
        f"max={stats['max_ms']:>6.2f}ms"
    )
    target = 200.0
    verdict = "PASS" if stats["mean_ms"] < target else "FAIL"
    print(f"[bench] Phase 1b DoD (<{target}ms mean): {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
