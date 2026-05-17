"""PgvectorEpisodicStore tests.

These tests need a live Postgres+pgvector on the URL named by `DATABASE_URL`
(default: the docker-compose one). When the DB isn't reachable they skip —
they do not fail — so CI without docker stays green.

Bring the DB up first:
    ./scripts/dev_db_up.sh && alembic upgrade head
    pytest tests/test_memory_store_pg.py -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from agent_loom.memory.embeddings import FakeEmbeddingService
from agent_loom.memory.store import (
    Episode,
    EpisodicStore,
)
from agent_loom.memory.store_pg import (
    PgvectorEpisodicStore,
    _default_database_url,
    reachable,
)


def _db_available() -> bool:
    """Synchronous wrapper so we can use it as a pytest marker condition."""
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        reachable(_default_database_url())
    )


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="Postgres+pgvector not reachable; run ./scripts/dev_db_up.sh && alembic upgrade head",
)


@pytest.fixture
async def store() -> PgvectorEpisodicStore:
    """A truncated PgvectorEpisodicStore per test (clean slate)."""
    s = PgvectorEpisodicStore()
    await s.truncate()
    return s


async def test_store_satisfies_protocol(store: PgvectorEpisodicStore) -> None:
    """Runtime-checkable Protocol — Pg store should match the same shape."""
    assert isinstance(store, EpisodicStore)


async def test_write_and_list(store: PgvectorEpisodicStore) -> None:
    svc = FakeEmbeddingService()
    emb = await svc.embed("alpha")
    ep = await store.write(Episode(content="alpha", importance=5.0, embedding=emb))

    items = await store.list_all()
    assert len(items) == 1
    assert items[0].episode_id == ep.episode_id
    assert items[0].content == "alpha"
    assert items[0].embedding is not None
    assert len(items[0].embedding) == 1536


async def test_write_is_upsert(store: PgvectorEpisodicStore) -> None:
    """Writing the same episode twice should not raise — ON CONFLICT UPDATE."""
    svc = FakeEmbeddingService()
    emb = await svc.embed("alpha")
    ep = Episode(content="alpha", importance=5.0, embedding=emb)
    await store.write(ep)
    # Mutate and write again with the same id.
    ep_v2 = ep.model_copy(update={"content": "alpha v2", "importance": 7.0})
    await store.write(ep_v2)

    items = await store.list_all()
    assert len(items) == 1
    assert items[0].content == "alpha v2"
    assert items[0].importance == 7.0


async def test_recall_returns_top_k(store: PgvectorEpisodicStore) -> None:
    svc = FakeEmbeddingService()
    contents = ["fib(n) implementation", "sort a list", "binary search recipe"]
    for c in contents:
        emb = await svc.embed(c)
        await store.write(Episode(content=c, importance=7.0, embedding=emb))

    query = await svc.embed("fib(n) implementation")
    hits = await store.recall(query, top_k=2)
    assert len(hits) == 2
    assert hits[0].content == "fib(n) implementation"


async def test_recall_increments_references_count(
    store: PgvectorEpisodicStore,
) -> None:
    svc = FakeEmbeddingService()
    emb = await svc.embed("x")
    await store.write(Episode(content="x", importance=5.0, embedding=emb))

    query = await svc.embed("x")
    hits = await store.recall(query, top_k=1)
    assert hits[0].references_count == 1

    hits2 = await store.recall(query, top_k=1)
    assert hits2[0].references_count == 2

    # Persisted on disk?
    stored = await store.list_all()
    assert stored[0].references_count == 2


async def test_recall_updates_last_referenced_at(store: PgvectorEpisodicStore) -> None:
    svc = FakeEmbeddingService()
    emb = await svc.embed("x")
    older = datetime.utcnow() - timedelta(days=10)
    await store.write(
        Episode(
            content="x",
            importance=5.0,
            embedding=emb,
            last_referenced_at=older,
        )
    )
    now = datetime.utcnow()
    query = await svc.embed("x")
    hits = await store.recall(query, top_k=1, now=now)
    # Allow microsecond-precision drift via Postgres timestamp rounding.
    delta = abs((hits[0].last_referenced_at - now).total_seconds())
    assert delta < 1.0


async def test_recall_orders_by_rir(store: PgvectorEpisodicStore) -> None:
    """Higher-recency same-relevance episode wins.

    This mirrors the InMemoryEpisodicStore test of the same name to prove
    the two stores agree on the ordering rule.
    """
    svc = FakeEmbeddingService()
    now = datetime.utcnow()

    # Stale: perfect match by content but 100 days old (recency = 0.1)
    stale_emb = await svc.embed("target")
    await store.write(
        Episode(
            content="target",
            importance=10.0,
            embedding=stale_emb,
            last_referenced_at=now - timedelta(days=100),
        )
    )
    # Fresh + same content
    fresh_emb = await svc.embed("target")
    fresh = await store.write(
        Episode(
            content="target",
            importance=10.0,
            embedding=fresh_emb,
            last_referenced_at=now,
        )
    )

    query = await svc.embed("target")
    hits = await store.recall(query, top_k=2, now=now)
    assert hits[0].episode_id == fresh.episode_id  # fresh wins


async def test_recall_skips_embeddingless_episodes(
    store: PgvectorEpisodicStore,
) -> None:
    svc = FakeEmbeddingService()
    await store.write(Episode(content="no embedding", importance=5.0))
    emb = await svc.embed("with embedding")
    await store.write(
        Episode(content="with embedding", importance=5.0, embedding=emb)
    )

    query = await svc.embed("with embedding")
    hits = await store.recall(query, top_k=10)
    assert len(hits) == 1
    assert hits[0].content == "with embedding"


async def test_count(store: PgvectorEpisodicStore) -> None:
    assert await store.count() == 0
    svc = FakeEmbeddingService()
    emb = await svc.embed("x")
    await store.write(Episode(content="x", importance=5.0, embedding=emb))
    assert await store.count() == 1


async def test_metadata_roundtrip(store: PgvectorEpisodicStore) -> None:
    """Pydantic dict[str,str] field must come back intact through jsonb."""
    svc = FakeEmbeddingService()
    emb = await svc.embed("with metadata")
    ep = await store.write(
        Episode(
            content="with metadata",
            importance=5.0,
            embedding=emb,
            metadata={"run_id": "abc", "passed": "true"},
            source="executor",
        )
    )
    stored = await store.list_all()
    assert stored[0].metadata == ep.metadata
    assert stored[0].source == "executor"
