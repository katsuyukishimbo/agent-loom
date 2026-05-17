"""InMemoryEpisodicStore + R × I × R scoring tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent_loom.memory.embeddings import FakeEmbeddingService
from agent_loom.memory.store import (
    Episode,
    EpisodicStore,
    InMemoryEpisodicStore,
    importance_normalized,
    recency_score,
    relevance_score,
    rir_score,
)

# --- Recency: stepwise decay -------------------------------------------


def test_recency_buckets_cover_all_steps() -> None:
    """Each documented bucket boundary returns the expected step value."""
    now = datetime.utcnow()
    cases = [
        (timedelta(hours=1), 1.0),
        (timedelta(hours=23), 1.0),
        (timedelta(days=2), 0.8),
        (timedelta(days=10), 0.6),
        (timedelta(days=20), 0.5),
        (timedelta(days=60), 0.3),
        (timedelta(days=120), 0.1),
    ]
    for age, expected in cases:
        ep = Episode(
            content="x", importance=5.0, last_referenced_at=now - age
        )
        assert recency_score(ep, now=now) == expected, (
            f"age={age} expected={expected} got={recency_score(ep, now=now)}"
        )


# --- Importance --------------------------------------------------------


def test_importance_normalised_endpoints() -> None:
    """0 -> 0, 10 -> 1, clamp on out-of-range values."""
    e0 = Episode(content="x", importance=0.0)
    e10 = Episode(content="x", importance=10.0)
    assert importance_normalized(e0) == 0.0
    assert importance_normalized(e10) == 1.0


# --- Relevance ---------------------------------------------------------


async def test_relevance_score_zero_for_missing_embedding() -> None:
    """An episode written without an embedding can't compete on relevance."""
    ep = Episode(content="x", importance=5.0, embedding=None)
    svc = FakeEmbeddingService()
    query = await svc.embed("query")
    assert relevance_score(ep, query) == 0.0


async def test_relevance_score_identical_query() -> None:
    """Same content -> same embedding -> relevance ~ 1.0."""
    svc = FakeEmbeddingService()
    content = "Write fib(n)."
    emb = await svc.embed(content)
    ep = Episode(content=content, importance=5.0, embedding=emb)
    query = await svc.embed(content)
    assert relevance_score(ep, query) == pytest.approx(1.0, abs=1e-9)


async def test_relevance_score_clamped_to_zero_for_negative_cosine() -> None:
    """Opposite-direction embeddings should not produce negative relevance."""
    ep = Episode(content="x", importance=5.0, embedding=[1.0, 0.0])
    query = [-1.0, 0.0]
    assert relevance_score(ep, query) == 0.0


# --- Composed R × I × R ------------------------------------------------


async def test_rir_score_combines_three_factors() -> None:
    """The composed score should equal the product of the three pieces."""
    svc = FakeEmbeddingService()
    now = datetime.utcnow()
    emb = await svc.embed("content")
    ep = Episode(
        content="content", importance=8.0, last_referenced_at=now, embedding=emb
    )
    query = await svc.embed("content")

    composed = rir_score(ep, query, now=now)
    expected = (
        recency_score(ep, now=now)
        * importance_normalized(ep)
        * relevance_score(ep, query)
    )
    assert composed == pytest.approx(expected)


# --- InMemoryEpisodicStore --------------------------------------------


async def test_in_memory_store_satisfies_protocol() -> None:
    """Runtime-checkable Protocol: the in-memory store should match."""
    store = InMemoryEpisodicStore()
    assert isinstance(store, EpisodicStore)


async def test_in_memory_write_and_list() -> None:
    store = InMemoryEpisodicStore()
    svc = FakeEmbeddingService()
    emb = await svc.embed("alpha")
    ep = await store.write(
        Episode(content="alpha", importance=5.0, embedding=emb)
    )
    items = await store.list_all()
    assert len(items) == 1
    assert items[0].episode_id == ep.episode_id


async def test_in_memory_recall_returns_top_k() -> None:
    store = InMemoryEpisodicStore()
    svc = FakeEmbeddingService()
    contents = ["fib(n) implementation", "sort a list", "binary search recipe"]
    for c in contents:
        emb = await svc.embed(c)
        await store.write(Episode(content=c, importance=7.0, embedding=emb))

    query = await svc.embed("fib(n) implementation")
    hits = await store.recall(query, top_k=2)
    assert len(hits) == 2
    # The exact-match content should rank first.
    assert hits[0].content == "fib(n) implementation"


async def test_in_memory_recall_increments_references_count() -> None:
    """Every returned episode must have references_count += 1."""
    store = InMemoryEpisodicStore()
    svc = FakeEmbeddingService()
    emb = await svc.embed("x")
    written = await store.write(
        Episode(content="x", importance=5.0, embedding=emb)
    )
    assert written.references_count == 0

    query = await svc.embed("x")
    hits = await store.recall(query, top_k=1)
    assert hits[0].references_count == 1

    hits2 = await store.recall(query, top_k=1)
    assert hits2[0].references_count == 2


async def test_in_memory_recall_updates_last_referenced_at() -> None:
    store = InMemoryEpisodicStore()
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
    assert hits[0].last_referenced_at == now


async def test_in_memory_recall_skips_embeddingless_episodes() -> None:
    """Episodes without embeddings can't be ranked; they should be skipped."""
    store = InMemoryEpisodicStore()
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


async def test_in_memory_recall_orders_by_rir() -> None:
    """A higher-importance hit should outrank a perfect-relevance but stale one."""
    store = InMemoryEpisodicStore()
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
    # Fresh + high importance + same content
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


async def test_in_memory_len() -> None:
    store = InMemoryEpisodicStore()
    assert len(store) == 0
    svc = FakeEmbeddingService()
    emb = await svc.embed("x")
    await store.write(Episode(content="x", importance=5.0, embedding=emb))
    assert len(store) == 1
