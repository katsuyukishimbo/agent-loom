"""Episode → Reasoning promotion (Phase 2)."""

from __future__ import annotations

import pytest

from agent_loom.memory.embeddings import FakeEmbeddingService
from agent_loom.memory.graph import InMemoryKnowledgeGraph, Subgraph
from agent_loom.memory.hub import MemoryHub
from agent_loom.memory.promotion import (
    PROMOTION_REFERENCE_THRESHOLD,
    build_reasoning_copy,
    mark_source_promoted,
    should_promote,
)
from agent_loom.memory.store import Episode, InMemoryEpisodicStore


def _make_episode(refs: int = 0, importance: float = 5.0, **meta: str) -> Episode:
    return Episode(
        content="GOAL: x\nOUTCOME: PASS",
        importance=importance,
        references_count=refs,
        embedding=[0.0] * 4,
        metadata=dict(meta),
    )


def test_should_promote_threshold() -> None:
    assert not should_promote(_make_episode(refs=PROMOTION_REFERENCE_THRESHOLD - 1))
    assert should_promote(_make_episode(refs=PROMOTION_REFERENCE_THRESHOLD))
    assert should_promote(_make_episode(refs=PROMOTION_REFERENCE_THRESHOLD + 10))


def test_should_not_re_promote() -> None:
    ep = _make_episode(refs=PROMOTION_REFERENCE_THRESHOLD, promoted_to="reasoning")
    assert not should_promote(ep)


def test_build_reasoning_copy_keeps_embedding_and_boosts_importance() -> None:
    src = _make_episode(refs=3, importance=7.0)
    copy = build_reasoning_copy(src)
    assert copy.source == "reasoning"
    assert copy.metadata["kind"] == "reasoning"
    assert copy.metadata["promoted_from"] == str(src.episode_id)
    assert copy.importance == pytest.approx(8.0)
    # The embedding must be a separate list (caller may mutate either side).
    assert copy.embedding == src.embedding
    assert copy.embedding is not src.embedding


def test_build_reasoning_copy_clamps_importance_to_ten() -> None:
    src = _make_episode(refs=3, importance=10.0)
    copy = build_reasoning_copy(src)
    assert copy.importance == 10.0


def test_mark_source_promoted_records_pointer() -> None:
    src = _make_episode(refs=3)
    copy = build_reasoning_copy(src)
    mark_source_promoted(src, copy.episode_id)
    assert src.metadata["promoted_to"] == "reasoning"
    assert src.metadata["reasoning_id"] == str(copy.episode_id)


# ---- end-to-end via MemoryHub.recall() ------------------------------


async def test_recall_triggers_promotion_when_threshold_crossed() -> None:
    """A third recall on the same Episode must produce a Reasoning Episode."""
    hub = MemoryHub(
        store=InMemoryEpisodicStore(),
        embedder=FakeEmbeddingService(),
        graph=InMemoryKnowledgeGraph(),
    )
    # Pre-seed an Episode with 2 refs so the 3rd recall pushes it over.
    ep = _make_episode(refs=2, importance=6.0)
    ep.embedding = await hub.embedder.embed("task A")
    await hub.store.write(ep)

    # First recall pushes refs from 2 to 3, AND triggers promotion (>=3).
    await hub.recall("task A", top_k=5)

    all_eps = await hub.store.list_all()
    sources = {e.source for e in all_eps}
    assert "reasoning" in sources
    # Source episode must now record that it has been promoted.
    refreshed_src = next(e for e in all_eps if e.episode_id == ep.episode_id)
    assert refreshed_src.metadata.get("promoted_to") == "reasoning"


async def test_promotion_writes_reasoning_node_to_graph() -> None:
    hub = MemoryHub(
        store=InMemoryEpisodicStore(),
        embedder=FakeEmbeddingService(),
        graph=InMemoryKnowledgeGraph(),
    )
    ep = _make_episode(refs=2, importance=6.0)
    ep.embedding = await hub.embedder.embed("task B")
    await hub.store.write(ep)
    await hub.recall("task B", top_k=5)

    nodes = await hub.graph.list_nodes()
    reasoning_nodes = [n for n in nodes if n.subgraph == Subgraph.REASONING]
    assert reasoning_nodes, "Expected a reasoning subgraph node after promotion"

    edges = await hub.graph.list_edges()
    derived = [e for e in edges if e.edge_type == "derived_from"]
    assert derived, "Expected at least one derived_from edge from reasoning → source"


async def test_promotion_does_not_repeat() -> None:
    """Calling recall multiple more times must not stack duplicate promotions."""
    hub = MemoryHub(
        store=InMemoryEpisodicStore(),
        embedder=FakeEmbeddingService(),
        graph=InMemoryKnowledgeGraph(),
    )
    ep = _make_episode(refs=2, importance=6.0)
    ep.embedding = await hub.embedder.embed("task C")
    await hub.store.write(ep)

    await hub.recall("task C", top_k=5)
    await hub.recall("task C", top_k=5)
    await hub.recall("task C", top_k=5)

    all_eps = await hub.store.list_all()
    reasoning_eps = [e for e in all_eps if e.source == "reasoning"]
    assert len(reasoning_eps) == 1
