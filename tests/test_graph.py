"""InMemoryKnowledgeGraph unit tests (Phase 2)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agent_loom.memory.graph import (
    GraphEdge,
    GraphNode,
    InMemoryKnowledgeGraph,
    KnowledgeGraph,
    Subgraph,
)


async def test_in_memory_kg_satisfies_protocol() -> None:
    """The default implementation must satisfy the runtime-checkable Protocol."""
    kg = InMemoryKnowledgeGraph()
    assert isinstance(kg, KnowledgeGraph)


async def test_add_node_is_idempotent() -> None:
    kg = InMemoryKnowledgeGraph()
    nid = uuid4()
    node = GraphNode(node_id=nid, subgraph=Subgraph.EXPERIENCE, label="t1")
    await kg.add_node(node)
    await kg.add_node(node)
    snap = await kg.snapshot()
    assert snap.total_nodes == 1


async def test_add_edge_is_idempotent_on_same_triple() -> None:
    kg = InMemoryKnowledgeGraph()
    a, b = uuid4(), uuid4()
    edge = GraphEdge(src=a, dst=b, edge_type="resembles")
    await kg.add_edge(edge)
    await kg.add_edge(edge)
    snap = await kg.snapshot()
    assert snap.total_edges == 1


async def test_different_edge_types_between_same_pair_are_separate() -> None:
    """resembles AND caused_by between A and B should both persist."""
    kg = InMemoryKnowledgeGraph()
    a, b = uuid4(), uuid4()
    await kg.add_edge(GraphEdge(src=a, dst=b, edge_type="resembles"))
    await kg.add_edge(GraphEdge(src=a, dst=b, edge_type="caused_by"))
    snap = await kg.snapshot()
    assert snap.total_edges == 2
    assert snap.edges_by_type["resembles"] == 1
    assert snap.edges_by_type["caused_by"] == 1


async def test_neighbors_returns_one_hop_in_both_directions() -> None:
    """An undirected-style traversal must walk both `_adj` and `_inverse`."""
    kg = InMemoryKnowledgeGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    await kg.add_node(GraphNode(node_id=a, subgraph=Subgraph.EXPERIENCE, label="a"))
    await kg.add_node(GraphNode(node_id=b, subgraph=Subgraph.EXPERIENCE, label="b"))
    await kg.add_node(GraphNode(node_id=c, subgraph=Subgraph.EXPERIENCE, label="c"))
    await kg.add_edge(GraphEdge(src=a, dst=b, edge_type="resembles"))
    await kg.add_edge(GraphEdge(src=c, dst=a, edge_type="caused_by"))

    nbrs = await kg.neighbors(a, hop=1)
    nbr_ids = {n.node_id for n in nbrs}
    assert nbr_ids == {b, c}


async def test_neighbors_two_hops() -> None:
    kg = InMemoryKnowledgeGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for nid, label in [(a, "a"), (b, "b"), (c, "c")]:
        await kg.add_node(GraphNode(node_id=nid, subgraph=Subgraph.EXPERIENCE, label=label))
    await kg.add_edge(GraphEdge(src=a, dst=b, edge_type="caused_by"))
    await kg.add_edge(GraphEdge(src=b, dst=c, edge_type="caused_by"))
    two_hops = await kg.neighbors(a, hop=2)
    assert {n.node_id for n in two_hops} == {b, c}


async def test_neighbors_zero_hop_returns_empty() -> None:
    kg = InMemoryKnowledgeGraph()
    nid = uuid4()
    await kg.add_node(GraphNode(node_id=nid, subgraph=Subgraph.EXPERIENCE, label="a"))
    assert await kg.neighbors(nid, hop=0) == []


async def test_snapshot_buckets_by_subgraph_and_edge_type() -> None:
    kg = InMemoryKnowledgeGraph()
    e1, e2, r1 = uuid4(), uuid4(), uuid4()
    await kg.add_node(GraphNode(node_id=e1, subgraph=Subgraph.EXPERIENCE, label="x"))
    await kg.add_node(GraphNode(node_id=e2, subgraph=Subgraph.EXPERIENCE, label="y"))
    await kg.add_node(GraphNode(node_id=r1, subgraph=Subgraph.REASONING, label="r"))
    await kg.add_edge(GraphEdge(src=e1, dst=e2, edge_type="resembles"))
    await kg.add_edge(GraphEdge(src=r1, dst=e1, edge_type="derived_from"))
    snap = await kg.snapshot()
    assert snap.nodes_by_subgraph["experience"] == 2
    assert snap.nodes_by_subgraph["reasoning"] == 1
    assert snap.edges_by_type["resembles"] == 1
    assert snap.edges_by_type["derived_from"] == 1


async def test_list_nodes_and_edges_return_copies() -> None:
    """Mutating the returned lists must not corrupt internal state."""
    kg = InMemoryKnowledgeGraph()
    nid = uuid4()
    await kg.add_node(GraphNode(node_id=nid, subgraph=Subgraph.EXPERIENCE, label="x"))
    nodes = await kg.list_nodes()
    nodes.clear()
    edges = await kg.list_edges()
    edges.clear()
    snap = await kg.snapshot()
    assert snap.total_nodes == 1


async def test_neighbors_returns_empty_for_unknown_node() -> None:
    kg = InMemoryKnowledgeGraph()
    stranger = uuid4()
    # No raise; just empty.
    assert await kg.neighbors(stranger, hop=1) == []


@pytest.mark.parametrize(
    "edge_type",
    [
        "caused_by",
        "resembles",
        "is_a",
        "decomposes_to",
        "applicable_to",
        "composed_of",
        "derived_from",
    ],
)
async def test_all_edge_types_round_trip(edge_type: str) -> None:
    kg = InMemoryKnowledgeGraph()
    a, b = uuid4(), uuid4()
    await kg.add_edge(GraphEdge(src=a, dst=b, edge_type=edge_type))  # type: ignore[arg-type]
    edges = await kg.list_edges()
    assert len(edges) == 1
    assert edges[0].edge_type == edge_type
