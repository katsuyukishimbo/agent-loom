"""Co-evolving knowledge graph (MAGE-style).

Four subgraphs:
    - experience: Episode nodes, edges = caused_by, resembles
    - task:       Task signature nodes, edges = is_a, decomposes_to
    - skill:      Skill nodes (refs SKILL.md), edges = applicable_to, composed_of
    - reasoning:  Reasoning trace nodes, edges = derived_from

Phase 2 ships an InMemoryKnowledgeGraph (used by tests, `MemoryHub.fake()`, and
fake-mode runs) plus PgvectorKnowledgeGraph for persistent storage.

YAGNI decision: edges live in a flat list. We don't pull Neo4j or networkx
because Phase 2's traversal needs are tiny (1-hop neighbours, by-type filters,
counts). The InMemory implementation scans; pgvector pushes the work to a
JSONB-indexed SQL table. If the graph grows past ~100k edges we revisit.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, Field


class Subgraph(str, Enum):
    EXPERIENCE = "experience"
    TASK = "task"
    SKILL = "skill"
    REASONING = "reasoning"


EdgeType = Literal[
    "caused_by",
    "resembles",
    "is_a",
    "decomposes_to",
    "applicable_to",
    "composed_of",
    "derived_from",
]


class GraphNode(BaseModel):
    node_id: UUID
    subgraph: Subgraph
    label: str
    payload_ref: UUID | None = None  # FK to episodes/skills/etc.


class GraphEdge(BaseModel):
    src: UUID
    dst: UUID
    edge_type: EdgeType
    weight: float = 1.0


class GraphSnapshot(BaseModel):
    """Read-only view used by the CLI and dashboard. Cheap to serialise."""

    total_nodes: int
    total_edges: int
    nodes_by_subgraph: dict[str, int] = Field(default_factory=dict)
    edges_by_type: dict[str, int] = Field(default_factory=dict)


@runtime_checkable
class KnowledgeGraph(Protocol):
    """Minimal KG surface needed by Phase 2.

    Implementations MUST be idempotent on `add_node` / `add_edge`: hub-side
    callers re-issue edges every recall and we never want duplicates.
    """

    async def add_node(self, node: GraphNode) -> None: ...

    async def add_edge(self, edge: GraphEdge) -> None: ...

    async def neighbors(self, node_id: UUID, *, hop: int = 1) -> list[GraphNode]: ...

    async def snapshot(self) -> GraphSnapshot: ...

    async def list_nodes(self) -> list[GraphNode]: ...

    async def list_edges(self) -> list[GraphEdge]: ...


class InMemoryKnowledgeGraph:
    """Dict-backed KG. The Phase 2 default; tests and fake mode use this.

    Why dict + list rather than a graph library: Phase 2 cares about counts and
    one-hop neighbours. A 30-line implementation is easier to audit than a
    third-party library import.
    """

    def __init__(self) -> None:
        self._nodes: dict[UUID, GraphNode] = {}
        # `_adj[src]` lists outgoing edges; `_inverse[dst]` lists incoming.
        # Both are needed because `resembles` and `caused_by` are conceptually
        # undirected for traversal but stored directionally for provenance.
        self._adj: dict[UUID, list[GraphEdge]] = defaultdict(list)
        self._inverse: dict[UUID, list[GraphEdge]] = defaultdict(list)
        self._all_edges: list[GraphEdge] = []
        # Edge dedup key — (src, dst, edge_type). Weight isn't part of identity.
        self._edge_keys: set[tuple[UUID, UUID, str]] = set()

    async def add_node(self, node: GraphNode) -> None:
        # Idempotent insert. Re-adding overwrites the payload_ref / label so
        # the latest write wins — desirable because reflective writes overwrite
        # the original episode's high-level metadata.
        self._nodes[node.node_id] = node

    async def add_edge(self, edge: GraphEdge) -> None:
        key = (edge.src, edge.dst, edge.edge_type)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self._adj[edge.src].append(edge)
        self._inverse[edge.dst].append(edge)
        self._all_edges.append(edge)

    async def neighbors(self, node_id: UUID, *, hop: int = 1) -> list[GraphNode]:
        """Return nodes reachable within `hop` edges (in or out).

        Why BFS over recursion: 1-hop is the dominant call; the loop body
        stays trivial and we never blow the recursion limit on cyclic graphs.
        """
        if hop < 1:
            return []
        visited: set[UUID] = {node_id}
        frontier: set[UUID] = {node_id}
        for _ in range(hop):
            next_frontier: set[UUID] = set()
            for nid in frontier:
                for e in self._adj.get(nid, []):
                    if e.dst not in visited:
                        next_frontier.add(e.dst)
                for e in self._inverse.get(nid, []):
                    if e.src not in visited:
                        next_frontier.add(e.src)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(node_id)
        return [self._nodes[nid] for nid in visited if nid in self._nodes]

    async def snapshot(self) -> GraphSnapshot:
        by_sub: dict[str, int] = defaultdict(int)
        for n in self._nodes.values():
            by_sub[n.subgraph.value] += 1
        by_type: dict[str, int] = defaultdict(int)
        for e in self._all_edges:
            by_type[e.edge_type] += 1
        return GraphSnapshot(
            total_nodes=len(self._nodes),
            total_edges=len(self._all_edges),
            nodes_by_subgraph=dict(by_sub),
            edges_by_type=dict(by_type),
        )

    async def list_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    async def list_edges(self) -> list[GraphEdge]:
        return list(self._all_edges)

    def __len__(self) -> int:
        return len(self._nodes)
