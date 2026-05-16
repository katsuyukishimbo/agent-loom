"""Co-evolving knowledge graph (MAGE-style).

Four subgraphs:
    - experience: Episode nodes, edges = caused_by, resembles
    - task:       Task signature nodes, edges = is_a, decomposes_to
    - skill:      Skill nodes (refs SKILL.md), edges = applicable_to, composed_of
    - reasoning:  Reasoning trace nodes, edges = derived_from

Phase 2 implementation target. Stub in Phase 0/1.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class Subgraph(str, Enum):
    EXPERIENCE = "experience"
    TASK = "task"
    SKILL = "skill"
    REASONING = "reasoning"


class GraphNode(BaseModel):
    node_id: UUID
    subgraph: Subgraph
    label: str
    payload_ref: UUID | None = None  # FK to episodes/skills/etc.


class GraphEdge(BaseModel):
    src: UUID
    dst: UUID
    edge_type: Literal[
        "caused_by",
        "resembles",
        "is_a",
        "decomposes_to",
        "applicable_to",
        "composed_of",
        "derived_from",
    ]
    weight: float = 1.0


class KnowledgeGraph:
    """Phase 2 implementation target."""

    async def add_node(self, node: GraphNode) -> None:
        raise NotImplementedError

    async def add_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError

    async def neighbors(self, node_id: UUID, *, hop: int = 1) -> list[GraphNode]:
        raise NotImplementedError
