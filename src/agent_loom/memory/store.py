"""Episodic memory store with R × I × R retrieval.

Phase 1 implementation target. Stub in Phase 0.

Score formula:
    score(episode) = recency(episode) * importance(episode) * relevance(episode, query)

- recency: stepwise decay (24h=1.0, 1w=0.8, 1m=0.5, 3m+=0.1)
- importance: LLM-assigned at write time (1-10), boosted on each reference
- relevance: cosine similarity over 1536-dim embeddings (OpenAI text-embedding-3-small)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Episode(BaseModel):
    episode_id: UUID = Field(default_factory=uuid4)
    content: str
    importance: float = Field(ge=0, le=10)
    references_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_referenced_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def recency_score(episode: Episode, now: datetime | None = None) -> float:
    """Stepwise decay (see Stanford Generative Agents)."""
    now = now or datetime.utcnow()
    age = now - episode.last_referenced_at
    if age < timedelta(hours=24):
        return 1.0
    if age < timedelta(weeks=1):
        return 0.8
    if age < timedelta(weeks=2):
        return 0.6
    if age < timedelta(days=30):
        return 0.5
    if age < timedelta(days=90):
        return 0.3
    return 0.1


def importance_normalized(episode: Episode) -> float:
    """Normalize importance from [0, 10] to [0, 1]."""
    return min(max(episode.importance, 0.0), 10.0) / 10.0


class EpisodicStore:
    """Phase 0 stub. Replace with real pgvector-backed store in Phase 1."""

    async def write(self, content: str, importance: float, embedding: list[float] | None = None) -> Episode:
        # TODO(phase-1): INSERT INTO episodes ... with embedding.
        raise NotImplementedError("EpisodicStore.write is implemented in Phase 1.")

    async def recall(self, query: str, *, top_k: int = 5) -> list[Episode]:
        # TODO(phase-1): SELECT scored by R × I × R, increment references_count.
        raise NotImplementedError("EpisodicStore.recall is implemented in Phase 1.")
