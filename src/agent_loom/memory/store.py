"""Episodic memory store with R × I × R retrieval.

Phase 1a: In-memory implementation. Phase 1b will swap in pgvector behind the
same `EpisodicStore` protocol.

Score formula:
    score(episode) = recency(episode) * importance(episode) * relevance(episode, query)

- recency: stepwise decay (24h=1.0, 1w=0.8, 1m=0.5, 3m+=0.1)
- importance: LLM-assigned at write time (1-10), boosted on each reference
- relevance: cosine similarity over 1536-dim embeddings (OpenAI text-embedding-3-small)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agent_loom.memory.embeddings import cosine_similarity


class Episode(BaseModel):
    episode_id: UUID = Field(default_factory=uuid4)
    content: str
    importance: float = Field(ge=0, le=10)
    references_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_referenced_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    source: str = Field(
        default="executor",
        description="Which module wrote this episode (e.g. 'executor', 'reflection').",
    )


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


def relevance_score(episode: Episode, query_embedding: list[float]) -> float:
    """Cosine similarity between episode and query embedding, clamped to [0, 1].

    Why clamp: cosine returns [-1, 1] but our R × I × R formula multiplies three
    factors all in [0, 1] so negative values would invert the ranking direction
    in confusing ways. A negative relevance just means "not similar" — treat as 0.
    """
    if episode.embedding is None:
        return 0.0
    sim = cosine_similarity(episode.embedding, query_embedding)
    return max(0.0, min(1.0, sim))


def rir_score(
    episode: Episode,
    query_embedding: list[float],
    *,
    now: datetime | None = None,
) -> float:
    """Compose the three factors. Each one is independently testable above."""
    return (
        recency_score(episode, now=now)
        * importance_normalized(episode)
        * relevance_score(episode, query_embedding)
    )


# --- Store protocol -------------------------------------------------------


@runtime_checkable
class EpisodicStore(Protocol):
    """Async store of episodes with vector search.

    Why a Protocol: pgvector (Phase 1b) will plug in here without changing
    `MemoryHub`. The in-memory implementation below is the Phase 1a default.
    """

    async def write(self, episode: Episode) -> Episode:
        """Persist `episode`. Returns the stored copy (same instance is fine)."""
        ...

    async def recall(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        now: datetime | None = None,
    ) -> list[Episode]:
        """Return up to `top_k` episodes ranked by R × I × R.

        Implementations MUST increment `references_count` and bump
        `last_referenced_at` for each returned episode.
        """
        ...

    async def list_all(self) -> list[Episode]:
        """Diagnostic helper. Not used in hot path."""
        ...


# --- In-memory implementation --------------------------------------------


class InMemoryEpisodicStore:
    """Phase 1a default. Dict-backed, brute-force scan on `recall`.

    Sized for ≤1k episodes — Phase 1b replaces this with pgvector for real
    deployments. The brute-force scan stays correct, just slow.
    """

    def __init__(self) -> None:
        self._episodes: dict[UUID, Episode] = {}

    async def write(self, episode: Episode) -> Episode:
        self._episodes[episode.episode_id] = episode
        return episode

    async def recall(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        now: datetime | None = None,
    ) -> list[Episode]:
        now = now or datetime.utcnow()
        scored: list[tuple[float, Episode]] = []
        for ep in self._episodes.values():
            if ep.embedding is None:
                # Skip episodes without embeddings — they can't be ranked by
                # relevance, so they would all share a 0 score and pollute the
                # output. They are still retrievable via `list_all`.
                continue
            s = rir_score(ep, query_embedding, now=now)
            scored.append((s, ep))

        # Stable sort by score descending. Ties keep insertion order, which
        # gives deterministic test output.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        winners = [ep for _, ep in scored[:top_k]]

        # References-count update: do it AFTER selection so a single recall
        # round bumps each returned episode exactly once.
        for ep in winners:
            ep.references_count += 1
            ep.last_referenced_at = now
        return winners

    async def list_all(self) -> list[Episode]:
        return list(self._episodes.values())

    def __len__(self) -> int:
        return len(self._episodes)
