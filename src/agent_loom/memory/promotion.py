"""Episode → Reasoning promotion (Stanford-style hippocampus → cortex).

When an Episode crosses `PROMOTION_REFERENCE_THRESHOLD` references it's been
recalled often enough that its content reads as a stable lesson rather than a
single observation. We then write a parallel `reasoning`-tagged Episode that
the recall loop can prefer.

Design notes:

- We do not *delete* the original Episode. The references-count history is
  diagnostic and removing it would muddy future promotion decisions.
- The promoted copy gets `source = "reasoning"` and a metadata flag pointing
  back at the originating episode. That makes the link traceable from the
  CLI snapshot without a second table.
- Promotion is idempotent per source episode: once the metadata flag
  `promoted_from` is recorded, callers can skip without an extra DB read.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from agent_loom.memory.store import Episode

# Phase 2 DoD anchor — Episode promotes to Reasoning at >=3 references.
# (Stanford Generative Agents §A.5.)
PROMOTION_REFERENCE_THRESHOLD: int = 3


def should_promote(episode: Episode) -> bool:
    """True when an Episode is eligible for Reasoning promotion.

    Why a function and not an inline predicate: the rule may grow (e.g. add an
    importance floor or a recency window) and we want one place to change it.
    """
    if episode.references_count < PROMOTION_REFERENCE_THRESHOLD:
        return False
    # Already promoted — don't re-emit.
    if episode.metadata.get("promoted_to") == "reasoning":
        return False
    return True


def build_reasoning_copy(episode: Episode) -> Episode:
    """Return a new Episode tagged as a reasoning node.

    The copy preserves the embedding so it competes in R×I×R recall against
    the original. Importance is boosted by 1 (clamped to 10) so the promoted
    form ranks above the source on equal recency.
    """
    new_id = uuid4()
    boosted = min(10.0, episode.importance + 1.0)
    new_meta = dict(episode.metadata)
    new_meta["promoted_from"] = str(episode.episode_id)
    new_meta["kind"] = "reasoning"
    return Episode(
        episode_id=new_id,
        content=f"[REASONING] {episode.content}",
        importance=boosted,
        embedding=list(episode.embedding) if episode.embedding else None,
        metadata=new_meta,
        source="reasoning",
    )


def mark_source_promoted(episode: Episode, reasoning_id: UUID) -> None:
    """Record on the source that it has been promoted.

    Mutating the source in place mirrors how the in-memory store stores
    references — the same instance is what list_all() returns, so callers see
    the flag without a re-read.
    """
    episode.metadata["promoted_to"] = "reasoning"
    episode.metadata["reasoning_id"] = str(reasoning_id)
