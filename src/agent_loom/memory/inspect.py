"""Diagnostic snapshots for the episodic store and the knowledge graph.

Pure read-only helpers. The CLI's `agent-loom memory inspect` calls these,
benchmarks call them, the dashboard (Phase 3) will reuse the same shape.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agent_loom.memory.graph import GraphSnapshot, KnowledgeGraph
from agent_loom.memory.store import Episode, EpisodicStore


@dataclass
class MemorySnapshot:
    """Aggregate over both the episodic store and the KG.

    Why a dataclass and not a Pydantic model: this is internal telemetry, not
    a wire format. Cheaper construction.
    """

    total_episodes: int
    failures: int
    promoted_to_reasoning: int
    by_failure_category: dict[str, int]
    top_referenced: list[Episode]
    graph: GraphSnapshot | None

    def failure_pct(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.failures / self.total_episodes


async def snapshot_episodic_store(
    store: EpisodicStore,
    *,
    top_n: int = 5,
) -> tuple[int, int, int, dict[str, int], list[Episode]]:
    """Return (total, failures, promoted, by_category, top_referenced).

    Pulled out so callers that already hold a KG snapshot can compose without
    re-reading the store.
    """
    all_eps = await store.list_all()
    failures = 0
    promoted = 0
    by_category: Counter[str] = Counter()

    for ep in all_eps:
        if ep.metadata.get("passed") == "false":
            failures += 1
        if ep.metadata.get("promoted_to") == "reasoning":
            promoted += 1
        cat = ep.metadata.get("failure_category") or ""
        if cat:
            by_category[cat] += 1

    # Top-referenced: ties broken by importance then by recency.
    top = sorted(
        all_eps,
        key=lambda e: (e.references_count, e.importance, e.last_referenced_at),
        reverse=True,
    )[:top_n]

    return len(all_eps), failures, promoted, dict(by_category), top


async def build_memory_snapshot(
    *,
    store: EpisodicStore,
    graph: KnowledgeGraph | None = None,
    top_n: int = 5,
) -> MemorySnapshot:
    total, failures, promoted, by_cat, top = await snapshot_episodic_store(
        store, top_n=top_n
    )
    graph_snap = await graph.snapshot() if graph is not None else None
    return MemorySnapshot(
        total_episodes=total,
        failures=failures,
        promoted_to_reasoning=promoted,
        by_failure_category=by_cat,
        top_referenced=top,
        graph=graph_snap,
    )


def format_snapshot(snap: MemorySnapshot) -> str:
    """Pretty-printer for the CLI. Matches the layout in docs/PHASES.md."""
    lines: list[str] = []
    lines.append("KG snapshot:")
    lines.append(f"  Total episodes:        {snap.total_episodes}")
    pct = snap.failure_pct() * 100
    lines.append(f"  Failures:              {snap.failures} ({pct:.0f}%)")
    lines.append(f"  Promoted to reasoning: {snap.promoted_to_reasoning}")
    if snap.graph is not None:
        lines.append(f"  Total nodes:           {snap.graph.total_nodes}")
        lines.append(f"  Total edges:           {snap.graph.total_edges}")
        if snap.graph.edges_by_type:
            lines.append("  Edge types:")
            for et, n in sorted(
                snap.graph.edges_by_type.items(), key=lambda kv: kv[1], reverse=True
            ):
                lines.append(f"    {et:<14} {n}")
    if snap.by_failure_category:
        lines.append("  Failure categories:")
        for cat, n in sorted(
            snap.by_failure_category.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(f"    {cat:<24} {n}")
    if snap.top_referenced:
        lines.append("")
        lines.append("Top referenced episodes:")
        for ep in snap.top_referenced:
            cat = ep.metadata.get("failure_category") or "-"
            preview = ep.content.replace("\n", " ")[:80]
            lines.append(
                f"  - id={ep.episode_id} refs={ep.references_count} "
                f"imp={ep.importance:.1f} cat={cat} {preview!r}"
            )
    return "\n".join(lines)
