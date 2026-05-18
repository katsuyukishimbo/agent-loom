"""MemoryHub — single read/write entry point for memory.

The Planner reads (recall). The Executor writes (write_from_judgement). Nobody
else touches the store. This mirrors AgentFlow's shared-memory pattern and
keeps the "writes single-threaded" invariant trivial at the memory layer.

Phase 1a wrote one Episode per Verifier judgement (success or failure). Phase
2 layers:
  * **Reflective Compaction** — on Verifier failure, an LLM produces a 1-2
    paragraph "what went wrong" Episode tagged `source="reflection"` with
    elevated importance.
  * **KG edges** — same-task-signature Episodes link with `resembles`; Episodes
    written within the same run link via `caused_by` to the prior Episode.
  * **Promotion** — Episodes with ≥3 references become `reasoning` copies.
"""

from __future__ import annotations

from agent_loom.config import get_settings
from agent_loom.core.types import Artifact, SprintContract, VerifierJudgement
from agent_loom.llm import complete
from agent_loom.memory.embeddings import (
    EmbeddingService,
    FakeEmbeddingService,
    default_embedder,
)
from agent_loom.memory.graph import (
    GraphEdge,
    GraphNode,
    InMemoryKnowledgeGraph,
    KnowledgeGraph,
    Subgraph,
)
from agent_loom.memory.promotion import (
    build_reasoning_copy,
    mark_source_promoted,
    should_promote,
)
from agent_loom.memory.reflection import ReflectionResult, reflect_on_failure
from agent_loom.memory.store import (
    Episode,
    EpisodicStore,
    InMemoryEpisodicStore,
)

_IMPORTANCE_PROMPT = (
    "You score the long-term importance of a single agent episode on a 1-10 "
    "scale.\n"
    "1 = trivial / one-off chatter. 10 = pivotal, must be remembered.\n"
    "Output ONLY a valid JSON object with a single key:\n"
    '  {"importance": <number 1-10>}\n'
)


async def _llm_importance(content: str, *, model: str | None = None) -> float:
    """Ask the LLM for an importance score. Falls back to 5.0 on parse errors.

    Why a fallback rather than crash: the Verifier already grades the artifact;
    a missing importance score should not break the run.
    """
    settings = get_settings()
    chosen_model = model or settings.verifier_model
    try:
        resp = await complete(
            model=chosen_model,
            system=_IMPORTANCE_PROMPT,
            user=content[:4000],
            # `verifier` role gets a canned fake response with no `importance`
            # field; we coerce below. Using verifier keeps the role enum tight.
            role="verifier",
        )
        raw = resp.parsed.get("importance")
        if raw is None:
            return 5.0
        return float(max(1.0, min(10.0, float(raw))))
    except Exception:
        return 5.0


def _task_signature(contract: SprintContract) -> str:
    """Cheap signature for grouping `resembles` edges.

    Why goal text rather than a hash of acceptance criteria: criteria mutate
    across iterations as forbidden constraints accumulate, but the underlying
    task identity is stable. The goal field is the most stable signal we own.
    """
    return contract.goal.strip().lower()


def _episode_text(
    *, contract: SprintContract, artifact: Artifact, judgement: VerifierJudgement
) -> str:
    """The text we embed and store.

    Combining goal + artifact summary + outcome lets future recall match on
    either the task framing or the result. Keep it short — embeddings degrade
    on >8k tokens.
    """
    status = "PASS" if judgement.passed else "FAIL"
    fc = judgement.failure_category.value if judgement.failure_category else "-"
    reflection = (judgement.reflection or "").strip()
    artifact_summary = artifact.content.strip().splitlines()
    head = "\n".join(artifact_summary[:8])
    return (
        f"GOAL: {contract.goal}\n"
        f"OUTCOME: {status} score={judgement.score:.2f} category={fc}\n"
        f"REFLECTION: {reflection}\n"
        f"ARTIFACT_HEAD:\n{head}"
    )


class MemoryHub:
    """Read-side and write-side facade for episodic memory + KG."""

    def __init__(
        self,
        store: EpisodicStore | None = None,
        embedder: EmbeddingService | None = None,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.store: EpisodicStore = store or InMemoryEpisodicStore()
        self.embedder: EmbeddingService = embedder or default_embedder()
        # The KG is opt-in in Phase 2: tests for the in-memory store don't have
        # to construct one, and pgvector stays unaware. A None graph silently
        # disables edge/promotion writes.
        self.graph: KnowledgeGraph = graph or InMemoryKnowledgeGraph()

        # Per-run state for edge construction. The Executor calls
        # `write_from_judgement` once per attempt; we track the most recent
        # Episode per run so the next write can attach a `caused_by` edge.
        # Keyed by run_id (str form to match Episode.metadata).
        self._last_episode_per_run: dict[str, Episode] = {}
        # Index of episodes per task signature. Lets us emit `resembles`
        # edges to the prior Episodes on the same goal without scanning the
        # whole store on every write.
        self._episodes_by_signature: dict[str, list[Episode]] = {}

    @classmethod
    def fake(cls) -> MemoryHub:
        """Convenience for tests + fake-mode runs."""
        return cls(
            store=InMemoryEpisodicStore(),
            embedder=FakeEmbeddingService(),
            graph=InMemoryKnowledgeGraph(),
        )

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    async def recall(self, query: str, *, top_k: int = 5) -> list[Episode]:
        """Planner-side read. Increments references_count on each hit.

        After recall, promote any Episode that crossed the references
        threshold. Promotion fires lazily here so we don't pay the cost on
        every write.
        """
        query_emb = await self.embedder.embed(query)
        hits = await self.store.recall(query_emb, top_k=top_k)
        await self._maybe_promote(hits)
        return hits

    async def recall_failures(self, query: str, *, top_k: int = 5) -> list[Episode]:
        """Recall but filter to failure-tagged Episodes only.

        Phase 2's Planner injects these into `SprintContract.forbidden`. We
        over-fetch (4× top_k) so the filter doesn't starve the result list
        when the store mixes passes and failures.
        """
        over_fetch = max(top_k * 4, top_k)
        all_hits = await self.recall(query, top_k=over_fetch)
        return [e for e in all_hits if e.metadata.get("passed") == "false"][:top_k]

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    async def write_from_judgement(
        self,
        *,
        contract: SprintContract,
        artifact: Artifact,
        judgement: VerifierJudgement,
    ) -> Episode:
        """Executor-side write. Called once per Verifier judgement.

        Phase 2 additions over Phase 1a:
          1. Emits a `GraphNode` in the experience subgraph for every write.
          2. Adds a `caused_by` edge to the previous Episode in the same run.
          3. Adds `resembles` edges to prior Episodes on the same task signature.
          4. On failure, triggers Reflective Compaction (separate Episode).
        """
        content = _episode_text(
            contract=contract, artifact=artifact, judgement=judgement
        )
        importance = await _llm_importance(content)
        embedding = await self.embedder.embed(content)

        signature = _task_signature(contract)
        episode = Episode(
            content=content,
            importance=importance,
            embedding=embedding,
            source="executor",
            metadata={
                "run_id": str(contract.run_id),
                "contract_id": str(contract.contract_id),
                "artifact_id": str(artifact.artifact_id),
                "passed": "true" if judgement.passed else "false",
                "score": f"{judgement.score:.4f}",
                "failure_category": (
                    judgement.failure_category.value
                    if judgement.failure_category
                    else ""
                ),
                "task_signature": signature,
            },
        )
        stored = await self.store.write(episode)
        await self._register_in_graph(stored, signature=signature)
        return stored

    async def write_reflection(
        self,
        *,
        contract: SprintContract,
        artifact: Artifact,
        judgement: VerifierJudgement,
        source_episode: Episode | None = None,
        model: str | None = None,
    ) -> Episode:
        """Run reflective compaction and persist the resulting Episode.

        `source_episode` is the original failure Episode from
        `write_from_judgement`. When supplied we add a `derived_from` edge to
        connect the reasoning node back to the source — handy for the dashboard
        and for debugging false-positive reflections.

        Pre-condition: the caller has already confirmed `judgement.passed` is
        False. Calling on a passed judgement still works but the produced
        Episode will be low-utility.
        """
        result: ReflectionResult = await reflect_on_failure(
            contract=contract,
            artifact=artifact,
            judgement=judgement,
            model=model,
        )

        signature = _task_signature(contract)
        # Embed the dense summary, not the raw artifact — recall on the next
        # task will key off the failure mode.
        embedding = await self.embedder.embed(result.summary)
        reflection_ep = Episode(
            content=result.summary,
            importance=result.importance,
            embedding=embedding,
            source="reflection",
            metadata={
                "run_id": str(contract.run_id),
                "contract_id": str(contract.contract_id),
                "artifact_id": str(artifact.artifact_id),
                "passed": "false",
                "score": f"{judgement.score:.4f}",
                "failure_category": (
                    judgement.failure_category.value
                    if judgement.failure_category
                    else ""
                ),
                "task_signature": signature,
                "kind": "reflection",
            },
        )
        stored = await self.store.write(reflection_ep)
        await self._register_in_graph(stored, signature=signature)
        if source_episode is not None:
            await self.graph.add_edge(
                GraphEdge(
                    src=stored.episode_id,
                    dst=source_episode.episode_id,
                    edge_type="derived_from",
                )
            )
        return stored

    # ------------------------------------------------------------------
    # Internal: graph + promotion
    # ------------------------------------------------------------------

    async def _register_in_graph(
        self,
        episode: Episode,
        *,
        signature: str,
    ) -> None:
        """Add the Episode as a node and wire up `caused_by` / `resembles`.

        Edge construction rules (cheap, deterministic; see ARCHITECTURE §2.1):
          * `caused_by`: new ep → previous ep in the SAME run
          * `resembles`: new ep ↔ each prior ep with the SAME task signature
        """
        node = GraphNode(
            node_id=episode.episode_id,
            subgraph=Subgraph.EXPERIENCE,
            label=signature[:80],
            payload_ref=episode.episode_id,
        )
        await self.graph.add_node(node)

        run_id = episode.metadata.get("run_id", "")
        prior_in_run = self._last_episode_per_run.get(run_id)
        if prior_in_run is not None and prior_in_run.episode_id != episode.episode_id:
            await self.graph.add_edge(
                GraphEdge(
                    src=episode.episode_id,
                    dst=prior_in_run.episode_id,
                    edge_type="caused_by",
                )
            )
        if run_id:
            self._last_episode_per_run[run_id] = episode

        siblings = self._episodes_by_signature.setdefault(signature, [])
        for sib in siblings:
            if sib.episode_id == episode.episode_id:
                continue
            # Resembles is directionally symmetric for the KG's purpose; we
            # emit one edge new→old. The InMemoryKnowledgeGraph's neighbour
            # query walks both directions so adding the reverse would just
            # duplicate work.
            await self.graph.add_edge(
                GraphEdge(
                    src=episode.episode_id,
                    dst=sib.episode_id,
                    edge_type="resembles",
                )
            )
        siblings.append(episode)

    async def _maybe_promote(self, episodes: list[Episode]) -> None:
        """Run the Episode → Reasoning promotion check on each recalled hit."""
        for ep in episodes:
            if not should_promote(ep):
                continue
            reasoning_ep = build_reasoning_copy(ep)
            stored = await self.store.write(reasoning_ep)
            mark_source_promoted(ep, stored.episode_id)
            # Re-write the source so its `promoted_to` metadata persists in
            # the underlying store (in-memory: no-op; pgvector: UPSERT).
            await self.store.write(ep)
            signature = ep.metadata.get(
                "task_signature", ep.content.splitlines()[0][:80].lower()
            )
            # Add the reasoning node into the reasoning subgraph and link it
            # back to the source via `derived_from`.
            await self.graph.add_node(
                GraphNode(
                    node_id=stored.episode_id,
                    subgraph=Subgraph.REASONING,
                    label=f"reasoning::{signature[:60]}",
                    payload_ref=stored.episode_id,
                )
            )
            await self.graph.add_edge(
                GraphEdge(
                    src=stored.episode_id,
                    dst=ep.episode_id,
                    edge_type="derived_from",
                )
            )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_recall_for_persona(episodes: list[Episode]) -> str:
        """Render recalled episodes as a short prefix for the Generator persona.

        Why not modify SprintContract: types.py is locked. We weave the recall
        summary into the persona string the Planner produces so it travels
        through the existing contract pipeline untouched. Phase 2 ALSO injects
        failure summaries into `SprintContract.forbidden` (see Planner).
        """
        if not episodes:
            return ""
        lines = ["Past relevant episodes (from memory, ranked by R×I×R):"]
        for i, ep in enumerate(episodes, 1):
            passed = ep.metadata.get("passed", "?")
            score = ep.metadata.get("score", "?")
            preview = ep.content.replace("\n", " ")[:240]
            lines.append(f"  {i}. [passed={passed} score={score}] {preview}")
        return "\n".join(lines)

    @staticmethod
    def format_failures_for_forbidden(episodes: list[Episode]) -> list[str]:
        """Build `SprintContract.forbidden` entries from failure Episodes.

        One string per Episode, prefixed with the canonical "Past failure" tag
        the Verifier prompt knows to look for.
        """
        out: list[str] = []
        for ep in episodes:
            cat = ep.metadata.get("failure_category") or "unknown"
            preview = ep.content.replace("\n", " ").strip()[:240]
            out.append(f"Past failure ({cat}): {preview}")
        return out
