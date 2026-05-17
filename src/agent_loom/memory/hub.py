"""MemoryHub — single read/write entry point for memory.

The Planner reads (recall). The Executor writes (write_from_judgement). Nobody
else touches the store. This mirrors AgentFlow's shared-memory pattern and
keeps the "writes single-threaded" invariant trivial at the memory layer.

Phase 1a writes one Episode per Verifier judgement (success or failure). Phase
2 will add Reflective Compaction that turns failure narratives into higher-
importance Episodes and promotes 3+ referenced Episodes into the Reasoning
subgraph.
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
    """Read-side and write-side facade for episodic memory."""

    def __init__(
        self,
        store: EpisodicStore | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.store: EpisodicStore = store or InMemoryEpisodicStore()
        self.embedder: EmbeddingService = embedder or default_embedder()

    @classmethod
    def fake(cls) -> MemoryHub:
        """Convenience for tests + fake-mode runs."""
        return cls(store=InMemoryEpisodicStore(), embedder=FakeEmbeddingService())

    async def recall(self, query: str, *, top_k: int = 5) -> list[Episode]:
        """Planner-side read. Increments references_count on each hit."""
        query_emb = await self.embedder.embed(query)
        return await self.store.recall(query_emb, top_k=top_k)

    async def write_from_judgement(
        self,
        *,
        contract: SprintContract,
        artifact: Artifact,
        judgement: VerifierJudgement,
    ) -> Episode:
        """Executor-side write. Called once per Verifier judgement.

        Phase 1a writes both successes and failures; the `metadata` field tags
        which is which so Phase 2 can filter on failure when injecting
        constraints into the SprintContract's `forbidden` list.
        """
        content = _episode_text(
            contract=contract, artifact=artifact, judgement=judgement
        )
        importance = await _llm_importance(content)
        embedding = await self.embedder.embed(content)

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
            },
        )
        return await self.store.write(episode)

    @staticmethod
    def format_recall_for_persona(episodes: list[Episode]) -> str:
        """Render recalled episodes as a short prefix for the Generator persona.

        Why not modify SprintContract: types.py is locked. We weave the recall
        summary into the persona string the Planner produces so it travels
        through the existing contract pipeline untouched. Phase 2 moves this
        into `SprintContract.forbidden` once we add the typed field.
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
