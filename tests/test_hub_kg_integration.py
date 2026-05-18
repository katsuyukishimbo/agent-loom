"""MemoryHub Phase 2 integration with the knowledge graph.

Covers automatic edge creation rules:
  - same run    → caused_by
  - same task signature → resembles
"""

from __future__ import annotations

from uuid import uuid4

from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.memory.graph import Subgraph
from agent_loom.memory.hub import MemoryHub


def _inputs(goal: str) -> tuple[SprintContract, Artifact, VerifierJudgement]:
    contract = SprintContract(
        run_id=uuid4(),
        goal=goal,
        acceptance_criteria=["x"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id, kind="text", content="content"
    )
    judgement = VerifierJudgement(
        artifact_id=artifact.artifact_id,
        passed=False,
        score=0.1,
        failure_category=FailureCategory.PARTIAL_IMPLEMENTATION,
        reflection="missed edge",
    )
    return contract, artifact, judgement


async def test_write_creates_experience_node() -> None:
    hub = MemoryHub.fake()
    contract, artifact, judgement = _inputs("task A")
    ep = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    nodes = await hub.graph.list_nodes()
    matching = [n for n in nodes if n.node_id == ep.episode_id]
    assert len(matching) == 1
    assert matching[0].subgraph == Subgraph.EXPERIENCE


async def test_consecutive_writes_in_same_run_create_caused_by_edge() -> None:
    """Two attempts under the same run_id link via caused_by."""
    hub = MemoryHub.fake()
    contract, artifact, judgement = _inputs("task A")
    ep1 = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    # Second attempt: same run_id (same contract instance reuses it) but
    # a freshly minted artifact so it's a different Episode.
    artifact2 = Artifact(
        contract_id=contract.contract_id, kind="text", content="content 2"
    )
    judgement2 = judgement.model_copy(update={"artifact_id": artifact2.artifact_id})
    ep2 = await hub.write_from_judgement(
        contract=contract, artifact=artifact2, judgement=judgement2
    )

    edges = await hub.graph.list_edges()
    caused = [
        e
        for e in edges
        if e.edge_type == "caused_by" and e.src == ep2.episode_id and e.dst == ep1.episode_id
    ]
    assert caused, f"Expected caused_by edge ep2 -> ep1, got {edges}"


async def test_same_task_signature_across_runs_creates_resembles_edge() -> None:
    """Two attempts on the same goal in DIFFERENT runs link via resembles."""
    hub = MemoryHub.fake()
    contract_a, art_a, jud_a = _inputs("Implement fib(n)")
    contract_b, art_b, jud_b = _inputs("Implement fib(n)")  # different run_id
    ep_a = await hub.write_from_judgement(
        contract=contract_a, artifact=art_a, judgement=jud_a
    )
    ep_b = await hub.write_from_judgement(
        contract=contract_b, artifact=art_b, judgement=jud_b
    )
    edges = await hub.graph.list_edges()
    resembles = [
        e
        for e in edges
        if e.edge_type == "resembles" and e.src == ep_b.episode_id and e.dst == ep_a.episode_id
    ]
    assert resembles, f"Expected resembles edge for same signature, got {edges}"


async def test_different_signatures_do_not_resemble() -> None:
    hub = MemoryHub.fake()
    contract_a, art_a, jud_a = _inputs("Implement fib(n)")
    contract_b, art_b, jud_b = _inputs("Implement bubble sort")
    ep_a = await hub.write_from_judgement(
        contract=contract_a, artifact=art_a, judgement=jud_a
    )
    ep_b = await hub.write_from_judgement(
        contract=contract_b, artifact=art_b, judgement=jud_b
    )
    edges = await hub.graph.list_edges()
    resembles = [
        e
        for e in edges
        if e.edge_type == "resembles"
        and {e.src, e.dst} == {ep_a.episode_id, ep_b.episode_id}
    ]
    assert not resembles


async def test_recall_failures_returns_only_failed_episodes() -> None:
    hub = MemoryHub.fake()
    contract, artifact, judgement = _inputs("task fail")
    await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    # Also plant a pass
    contract2 = SprintContract(
        run_id=uuid4(), goal="task ok", acceptance_criteria=["x"]
    )
    artifact2 = Artifact(
        contract_id=contract2.contract_id, kind="text", content="ok"
    )
    judgement2 = VerifierJudgement(
        artifact_id=artifact2.artifact_id,
        passed=True,
        score=0.9,
        failure_category=None,
        reflection="fine",
    )
    await hub.write_from_judgement(
        contract=contract2, artifact=artifact2, judgement=judgement2
    )

    failures = await hub.recall_failures("task", top_k=5)
    assert failures
    for ep in failures:
        assert ep.metadata["passed"] == "false"
