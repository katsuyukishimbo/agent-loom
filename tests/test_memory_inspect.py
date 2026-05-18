"""Tests for memory/inspect.py and the `agent-loom memory inspect` CLI."""

from __future__ import annotations

from uuid import uuid4

from typer.testing import CliRunner

from agent_loom.cli import app
from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.memory.graph import (
    GraphEdge,
    GraphNode,
    InMemoryKnowledgeGraph,
    Subgraph,
)
from agent_loom.memory.hub import MemoryHub
from agent_loom.memory.inspect import build_memory_snapshot, format_snapshot
from agent_loom.memory.store import Episode, InMemoryEpisodicStore


async def test_snapshot_counts_failures_and_categories() -> None:
    store = InMemoryEpisodicStore()
    graph = InMemoryKnowledgeGraph()
    # Plant 1 pass + 2 failures (1 spec_misread, 1 partial)
    for passed, cat in [
        ("true", ""),
        ("false", "spec_misread"),
        ("false", "partial_implementation"),
    ]:
        await store.write(
            Episode(
                content=f"pass={passed}",
                importance=5.0,
                embedding=[0.0] * 4,
                metadata={"passed": passed, "failure_category": cat},
            )
        )
    snap = await build_memory_snapshot(store=store, graph=graph)
    assert snap.total_episodes == 3
    assert snap.failures == 2
    assert snap.by_failure_category["spec_misread"] == 1
    assert snap.by_failure_category["partial_implementation"] == 1
    assert snap.failure_pct() > 0


async def test_snapshot_includes_graph_when_provided() -> None:
    store = InMemoryEpisodicStore()
    graph = InMemoryKnowledgeGraph()
    nid = uuid4()
    await graph.add_node(GraphNode(node_id=nid, subgraph=Subgraph.EXPERIENCE, label="x"))
    nid2 = uuid4()
    await graph.add_node(GraphNode(node_id=nid2, subgraph=Subgraph.EXPERIENCE, label="y"))
    await graph.add_edge(GraphEdge(src=nid, dst=nid2, edge_type="resembles"))

    snap = await build_memory_snapshot(store=store, graph=graph)
    assert snap.graph is not None
    assert snap.graph.total_nodes == 2
    assert snap.graph.total_edges == 1


async def test_format_snapshot_renders_human_readable() -> None:
    """The CLI's `inspect` output must contain the headline numbers."""
    store = InMemoryEpisodicStore()
    graph = InMemoryKnowledgeGraph()
    await store.write(
        Episode(
            content="something",
            importance=6.0,
            references_count=5,
            embedding=[0.0] * 4,
            metadata={"passed": "false", "failure_category": "spec_misread"},
        )
    )
    snap = await build_memory_snapshot(store=store, graph=graph)
    rendered = format_snapshot(snap)
    assert "Total episodes" in rendered
    assert "Failures" in rendered
    # No exception when the graph is non-empty:
    assert "Total nodes" in rendered


def test_cli_memory_inspect_runs() -> None:
    """`agent-loom memory inspect` exits 0 with empty store.

    Why sync: the CLI command uses `asyncio.run()` internally to drive its
    work. Running this test under pytest-asyncio (`async def`) would mean two
    nested event loops and a RuntimeError. CliRunner.invoke is sync anyway —
    making the test sync keeps the loops untangled.
    """
    runner = CliRunner()
    result = runner.invoke(app, ["memory", "inspect"])
    assert result.exit_code == 0, result.output
    assert "KG snapshot" in result.output


async def test_end_to_end_snapshot_after_pipeline() -> None:
    """Run the pipeline once; snapshot should reflect at least one episode."""
    hub = MemoryHub.fake()
    contract = SprintContract(
        run_id=uuid4(),
        goal="task X",
        acceptance_criteria=["thing"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id, kind="text", content="content"
    )
    judgement = VerifierJudgement(
        artifact_id=artifact.artifact_id,
        passed=False,
        score=0.1,
        failure_category=FailureCategory.SPEC_MISREAD,
        reflection="thing missing",
    )
    source = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    await hub.write_reflection(
        contract=contract,
        artifact=artifact,
        judgement=judgement,
        source_episode=source,
    )

    snap = await build_memory_snapshot(store=hub.store, graph=hub.graph)
    assert snap.total_episodes >= 2  # source + reflection
    assert snap.failures >= 1
    assert snap.graph is not None
    assert snap.graph.total_edges >= 1  # derived_from
