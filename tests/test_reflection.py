"""Reflective Compaction tests (Phase 2)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.memory import reflection as reflection_module
from agent_loom.memory.embeddings import FakeEmbeddingService
from agent_loom.memory.graph import InMemoryKnowledgeGraph
from agent_loom.memory.hub import MemoryHub
from agent_loom.memory.reflection import (
    MAX_IMPORTANCE,
    MIN_REFLECTION_IMPORTANCE,
    reflect_on_failure,
)
from agent_loom.memory.store import InMemoryEpisodicStore


def _fixture() -> tuple[SprintContract, Artifact, VerifierJudgement]:
    contract = SprintContract(
        run_id=uuid4(),
        goal="Write fib(n) that returns the n-th Fibonacci number.",
        acceptance_criteria=["fib(10) == 55"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id,
        kind="text",
        content="def fib(n): return n  # wrong",
        files_touched=["fib.py"],
    )
    judgement = VerifierJudgement(
        artifact_id=artifact.artifact_id,
        passed=False,
        score=0.1,
        failure_category=FailureCategory.PARTIAL_IMPLEMENTATION,
        reflection="fib(10) returned 10 instead of 55.",
    )
    return contract, artifact, judgement


async def test_reflect_on_failure_returns_summary_and_high_importance() -> None:
    contract, artifact, judgement = _fixture()
    result = await reflect_on_failure(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert result.summary, "Reflection must produce a non-empty summary"
    # Fake mode emits importance=9.0 (see _fake_response in llm.py).
    assert MIN_REFLECTION_IMPORTANCE <= result.importance <= MAX_IMPORTANCE


async def test_reflect_on_failure_falls_back_when_llm_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(reflection_module, "complete", _boom)
    contract, artifact, judgement = _fixture()
    result = await reflect_on_failure(
        contract=contract, artifact=artifact, judgement=judgement
    )
    # Fallback uses the verifier's own reflection seed.
    assert "fib(10)" in result.summary or judgement.failure_category.value in result.summary
    assert result.importance == MIN_REFLECTION_IMPORTANCE


async def test_reflect_on_failure_clamps_out_of_band_importance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the LLM returns importance=50 we still clamp to MAX."""
    from agent_loom import llm

    async def _stub(*, model, system, user, role, max_tokens=2048):
        return llm.LLMResponse(
            text="{}",
            parsed={"summary": "x", "importance": 50},
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
            provider="fake",
            model="fake",
        )

    monkeypatch.setattr(reflection_module, "complete", _stub)
    contract, artifact, judgement = _fixture()
    result = await reflect_on_failure(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert result.importance == MAX_IMPORTANCE


async def test_reflect_on_failure_low_importance_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_loom import llm

    async def _stub(*, model, system, user, role, max_tokens=2048):
        return llm.LLMResponse(
            text="{}",
            parsed={"summary": "x", "importance": 1},
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
            provider="fake",
            model="fake",
        )

    monkeypatch.setattr(reflection_module, "complete", _stub)
    contract, artifact, judgement = _fixture()
    result = await reflect_on_failure(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert result.importance == MIN_REFLECTION_IMPORTANCE


# ---- MemoryHub.write_reflection integration --------------------------


async def test_write_reflection_persists_dedicated_episode() -> None:
    hub = MemoryHub(
        store=InMemoryEpisodicStore(),
        embedder=FakeEmbeddingService(),
        graph=InMemoryKnowledgeGraph(),
    )
    contract, artifact, judgement = _fixture()
    # Write the source failure episode first so we can attach a derived_from.
    source = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    reflection_ep = await hub.write_reflection(
        contract=contract,
        artifact=artifact,
        judgement=judgement,
        source_episode=source,
    )

    assert reflection_ep.source == "reflection"
    assert reflection_ep.metadata["passed"] == "false"
    assert reflection_ep.metadata["kind"] == "reflection"
    assert reflection_ep.importance >= MIN_REFLECTION_IMPORTANCE


async def test_write_reflection_adds_derived_from_edge() -> None:
    hub = MemoryHub(
        store=InMemoryEpisodicStore(),
        embedder=FakeEmbeddingService(),
        graph=InMemoryKnowledgeGraph(),
    )
    contract, artifact, judgement = _fixture()
    source = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    reflection_ep = await hub.write_reflection(
        contract=contract,
        artifact=artifact,
        judgement=judgement,
        source_episode=source,
    )
    edges = await hub.graph.list_edges()
    derived = [
        e
        for e in edges
        if e.edge_type == "derived_from"
        and e.src == reflection_ep.episode_id
        and e.dst == source.episode_id
    ]
    assert derived, "Reflection must derive_from its source episode"
