"""MemoryHub tests (fake mode)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.memory.embeddings import FakeEmbeddingService
from agent_loom.memory.hub import MemoryHub
from agent_loom.memory.store import InMemoryEpisodicStore


def _fixture_inputs() -> tuple[SprintContract, Artifact, VerifierJudgement]:
    run_id = uuid4()
    contract = SprintContract(
        run_id=run_id,
        goal="Write a Python function fib(n).",
        acceptance_criteria=["fib(10) == 55"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id,
        kind="text",
        content="def fib(n):\n    return n",
        files_touched=["fib.py"],
    )
    judgement = VerifierJudgement(
        artifact_id=artifact.artifact_id,
        passed=True,
        score=0.92,
        rubric_breakdown={"correctness": 0.95, "readability": 0.85},
        failure_category=None,
        reflection="Iterative form satisfies fib(10) == 55.",
    )
    return contract, artifact, judgement


async def test_fake_factory_yields_in_memory_store() -> None:
    hub = MemoryHub.fake()
    assert isinstance(hub.store, InMemoryEpisodicStore)
    assert isinstance(hub.embedder, FakeEmbeddingService)


async def test_write_from_judgement_persists_episode() -> None:
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    episode = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )

    stored = await hub.store.list_all()
    assert len(stored) == 1
    assert stored[0].episode_id == episode.episode_id
    assert episode.embedding is not None
    assert len(episode.embedding) == 1536


async def test_write_from_judgement_records_pass_metadata() -> None:
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    episode = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert episode.metadata["passed"] == "true"
    assert episode.metadata["run_id"] == str(contract.run_id)
    assert episode.metadata["contract_id"] == str(contract.contract_id)


async def test_write_from_judgement_records_failure_metadata() -> None:
    """Failures should still be persisted (for Phase 2 reflection)."""
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    failed = judgement.model_copy(
        update={
            "passed": False,
            "score": 0.2,
            "failure_category": FailureCategory.SPEC_MISREAD,
        }
    )
    episode = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=failed
    )
    assert episode.metadata["passed"] == "false"
    assert episode.metadata["failure_category"] == "spec_misread"


async def test_write_from_judgement_assigns_default_importance() -> None:
    """Fake mode never returns an `importance` key, so we fall back to 5.0."""
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    ep = await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert ep.importance == pytest.approx(5.0)


async def test_recall_returns_recently_written_episode() -> None:
    """Roundtrip: write -> recall with similar text -> get the episode back."""
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )

    hits = await hub.recall(
        "Write a Python function fib(n) implementation.", top_k=3
    )
    assert len(hits) >= 1


async def test_recall_increments_references_count() -> None:
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )

    before = (await hub.store.list_all())[0].references_count
    await hub.recall("fib", top_k=3)
    after = (await hub.store.list_all())[0].references_count
    assert after == before + 1


async def test_format_recall_for_persona_empty_list_yields_empty_string() -> None:
    """An empty recall must not pollute the persona prompt."""
    assert MemoryHub.format_recall_for_persona([]) == ""


async def test_format_recall_for_persona_lists_episodes() -> None:
    hub = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    await hub.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    hits = await hub.recall("fib", top_k=3)
    rendered = MemoryHub.format_recall_for_persona(hits)
    assert "Past relevant episodes" in rendered
    assert "passed=true" in rendered


async def test_importance_fallback_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the importance LLM call blows up, we coerce to 5.0 rather than fail the run."""
    from agent_loom.memory import hub as hub_module

    async def _boom(*args, **kwargs):
        raise RuntimeError("network melted")

    monkeypatch.setattr(hub_module, "complete", _boom)
    h = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    ep = await h.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert ep.importance == 5.0


async def test_importance_uses_llm_value_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the LLM returns a real importance, we adopt and clamp it."""
    from agent_loom import llm
    from agent_loom.memory import hub as hub_module

    async def _stub(*, model, system, user, role, max_tokens=2048):
        return llm.LLMResponse(
            text="{}",
            parsed={"importance": 9.3},
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0,
            provider="fake",
            model="fake",
        )

    monkeypatch.setattr(hub_module, "complete", _stub)
    h = MemoryHub.fake()
    contract, artifact, judgement = _fixture_inputs()
    ep = await h.write_from_judgement(
        contract=contract, artifact=artifact, judgement=judgement
    )
    assert ep.importance == pytest.approx(9.3)
