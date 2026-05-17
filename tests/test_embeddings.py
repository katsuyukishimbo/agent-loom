"""Embedding service tests (fake mode)."""

from __future__ import annotations

import math

import pytest

from agent_loom.memory.embeddings import (
    EMBEDDING_DIM,
    FakeEmbeddingService,
    OpenAIEmbeddingService,
    cosine_similarity,
    default_embedder,
)


async def test_fake_embedding_has_default_dim() -> None:
    svc = FakeEmbeddingService()
    vec = await svc.embed("hello world")
    assert len(vec) == EMBEDDING_DIM == 1536


async def test_fake_embedding_is_deterministic() -> None:
    """Same input must yield exactly the same vector across calls.

    Why: ranking tests depend on this; a stochastic embedder would make R×I×R
    cases flake.
    """
    svc = FakeEmbeddingService()
    a = await svc.embed("Write fib(n).")
    b = await svc.embed("Write fib(n).")
    assert a == b


async def test_fake_embedding_is_unit_norm() -> None:
    svc = FakeEmbeddingService()
    vec = await svc.embed("anything")
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)


async def test_fake_embedding_distinguishes_inputs() -> None:
    svc = FakeEmbeddingService()
    a = await svc.embed("Write fib(n).")
    b = await svc.embed("Sort a list.")
    assert a != b


async def test_fake_embedding_custom_dim() -> None:
    svc = FakeEmbeddingService(dim=64)
    vec = await svc.embed("x")
    assert len(vec) == 64


def test_cosine_similarity_identical_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(a, a), 1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-9)


def test_cosine_similarity_opposite_vectors() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert math.isclose(cosine_similarity(a, b), -1.0)


def test_cosine_similarity_zero_vector() -> None:
    """Zero vector -> 0 similarity (no crash, no NaN)."""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_default_embedder_returns_fake_in_fake_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """conftest sets AGENT_LOOM_FAKE_LLM=1, so we should get the fake embedder."""
    monkeypatch.setenv("AGENT_LOOM_FAKE_LLM", "1")
    assert isinstance(default_embedder(), FakeEmbeddingService)


def test_default_embedder_falls_back_to_fake_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No API key + no explicit fake flag -> still fake (graceful)."""
    monkeypatch.delenv("AGENT_LOOM_FAKE_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(default_embedder(), FakeEmbeddingService)


def test_openai_service_locks_dimension() -> None:
    """OpenAI text-embedding-3-small is locked to 1536-d; any other dim must reject."""
    with pytest.raises(ValueError, match="1536"):
        OpenAIEmbeddingService(dim=512)


async def test_openai_service_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-mode call without API key must fail loud, not silently embed garbage."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_LOOM_FAKE_LLM", raising=False)
    svc = OpenAIEmbeddingService()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await svc.embed("anything")


def test_default_embedder_returns_openai_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real mode + key present -> OpenAIEmbeddingService.

    Why: the factory's last branch is hard to hit otherwise.
    """
    monkeypatch.delenv("AGENT_LOOM_FAKE_LLM", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    svc = default_embedder()
    assert isinstance(svc, OpenAIEmbeddingService)


def test_default_embedder_respects_settings_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings-side fake flag still wins even without the env var."""
    monkeypatch.delenv("AGENT_LOOM_FAKE_LLM", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from agent_loom import config

    real_get_settings = config.get_settings

    def _patched() -> config.Settings:
        s = real_get_settings()
        return s.model_copy(update={"agent_loom_fake_llm": True})

    # Patch where the factory looks it up, not the import site.
    from agent_loom.memory import embeddings as emb_module

    monkeypatch.setattr(emb_module, "get_settings", _patched)
    assert isinstance(default_embedder(), FakeEmbeddingService)
