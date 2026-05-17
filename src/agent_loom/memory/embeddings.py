"""Embedding services for memory retrieval.

Two implementations:
- `FakeEmbeddingService` — deterministic SHA-256 based, no network. Same input
  always yields the same vector, so cosine similarity tests are stable.
- `OpenAIEmbeddingService` — real provider, model `text-embedding-3-small`
  (1536 dimensions). Mirrors the `llm.py` lazy-import + Settings pattern so
  fake mode never needs an OpenAI key.

Why a protocol rather than a base class: tests want to inject their own embedder
and the harness only depends on the two-method shape. Inheritance would force
the in-memory store to know about provider details it does not care about.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol, runtime_checkable

from agent_loom.config import get_settings

EMBEDDING_DIM = 1536


@runtime_checkable
class EmbeddingService(Protocol):
    """Anything that can turn text into a fixed-length vector."""

    dim: int

    async def embed(self, text: str) -> list[float]:
        """Return a unit-norm vector of length `self.dim` for `text`."""
        ...


# --- Fake (deterministic) -------------------------------------------------


def _hash_to_unit_vector(text: str, dim: int) -> list[float]:
    """Map `text` to a deterministic unit vector of length `dim`.

    Why SHA-256 tiled: it gives us a reproducible byte stream we can slice into
    `dim` floats without depending on numpy. Centering each byte at zero
    (subtract 128) and normalising to unit length keeps cosine similarity
    well-defined.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 32 bytes per SHA-256; tile until we have at least `dim` bytes.
    repeats = (dim // len(digest)) + 1
    stream = (digest * repeats)[:dim]
    raw = [(b - 128) / 128.0 for b in stream]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0.0:
        # Degenerate input — shouldn't happen for non-empty UTF-8 bytes, but
        # guard against division by zero anyway.
        return [0.0] * dim
    return [x / norm for x in raw]


class FakeEmbeddingService:
    """Hash-based deterministic embedder. Used by tests and fake mode."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        return _hash_to_unit_vector(text, self.dim)


# --- Real provider --------------------------------------------------------


class OpenAIEmbeddingService:
    """OpenAI `text-embedding-3-small` (1536-d).

    The actual HTTP call is lazy-imported so test environments without the SDK
    installed still work in fake mode.
    """

    MODEL = "text-embedding-3-small"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"OpenAIEmbeddingService is locked to {EMBEDDING_DIM}-d "
                f"({self.MODEL}); got dim={dim}."
            )
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        from openai import AsyncOpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; cannot call OpenAI embeddings in real mode."
            )
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.embeddings.create(model=self.MODEL, input=text)
        return list(resp.data[0].embedding)


# --- Factory --------------------------------------------------------------


def default_embedder() -> EmbeddingService:
    """Pick fake or real based on the same env flag as `llm.py`.

    Why a factory: callers (MemoryHub, hello_harness) shouldn't repeat the
    env-checking logic. One place to flip.
    """
    if os.environ.get("AGENT_LOOM_FAKE_LLM", "").strip() in {"1", "true", "True"}:
        return FakeEmbeddingService()
    try:
        if get_settings().agent_loom_fake_llm:
            return FakeEmbeddingService()
    except Exception:
        pass
    if not get_settings().openai_api_key:
        # No key available — fall back to fake rather than crashing at first
        # use. Mirrors hello_harness's _auto_fake_if_no_keys behaviour.
        return FakeEmbeddingService()
    return OpenAIEmbeddingService()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product on unit vectors == cosine similarity.

    Why we still divide: callers may pass non-unit vectors (e.g. a freshly
    computed query embedding that hasn't been normalised). We compute the full
    formula so the helper is correct regardless.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
