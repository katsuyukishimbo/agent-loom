"""Runtime configuration loaded from environment variables.

Why pydantic-settings: a single source of truth for env vars + typed defaults.
Anything that needs a model name, budget, or path should pull from here rather
than reading os.environ directly. That keeps fake-mode and real-mode swappable
via one knob (AGENT_LOOM_FAKE_LLM).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class _PricePerMTok(dict[str, tuple[float, float]]):
    """Per-million-token (input_price, output_price) for known models in USD."""


# Why hard-coded: Phase 0 only needs rough cost gating. A real price oracle is
# Phase 3 (`benchmarks/`). The numbers below are conservative ceilings as of
# 2026-05-16; bump them in one place if pricing changes.
MODEL_PRICING: _PricePerMTok = _PricePerMTok(
    {
        "claude-opus-4-7": (15.0, 75.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5": (0.80, 4.0),
        "gpt-5.5": (5.0, 20.0),
        "gpt-5.3-codex": (2.0, 8.0),
        # Fake mode model name — zero cost so tests don't assert against pricing drift.
        "fake": (0.0, 0.0),
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    planner_model: str = "claude-opus-4-7"
    generator_model: str = "claude-haiku-4-5"
    verifier_model: str = "claude-sonnet-4-6"

    default_max_cost_per_run_usd: float = 2.0
    default_max_llm_calls_per_run: int = 50

    log_level: str = "INFO"
    trace_dir: Path = Path("./runs")

    # AGENT_LOOM_FAKE_LLM=1 forces every llm.complete() call to return canned
    # responses. Used by tests and by hello_harness when no API key is set.
    agent_loom_fake_llm: bool = False

    # Phase 1b — episodic store backend.
    # `database_url` is read but ALSO ignored when AGENT_LOOM_USE_PG is unset
    # (so existing in-memory tests don't accidentally hit a running Postgres).
    database_url: str | None = None
    agent_loom_use_pg: bool = False


def get_settings() -> Settings:
    """Factory so tests can monkeypatch env between cases."""
    return Settings()


def cost_for(model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute USD cost for a single LLM call. Unknown models cost $0 (logged elsewhere)."""
    if model not in MODEL_PRICING:
        return 0.0
    in_price, out_price = MODEL_PRICING[model]
    return (tokens_in / 1_000_000) * in_price + (tokens_out / 1_000_000) * out_price
