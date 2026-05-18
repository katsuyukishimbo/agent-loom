"""Unified async LLM client used by Planner / Generator / Verifier.

Why a thin wrapper instead of LangChain: each module makes exactly one call, with
a system prompt and a user prompt, expecting a JSON object back. A 60-line
wrapper is more honest than 50 MB of indirection — and we can swap providers per
module without ceremony.

Fake mode (AGENT_LOOM_FAKE_LLM=1) returns canned responses keyed by `role` so
end-to-end tests and hello_harness work with zero API keys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from agent_loom.config import cost_for, get_settings

Role = Literal["planner", "generator", "verifier", "reflection"]


@dataclass
class LLMResponse:
    """One LLM call result. `parsed` is the JSON object extracted from text."""

    text: str
    parsed: dict[str, Any]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    provider: str
    model: str


def _is_fake_mode() -> bool:
    """Fake mode wins if either env var or settings flag is set.

    Why both: tests typically set the env var via conftest; programmatic callers
    (hello_harness) may flip the settings flag instead.
    """
    if os.environ.get("AGENT_LOOM_FAKE_LLM", "").strip() in {"1", "true", "True"}:
        return True
    try:
        return bool(get_settings().agent_loom_fake_llm)
    except Exception:
        return False


def _provider_for(model: str) -> str:
    if model.startswith("claude") or model == "fake":
        return "anthropic"
    if model.startswith("gpt"):
        return "openai"
    return "unknown"


# --- Fake responses --------------------------------------------------------

_FAKE_FIB_CODE = '''def fib(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''


def _fake_response(role: Role, user_prompt: str) -> dict[str, Any]:
    """Canned per-role JSON payloads.

    Why per-role and not per-prompt: the harness shape is what we want to test
    in fake mode, not specific prompt content. Each role returns the minimum
    valid shape its caller can `model_validate`.
    """
    if role == "planner":
        # Echo the user goal into acceptance criteria — keeps the contract honest
        # enough that downstream tests can assert traceability.
        return {
            "goal": user_prompt.strip()[:200] or "Write Fibonacci function.",
            "non_goals": ["No memoization, no recursion-depth tuning."],
            "acceptance_criteria": [
                "Function fib(n: int) -> int is defined.",
                "fib(10) == 55",
            ],
            "target_files": ["fib.py"],
            "forbidden": [],
        }
    if role == "generator":
        return {
            "kind": "text",
            "content": _FAKE_FIB_CODE,
            "files_touched": ["fib.py"],
            "notes": "Iterative O(n) implementation. Fake-mode canned output.",
        }
    if role == "verifier":
        return {
            "passed": True,
            "score": 0.9,
            "rubric_breakdown": {"correctness": 0.95, "readability": 0.85},
            "failure_category": None,
            "reflection": "Iterative form satisfies fib(10) == 55. Fake-mode pass.",
        }
    if role == "reflection":
        # Reflective Compaction fake payload. Phase 2 expects a one-paragraph
        # summary plus an importance score in [8, 10]; we return a constant
        # high-importance value so tests can assert the failure episode gets
        # written with the elevated weight.
        return {
            "summary": (
                "Past failure summary (fake mode): the artifact did not satisfy the "
                "acceptance criteria. Future plans must avoid this failure mode."
            ),
            "importance": 9.0,
        }
    raise ValueError(f"Unknown role: {role}")


# --- Real providers --------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Why be lenient: even when we ask for pure JSON, models occasionally wrap it
    in ```json fences or add a one-line preface. We slice between the first `{`
    and the matching last `}` rather than relying on regex.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    blob = text[start : end + 1]
    return json.loads(blob)


async def _call_anthropic(
    *, model: str, system: str, user: str, max_tokens: int
) -> LLMResponse:
    from anthropic import AsyncAnthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set; cannot call Anthropic in real mode.")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    msg = await client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    # SDK shape: msg.content is a list of content blocks; we use the first text block.
    text = "".join(getattr(block, "text", "") for block in msg.content)
    tokens_in = getattr(msg.usage, "input_tokens", 0) or 0
    tokens_out = getattr(msg.usage, "output_tokens", 0) or 0
    return LLMResponse(
        text=text,
        parsed=_extract_json(text),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_for(model, tokens_in, tokens_out),
        provider="anthropic",
        model=model,
    )


async def _call_openai(
    *, model: str, system: str, user: str, max_tokens: int
) -> LLMResponse:
    from openai import AsyncOpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call OpenAI in real mode.")

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    tokens_in = resp.usage.prompt_tokens if resp.usage else 0
    tokens_out = resp.usage.completion_tokens if resp.usage else 0
    return LLMResponse(
        text=text,
        parsed=_extract_json(text),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_for(model, tokens_in, tokens_out),
        provider="openai",
        model=model,
    )


# --- Public entry point ----------------------------------------------------


async def complete(
    *,
    model: str,
    system: str,
    user: str,
    role: Role,
    max_tokens: int = 2048,
) -> LLMResponse:
    """Call the configured provider for `model`. Fake mode short-circuits.

    `role` is required so fake mode knows which canned payload to return; in
    real mode it's used only for provenance.
    """
    if _is_fake_mode() or model == "fake":
        parsed = _fake_response(role, user)
        text = json.dumps(parsed)
        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=len(system) // 4 + len(user) // 4,
            tokens_out=len(text) // 4,
            cost_usd=0.0,
            provider="fake",
            model="fake",
        )

    provider = _provider_for(model)
    if provider == "anthropic":
        return await _call_anthropic(model=model, system=system, user=user, max_tokens=max_tokens)
    if provider == "openai":
        return await _call_openai(model=model, system=system, user=user, max_tokens=max_tokens)
    raise ValueError(f"Unrecognized model name (no provider mapping): {model!r}")
