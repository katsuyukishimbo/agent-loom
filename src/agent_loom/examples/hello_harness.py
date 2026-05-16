"""The smallest end-to-end run.

Usage:
    AGENT_LOOM_FAKE_LLM=1 python -m agent_loom.examples.hello_harness   # offline
    python -m agent_loom.examples.hello_harness                          # real LLM

The fake-mode path is what `pytest` exercises in CI. Real mode requires
ANTHROPIC_API_KEY (or OPENAI_API_KEY if you swap the default model routing in
.env).
"""

from __future__ import annotations

import asyncio
import os

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.verifier import Verifier

GOAL = "Write a Python function fib(n) that returns the n-th Fibonacci number."


def _auto_fake_if_no_keys() -> None:
    """If no API key is configured, flip on fake mode so the demo still runs.

    Why: README promises one command works after `git clone`. Surprising the
    user with an auth error after `python -m ...` would violate that.
    """
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if not has_anthropic and not has_openai:
        os.environ.setdefault("AGENT_LOOM_FAKE_LLM", "1")


async def main() -> None:
    _auto_fake_if_no_keys()
    fake_mode = os.environ.get("AGENT_LOOM_FAKE_LLM") in {"1", "true", "True"}
    mode_label = "fake" if fake_mode else "real"

    planner = Planner()
    generator = Generator()
    verifier = Verifier()
    executor = Executor(planner=planner, generator=generator, verifier=verifier)

    print(f"[hello_harness] Mode: {mode_label}")
    print(f"[hello_harness] Submitting goal: {GOAL!r}")

    run = await executor.run(GOAL)

    print(f"[hello_harness] Run status: {run.status.value}")
    print(f"[hello_harness] Iterations: {run.iterations}")
    print(f"[hello_harness] Total cost: ${run.total_cost_usd:.4f}")
    print(f"[hello_harness] Trace: runs/{run.run_id}/trace.jsonl")
    print(f"[hello_harness] Artifact: runs/{run.run_id}/artifact.txt")


if __name__ == "__main__":
    asyncio.run(main())
