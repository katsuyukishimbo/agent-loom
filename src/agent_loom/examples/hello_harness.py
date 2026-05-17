"""The smallest end-to-end run.

Usage:
    AGENT_LOOM_FAKE_LLM=1 python -m agent_loom.examples.hello_harness   # offline
    python -m agent_loom.examples.hello_harness                          # real LLM
    AGENT_LOOM_USE_PG=1 python -m agent_loom.examples.hello_harness      # pgvector

The fake-mode path is what `pytest` exercises in CI. Real mode requires
ANTHROPIC_API_KEY (or OPENAI_API_KEY if you swap the default model routing in
.env).

Phase 1b: when `AGENT_LOOM_USE_PG=1` (or `--store pg` is passed), episodes are
persisted to pgvector. Running this script twice across separate processes
demonstrates cross-process recall — the second invocation reads back the first
invocation's episode through the database. Without that flag the script uses
the original in-memory store and only demonstrates same-process recall.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from agent_loom.core.executor import Executor
from agent_loom.core.generator import Generator
from agent_loom.core.planner import Planner
from agent_loom.core.verifier import Verifier
from agent_loom.memory.embeddings import FakeEmbeddingService, default_embedder
from agent_loom.memory.hub import MemoryHub
from agent_loom.memory.store import InMemoryEpisodicStore

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flag for store selection.

    Why a flag in addition to the env var: the test suite shells out to this
    module with explicit subprocess args and we don't want it leaking env
    state into the child. The env var is the user-facing knob, the flag is
    the test harness's knob.
    """
    parser = argparse.ArgumentParser(prog="hello_harness")
    parser.add_argument(
        "--store",
        choices=("auto", "memory", "pg"),
        default="auto",
        help=(
            "Episodic store backend. 'auto' picks pg when AGENT_LOOM_USE_PG=1 "
            "or DATABASE_URL is reachable, else memory."
        ),
    )
    return parser.parse_args(argv)


def _build_memory_hub(store_choice: str, fake_mode: bool) -> tuple[MemoryHub, str]:
    """Return (hub, label). Label is printed for visibility in run logs.

    The pg branch deliberately uses a real embedder fallback chain (default
    embedder picks fake if no key) so fake-mode + pg-mode compose cleanly.
    """
    if store_choice == "memory":
        return (MemoryHub.fake() if fake_mode else MemoryHub()), "memory"

    use_pg = store_choice == "pg" or os.environ.get("AGENT_LOOM_USE_PG") in {
        "1",
        "true",
        "True",
    }
    if use_pg:
        from agent_loom.memory.store_pg import PgvectorEpisodicStore

        store = PgvectorEpisodicStore()
        embedder = FakeEmbeddingService() if fake_mode else default_embedder()
        return MemoryHub(store=store, embedder=embedder), "pg"

    return (MemoryHub.fake() if fake_mode else MemoryHub()), "memory"


async def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _auto_fake_if_no_keys()
    fake_mode = os.environ.get("AGENT_LOOM_FAKE_LLM") in {"1", "true", "True"}
    mode_label = "fake" if fake_mode else "real"

    memory_hub, store_label = _build_memory_hub(args.store, fake_mode)
    # Silence unused-import warning for InMemoryEpisodicStore — kept available
    # for callers that import the helper module directly.
    _ = InMemoryEpisodicStore

    planner = Planner(memory_hub=memory_hub)
    generator = Generator()
    verifier = Verifier()
    executor = Executor(
        planner=planner,
        generator=generator,
        verifier=verifier,
        memory_hub=memory_hub,
    )

    print(f"[hello_harness] Mode: {mode_label}  Store: {store_label}")
    print(f"[hello_harness] Submitting goal: {GOAL!r}")

    run = await executor.run(GOAL)

    print(f"[hello_harness] Run status: {run.status.value}")
    print(f"[hello_harness] Iterations: {run.iterations}")
    print(f"[hello_harness] Total cost: ${run.total_cost_usd:.4f}")
    print(f"[hello_harness] Trace: runs/{run.run_id}/trace.jsonl")
    print(f"[hello_harness] Artifact: runs/{run.run_id}/artifact.txt")
    episodes = await memory_hub.store.list_all()
    print(f"[hello_harness] Memory episodes after run: {len(episodes)}")
    for ep in episodes:
        print(
            f"  - id={ep.episode_id} importance={ep.importance:.1f} "
            f"refs={ep.references_count} passed={ep.metadata.get('passed')}"
        )

    # Phase 1a recall demo: re-plan with a similar goal in the SAME process to
    # verify the episode written above is recalled. Cross-process recall lands
    # in Phase 1b with pgvector.
    print()
    print("[hello_harness] Second pass: recalling memory for a related goal...")
    similar_goal = "Implement a Python fibonacci function for integer n."
    recalled = await memory_hub.recall(similar_goal, top_k=3)
    print(f"[hello_harness] Recalled {len(recalled)} episode(s).")
    for ep in recalled:
        preview = ep.content.replace("\n", " ")[:120]
        print(f"  - refs={ep.references_count} preview={preview!r}")


if __name__ == "__main__":
    asyncio.run(main())
