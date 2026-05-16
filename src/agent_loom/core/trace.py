"""Trace persistence — JSONL append-only.

Writes are single-threaded: only the Executor owns a TraceWriter. Planner /
Generator / Verifier return TraceEvent objects (as values) and never touch the
filesystem. That preserves the "Writes single-threaded" invariant from the
Cognition harness paper.

One trace file per run: `runs/<run_id>/trace.jsonl`. Append-only so replays can
re-read it cheaply, and a crash mid-run still leaves a partial trace on disk.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from agent_loom.core.types import TraceEvent

Module = Literal["planner", "generator", "verifier", "executor"]
Kind = Literal["llm_call", "tool_call", "memory_read", "memory_write", "skill_load"]


class TraceWriter:
    """Append-only JSONL writer scoped to one run."""

    def __init__(self, run_id: UUID, base_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = base_dir / str(run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        # Why touch on init: a run that fails before any event still produces a
        # discoverable empty trace file. Easier to reason about than absence.
        self.trace_path.touch(exist_ok=True)

    def write(self, event: TraceEvent) -> None:
        """Persist a single event. Caller is responsible for constructing it."""
        line = event.model_dump_json()
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def write_many(self, events: list[TraceEvent]) -> None:
        for event in events:
            self.write(event)

    def save_artifact(self, content: str, filename: str = "artifact.txt") -> Path:
        """Save the final artifact next to the trace."""
        path = self.run_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def save_summary(self, summary: dict[str, Any]) -> Path:
        """Save a JSON summary of the run (Run model serialised)."""
        path = self.run_dir / "summary.json"
        path.write_text(json.dumps(summary, default=str, indent=2), encoding="utf-8")
        return path


def make_event(
    *,
    run_id: UUID,
    module: Module,
    kind: Kind,
    started_at: datetime,
    ended_at: datetime,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    cost_usd: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    provider: str = "",
    model: str = "",
    parent_id: UUID | None = None,
) -> TraceEvent:
    """Tiny ergonomic constructor that pre-fills the boilerplate fields.

    Why a function and not subclassing TraceEvent: types.py is locked. Stay out
    of its way and let modules use kwargs.
    """
    return TraceEvent(
        event_id=uuid4(),
        run_id=run_id,
        parent_id=parent_id,
        module=module,
        kind=kind,
        started_at=started_at,
        ended_at=ended_at,
        inputs=inputs or {},
        outputs=outputs or {},
        cost_usd=cost_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        provider=provider,
        model=model,
    )
