"""Tests for trace persistence (TraceWriter)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agent_loom.core.trace import TraceWriter, make_event


def test_write_event_appends_jsonl(tmp_path: Path) -> None:
    run_id = uuid4()
    writer = TraceWriter(run_id, base_dir=tmp_path)
    now = datetime.utcnow()
    event = make_event(
        run_id=run_id,
        module="planner",
        kind="llm_call",
        started_at=now,
        ended_at=now,
    )
    writer.write(event)
    lines = writer.trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["module"] == "planner"
    assert parsed["run_id"] == str(run_id)


def test_write_many_preserves_order(tmp_path: Path) -> None:
    run_id = uuid4()
    writer = TraceWriter(run_id, base_dir=tmp_path)
    now = datetime.utcnow()
    events = [
        make_event(run_id=run_id, module=m, kind="llm_call", started_at=now, ended_at=now)
        for m in ("planner", "generator", "verifier")
    ]
    writer.write_many(events)
    lines = writer.trace_path.read_text(encoding="utf-8").splitlines()
    modules = [json.loads(line)["module"] for line in lines]
    assert modules == ["planner", "generator", "verifier"]


def test_save_artifact_and_summary(tmp_path: Path) -> None:
    run_id = uuid4()
    writer = TraceWriter(run_id, base_dir=tmp_path)
    artifact_path = writer.save_artifact("print('hi')\n")
    summary_path = writer.save_summary({"status": "completed", "iterations": 1})
    assert artifact_path.read_text(encoding="utf-8") == "print('hi')\n"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "status": "completed",
        "iterations": 1,
    }


def test_trace_path_exists_even_before_first_write(tmp_path: Path) -> None:
    """Why: downstream tools enumerate runs by listing trace.jsonl files."""
    run_id = uuid4()
    writer = TraceWriter(run_id, base_dir=tmp_path)
    assert writer.trace_path.exists()
