"""Pydantic models shared across the four modules.

These types are the wire format between Planner, Generator, Verifier, and Executor.
The Verifier in particular is forbidden from receiving anything outside SprintContract
and the resulting Artifact (Clean Context invariant).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """Failure taxonomy v0.1 — see docs/ARCHITECTURE.md §5."""

    SPEC_MISREAD = "spec_misread"
    PARTIAL_IMPLEMENTATION = "partial_implementation"
    HALLUCINATED_ARTIFACT = "hallucinated_artifact"
    TOOL_ERROR = "tool_error"
    CONTEXT_ROT = "context_rot"
    EVAL_DRIFT = "eval_drift"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"


class SprintContract(BaseModel):
    """Planner → Generator handoff.

    This is the *only* thing the Generator and Verifier may see about the original goal.
    """

    contract_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    goal: str = Field(description="One-sentence description of what success looks like.")
    non_goals: list[str] = Field(default_factory=list, description="Out of scope for this sprint.")
    acceptance_criteria: list[str] = Field(
        description="Concrete checks that must pass (e.g., 'pytest tests/test_x.py' or 'output is valid JSON')."
    )
    target_files: list[str] = Field(
        default_factory=list,
        description="Files the Generator may modify. Empty list means no filesystem writes.",
    )
    forbidden: list[str] = Field(
        default_factory=list,
        description="Negative constraints injected from memory's failure episodes.",
    )
    persona: str = Field(
        default="You are a pragmatic senior engineer. YAGNI and DRY. Red-green TDD.",
        description="Generator persona prompt prefix.",
    )
    max_cost_usd: float = Field(default=2.0, ge=0)
    max_llm_calls: int = Field(default=50, ge=1)


class Artifact(BaseModel):
    """Generator → Verifier handoff. The thing produced."""

    artifact_id: UUID = Field(default_factory=uuid4)
    contract_id: UUID
    kind: Literal["diff", "text", "json", "tool_call_sequence"]
    content: str = Field(description="The actual output. Format depends on `kind`.")
    files_touched: list[str] = Field(default_factory=list)
    notes: str = Field(default="", description="Generator's brief commentary, not the artifact itself.")


class VerifierJudgement(BaseModel):
    """Verifier → Executor handoff. The grading."""

    artifact_id: UUID
    passed: bool
    score: float = Field(ge=0, le=1, description="Top-down rubric score [0, 1].")
    rubric_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion sub-scores summing roughly to `score`.",
    )
    span_judgements: list["SpanJudgement"] = Field(
        default_factory=list,
        description="Bottom-up per-trace-event judgements (Holistic Evaluation).",
    )
    failure_category: FailureCategory | None = None
    reflection: str = Field(
        default="",
        description="One-paragraph 'what went wrong and why' — written to memory on failure.",
    )
    judge_confirmed: bool | None = Field(
        default=None,
        description="If a cross-provider Judge was run, did it agree?",
    )


class SpanJudgement(BaseModel):
    """Bottom-up classification of a single TraceEvent (Holistic Evaluation)."""

    event_id: UUID
    label: Literal["ok", "suboptimal", "hallucinated", "tool_error", "missing_step"]
    rationale: str = ""


class TraceEvent(BaseModel):
    """One observable atom in a run. Persisted to PostgreSQL and replayable."""

    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    parent_id: UUID | None = None
    module: Literal["planner", "generator", "verifier", "executor"]
    kind: Literal["llm_call", "tool_call", "memory_read", "memory_write", "skill_load"]
    started_at: datetime
    ended_at: datetime
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    provider: str = ""
    model: str = ""


class RunStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class Run(BaseModel):
    """A single user-submitted goal, end to end."""

    run_id: UUID = Field(default_factory=uuid4)
    user_goal: str
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    final_artifact_id: UUID | None = None
    total_cost_usd: float = 0.0
    iterations: int = 0
