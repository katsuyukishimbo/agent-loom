"""Swarm Skills file format (SKILL.md) parsing and validation.

Phase 3 implementation target.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class SkillExecutionBounds(BaseModel):
    max_llm_calls: int = 5
    max_tool_calls: int = 10


class SkillSelfEvolution(BaseModel):
    success_count: int = 0
    failure_count: int = 0
    last_patched: date | None = None
    multi_dim_score: dict[Literal["correctness", "cost", "readability"], float] = Field(
        default_factory=dict
    )


class SkillApplicableWhen(BaseModel):
    """Predicates that match a SprintContract to this skill."""

    language: str | None = None
    failure_type: str | None = None
    task_signature: str | None = None
    tags: list[str] = Field(default_factory=list)


class Skill(BaseModel):
    """A single SKILL.md frontmatter, parsed."""

    name: str
    version: int = 1
    description: str
    applicable_when: SkillApplicableWhen = Field(default_factory=SkillApplicableWhen)
    roles: list[Literal["planner", "generator", "verifier"]] = Field(default_factory=list)
    execution_bounds: SkillExecutionBounds = Field(default_factory=SkillExecutionBounds)
    self_evolution: SkillSelfEvolution = Field(default_factory=SkillSelfEvolution)
    body: str = Field(default="", description="The Markdown body after the frontmatter.")

    def matches(self, *, language: str | None = None, failure_type: str | None = None) -> bool:
        aw = self.applicable_when
        if aw.language and language and aw.language != language:
            return False
        if aw.failure_type and failure_type and aw.failure_type != failure_type:
            return False
        return True
