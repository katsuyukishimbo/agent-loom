"""Smoke tests for the type system. Run with `pytest`."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.memory.store import Episode, importance_normalized, recency_score


def test_sprint_contract_minimal() -> None:
    contract = SprintContract(
        run_id=uuid4(),
        goal="Write a function that returns Fibonacci(n).",
        acceptance_criteria=["fib(10) == 55"],
    )
    assert contract.max_cost_usd == 2.0
    assert "pragmatic senior engineer" in contract.persona


def test_artifact_kinds() -> None:
    contract_id = uuid4()
    artifact = Artifact(contract_id=contract_id, kind="diff", content="--- a/fib.py\n+++ b/fib.py")
    assert artifact.kind == "diff"


def test_verifier_judgement_score_bounds() -> None:
    with pytest.raises(ValidationError):
        VerifierJudgement(artifact_id=uuid4(), passed=False, score=1.5)


def test_failure_category_enum() -> None:
    assert FailureCategory.SPEC_MISREAD.value == "spec_misread"


def test_recency_score_stepwise() -> None:
    now = datetime.utcnow()
    fresh = Episode(content="x", importance=5.0, last_referenced_at=now)
    week_old = Episode(content="y", importance=5.0, last_referenced_at=now - timedelta(days=4))
    month_old = Episode(content="z", importance=5.0, last_referenced_at=now - timedelta(days=25))
    assert recency_score(fresh, now=now) == 1.0
    assert recency_score(week_old, now=now) == 0.8
    assert recency_score(month_old, now=now) == 0.5


def test_importance_normalized() -> None:
    e = Episode(content="x", importance=7.5)
    assert importance_normalized(e) == 0.75
