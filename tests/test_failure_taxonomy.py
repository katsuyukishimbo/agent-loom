"""Verifier emits one of the 7 FailureCategory values.

We exercise the parser by stubbing the LLM response to each enum value and
asserting it parses through to the Pydantic model. The verifier prompt
itself is asserted to mention every category so the model knows to pick from.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    VerifierJudgement,
)
from agent_loom.core.verifier import Verifier
from agent_loom.prompts import load_prompt


@pytest.fixture
def fixture() -> tuple[SprintContract, Artifact]:
    contract = SprintContract(
        run_id=uuid4(),
        goal="trivial",
        acceptance_criteria=["always fail in this test"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id,
        kind="text",
        content="anything",
    )
    return contract, artifact


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.SPEC_MISREAD,
        FailureCategory.PARTIAL_IMPLEMENTATION,
        FailureCategory.HALLUCINATED_ARTIFACT,
        FailureCategory.TOOL_ERROR,
        FailureCategory.CONTEXT_ROT,
        FailureCategory.EVAL_DRIFT,
        FailureCategory.COST_BUDGET_EXCEEDED,
    ],
)
async def test_verifier_round_trips_each_failure_category(
    fixture, monkeypatch: pytest.MonkeyPatch, category: FailureCategory
) -> None:
    """For each of the 7 enum values, a stubbed Verifier response must parse."""
    from agent_loom import llm
    from agent_loom.core import verifier as verifier_module

    async def _stub(*, model, system, user, role, max_tokens=2048):
        return llm.LLMResponse(
            text="{}",
            parsed={
                "passed": False,
                "score": 0.2,
                "rubric_breakdown": {"x": 0.2},
                "failure_category": category.value,
                "reflection": f"forced category {category.value}",
            },
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0,
            provider="fake",
            model="fake",
        )

    monkeypatch.setattr(verifier_module, "complete", _stub)
    contract, artifact = fixture
    verifier = Verifier()
    judgement, _ = await verifier.verify(contract=contract, artifact=artifact, trace=[])
    assert isinstance(judgement, VerifierJudgement)
    assert judgement.passed is False
    assert judgement.failure_category == category


def test_verifier_prompt_documents_all_seven_categories() -> None:
    prompt = load_prompt("verifier")
    for cat in FailureCategory:
        assert cat.value in prompt, f"Verifier prompt missing {cat.value!r}"


def test_verifier_prompt_marks_failure_category_as_required_on_failure() -> None:
    """The prompt must enforce 'category required when failed' to avoid null drift."""
    prompt = load_prompt("verifier")
    assert "REQUIRED when `passed` is false" in prompt
