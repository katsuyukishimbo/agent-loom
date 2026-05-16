"""Verifier unit tests (fake mode) + Clean Context invariant tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agent_loom.core.types import Artifact, SprintContract, TraceEvent, VerifierJudgement
from agent_loom.core.verifier import CleanContextViolation, Verifier


def _make_contract_and_artifact() -> tuple[SprintContract, Artifact]:
    contract = SprintContract(
        run_id=uuid4(),
        goal="Write fib(n).",
        acceptance_criteria=["fib(10) == 55"],
    )
    artifact = Artifact(
        contract_id=contract.contract_id,
        kind="text",
        content="def fib(n): ...",
    )
    return contract, artifact


async def test_verify_returns_judgement_and_event() -> None:
    contract, artifact = _make_contract_and_artifact()
    judgement, event = await Verifier(model="claude-sonnet-4-6").verify(
        contract=contract, artifact=artifact, trace=[]
    )
    assert isinstance(judgement, VerifierJudgement)
    assert isinstance(event, TraceEvent)
    assert judgement.artifact_id == artifact.artifact_id
    assert 0.0 <= judgement.score <= 1.0


async def test_verify_signature_is_locked_to_clean_context() -> None:
    """Subclassing with extra params must blow up at class-construction time.

    Why: this is the one place where someone could accidentally smuggle the
    Generator's chat history into the Verifier. Test the guard exists.
    """
    with pytest.raises(CleanContextViolation):

        class LeakyVerifier(Verifier):
            async def verify(  # type: ignore[override]
                self,
                *,
                contract: SprintContract,
                artifact: Artifact,
                trace: list[TraceEvent],
                chat_history: list[str] | None = None,  # the forbidden param
            ) -> tuple[VerifierJudgement, TraceEvent]:
                raise NotImplementedError


def test_verify_signature_runtime_check_lists_extras() -> None:
    """The error message must name the offending parameter so reviewers can see it."""
    with pytest.raises(CleanContextViolation, match="chat_history"):

        class AnotherLeakyVerifier(Verifier):
            async def verify(  # type: ignore[override]
                self,
                *,
                contract: SprintContract,
                artifact: Artifact,
                trace: list[TraceEvent],
                chat_history: list[str] | None = None,
            ) -> tuple[VerifierJudgement, TraceEvent]:
                raise NotImplementedError


async def test_failure_category_normalisation_from_null_string() -> None:
    """Models occasionally emit 'null' or '' instead of JSON null; normalise both."""
    contract, artifact = _make_contract_and_artifact()
    judgement, _ = await Verifier().verify(contract=contract, artifact=artifact, trace=[])
    assert judgement.failure_category is None


def test_render_trace_summary_with_events() -> None:
    """The trace summary helper builds one line per event."""
    from datetime import datetime

    from agent_loom.core.trace import make_event
    from agent_loom.core.verifier import _render_trace_summary

    now = datetime.utcnow()
    events = [
        make_event(
            run_id=uuid4(),
            module="generator",
            kind="llm_call",
            started_at=now,
            ended_at=now,
            model="claude-haiku-4-5",
            cost_usd=0.01,
        )
    ]
    summary = _render_trace_summary(events)
    assert "module=generator" in summary
    assert "claude-haiku-4-5" in summary


def test_render_trace_summary_empty() -> None:
    from agent_loom.core.verifier import _render_trace_summary

    assert _render_trace_summary([]) == "(no events)"


async def test_failure_category_parses_enum_value(monkeypatch) -> None:
    """When the LLM returns a real category name, we parse it into the enum."""
    from agent_loom import llm

    async def _fake_complete(*, model, system, user, role, max_tokens=2048):
        return llm.LLMResponse(
            text="{}",
            parsed={
                "passed": False,
                "score": 0.2,
                "rubric_breakdown": {"correctness": 0.2},
                "failure_category": "spec_misread",
                "reflection": "Wrong problem.",
            },
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0,
            provider="fake",
            model="fake",
        )

    monkeypatch.setattr("agent_loom.core.verifier.complete", _fake_complete)
    contract, artifact = _make_contract_and_artifact()
    judgement, _ = await Verifier().verify(contract=contract, artifact=artifact, trace=[])
    from agent_loom.core.types import FailureCategory

    assert judgement.failure_category == FailureCategory.SPEC_MISREAD
