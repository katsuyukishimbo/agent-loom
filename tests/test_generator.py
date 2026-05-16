"""Generator unit tests (fake mode)."""

from __future__ import annotations

from uuid import uuid4

from agent_loom.core.generator import Generator
from agent_loom.core.types import Artifact, SprintContract, TraceEvent


def _make_contract() -> SprintContract:
    return SprintContract(
        run_id=uuid4(),
        goal="Write fib(n).",
        acceptance_criteria=["fib(10) == 55"],
        target_files=["fib.py"],
    )


async def test_generate_returns_artifact_bound_to_contract() -> None:
    contract = _make_contract()
    artifact, event = await Generator(model="claude-haiku-4-5").generate(contract=contract)
    assert isinstance(artifact, Artifact)
    assert isinstance(event, TraceEvent)
    assert artifact.contract_id == contract.contract_id


async def test_generate_artifact_has_content() -> None:
    contract = _make_contract()
    artifact, _ = await Generator(model="claude-haiku-4-5").generate(contract=contract)
    assert "def fib" in artifact.content


async def test_generator_ignores_skills_kwarg_safely() -> None:
    """skills is a forward-compat passthrough; should not crash."""
    contract = _make_contract()
    artifact, event = await Generator().generate(
        contract=contract, skills=["python-typeerror-pattern"]
    )
    assert artifact.kind in ("diff", "text", "json", "tool_call_sequence")
    assert event.inputs.get("skills") == ["python-typeerror-pattern"]
