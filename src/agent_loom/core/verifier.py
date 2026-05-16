"""Verifier — Clean Context grader.

Critical invariant: `verify()`'s signature accepts ONLY SprintContract, the
resulting Artifact, and a read-only list of TraceEvent. No chat history, no
free-form context dict, no Generator messages. The signature itself is the
contract — the runtime check in `_assert_clean_context` mirrors it so that
subclassing or attribute injection can't sneak a backdoor.

Phase 0 produces a top-down rubric judgement. Bottom-up per-span classification
is wired in Phase 2.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime

from agent_loom.config import get_settings
from agent_loom.core.trace import make_event
from agent_loom.core.types import (
    Artifact,
    FailureCategory,
    SprintContract,
    TraceEvent,
    VerifierJudgement,
)
from agent_loom.llm import complete
from agent_loom.prompts import load_prompt

# The exact, locked set of parameter names the Verifier may accept.
# Why a module-level constant: keep the rule visible to anyone reading the file.
_ALLOWED_VERIFY_PARAMS: frozenset[str] = frozenset(
    {"self", "contract", "artifact", "trace"}
)


class CleanContextViolation(TypeError):
    """Raised when a subclass tries to add forbidden parameters to verify()."""


def _assert_clean_context(cls: type) -> None:
    """Enforce the Clean Context invariant on Verifier and subclasses.

    Why both signature inspection and __init_subclass__: a typo-as-API like
    `chat_history=None` could otherwise smuggle Generator state through a
    kwarg. We fail loudly at class-construction time.
    """
    sig = inspect.signature(cls.verify)
    actual = set(sig.parameters.keys())
    extras = actual - _ALLOWED_VERIFY_PARAMS
    if extras:
        raise CleanContextViolation(
            f"{cls.__name__}.verify must only accept "
            f"{sorted(_ALLOWED_VERIFY_PARAMS - {'self'})}; got extra params: {sorted(extras)}"
        )


def _render_trace_summary(trace: list[TraceEvent]) -> str:
    """Compress trace into a short list of {module, kind, model, cost} tuples.

    Why summarise instead of dumping full inputs/outputs: the Verifier needs to
    know *what happened* (which modules ran, did anything fail) but not the raw
    LLM messages. Passing full chat history would violate Clean Context.
    """
    lines = []
    for e in trace:
        lines.append(
            f"- module={e.module} kind={e.kind} model={e.model or 'n/a'} "
            f"cost=${e.cost_usd:.4f}"
        )
    return "\n".join(lines) or "(no events)"


class Verifier:
    def __init__(
        self,
        model: str | None = None,
        ship_threshold: float = 0.85,
    ) -> None:
        self.model = model or get_settings().verifier_model
        self.ship_threshold = ship_threshold

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _assert_clean_context(cls)

    async def verify(
        self,
        *,
        contract: SprintContract,
        artifact: Artifact,
        trace: list[TraceEvent],
    ) -> tuple[VerifierJudgement, TraceEvent]:
        system = load_prompt("verifier")
        user = json.dumps(
            {
                "contract": {
                    "goal": contract.goal,
                    "non_goals": contract.non_goals,
                    "acceptance_criteria": contract.acceptance_criteria,
                    "forbidden": contract.forbidden,
                },
                "artifact": {
                    "kind": artifact.kind,
                    "content": artifact.content,
                    "files_touched": artifact.files_touched,
                },
                "trace_summary": _render_trace_summary(trace),
            },
            indent=2,
        )
        started_at = datetime.utcnow()
        resp = await complete(model=self.model, system=system, user=user, role="verifier")
        ended_at = datetime.utcnow()

        parsed = dict(resp.parsed)
        # Normalise failure_category: the model may emit null/empty string.
        raw_cat = parsed.get("failure_category")
        if raw_cat in (None, "", "null"):
            parsed["failure_category"] = None
        else:
            parsed["failure_category"] = FailureCategory(raw_cat)
        parsed["artifact_id"] = artifact.artifact_id
        # span_judgements and judge_confirmed are Phase 2; leave defaults.
        judgement = VerifierJudgement(**parsed)

        event = make_event(
            run_id=contract.run_id,
            module="verifier",
            kind="llm_call",
            started_at=started_at,
            ended_at=ended_at,
            inputs={
                "contract_id": str(contract.contract_id),
                "artifact_id": str(artifact.artifact_id),
            },
            outputs={
                "passed": judgement.passed,
                "score": judgement.score,
                "failure_category": (
                    judgement.failure_category.value if judgement.failure_category else None
                ),
            },
            cost_usd=resp.cost_usd,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            provider=resp.provider,
            model=resp.model,
        )
        return judgement, event


# Run the check on the base class itself so the rule is honored from import time.
_assert_clean_context(Verifier)
