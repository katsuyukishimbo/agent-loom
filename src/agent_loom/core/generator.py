"""Generator — produces the artifact from a SprintContract.

Clean Context invariant: the Generator's input is the SprintContract and
nothing else. We do not accept or thread chat history. Skills are a Phase 2+
feature; the `skills` kwarg is accepted for forward-compat but currently
ignored (passed through as metadata).
"""

from __future__ import annotations

import json
from datetime import datetime

from agent_loom.config import get_settings
from agent_loom.core.trace import make_event
from agent_loom.core.types import Artifact, SprintContract, TraceEvent
from agent_loom.llm import complete
from agent_loom.prompts import load_prompt


def _render_contract_for_generator(contract: SprintContract) -> str:
    """Render only the fields the Generator needs. No run_id, no contract_id.

    Why hide ids: the Generator does not need them. Withholding them is one
    extra small reinforcement of Clean Context.
    """
    payload = {
        "goal": contract.goal,
        "non_goals": contract.non_goals,
        "acceptance_criteria": contract.acceptance_criteria,
        "target_files": contract.target_files,
        "forbidden": contract.forbidden,
        "persona": contract.persona,
    }
    return json.dumps(payload, indent=2)


class Generator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or get_settings().generator_model

    async def generate(
        self,
        *,
        contract: SprintContract,
        skills: list[str] | None = None,
    ) -> tuple[Artifact, TraceEvent]:
        system = load_prompt("generator")
        user = _render_contract_for_generator(contract)
        started_at = datetime.utcnow()
        resp = await complete(model=self.model, system=system, user=user, role="generator")
        ended_at = datetime.utcnow()

        artifact_data = dict(resp.parsed)
        artifact_data["contract_id"] = contract.contract_id
        artifact = Artifact(**artifact_data)

        event = make_event(
            run_id=contract.run_id,
            module="generator",
            kind="llm_call",
            started_at=started_at,
            ended_at=ended_at,
            inputs={"contract_id": str(contract.contract_id), "skills": skills or []},
            outputs={
                "artifact_id": str(artifact.artifact_id),
                "kind": artifact.kind,
                "files_touched": artifact.files_touched,
            },
            cost_usd=resp.cost_usd,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            provider=resp.provider,
            model=resp.model,
        )
        return artifact, event
