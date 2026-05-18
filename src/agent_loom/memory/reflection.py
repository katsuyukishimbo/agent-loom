"""Reflective Compaction (Shinn 2023's *Reflexion* pattern, MAGE write-path).

When the Verifier rejects an artifact, we ask an LLM to produce a dense
"what went wrong and why" paragraph. That paragraph becomes a high-importance
Episode in the experience subgraph and a `forbidden` constraint on the next
SprintContract.

### Clean-Context invariant for reflection

The reflection call MUST NOT see the Generator's chat history. We hand it:

- `goal` from the SprintContract
- `failure_category` from the VerifierJudgement
- `reflection` seed text (also from the Verifier)
- the first ~500 chars of the Artifact

…and nothing else. That mirrors the Phase 0/1 invariant: every LLM call sees
only the contract artefacts, never the upstream module's working memory.
"""

from __future__ import annotations

from agent_loom.config import get_settings
from agent_loom.core.types import Artifact, SprintContract, VerifierJudgement
from agent_loom.llm import complete


# Why a module-level prompt rather than a .md file: the prompt is tiny (≈10
# lines) and lives next to its only caller. Promoting it to prompts/ would
# split the implementation across two files for no reviewer benefit.
_REFLECTION_PROMPT = (
    "You are reflecting on a failed agent run.\n\n"
    "Goal: {goal}\n"
    "Failure category: {failure_category}\n"
    "Verifier note: {verifier_note}\n"
    "Artifact preview:\n{artifact_preview}\n\n"
    "In 2-3 sentences, write a high-density 'what went wrong and why' that a "
    "future Planner can read as a forbidden constraint. Be specific. Avoid "
    "generic advice. Reference the concrete failure mode.\n\n"
    "Output ONLY a JSON object:\n"
    '  {{"summary": "<2-3 sentences>", "importance": <number 8-10>}}\n'
)


# Hard floor on reflective importance. Reflections are higher-signal than
# regular pass/fail episodes by construction; clamping to ≥8 makes R×I×R put
# them ahead of routine successes in next-task recall.
MIN_REFLECTION_IMPORTANCE = 8.0
MAX_IMPORTANCE = 10.0


class ReflectionResult:
    """Plain holder so callers don't have to import the LLMResponse type.

    Why not a Pydantic model: the fields are scalar and the holder is internal
    to the reflection pipeline. Skip the validation overhead.
    """

    __slots__ = ("summary", "importance")

    def __init__(self, summary: str, importance: float) -> None:
        self.summary = summary
        self.importance = importance

    def __repr__(self) -> str:  # pragma: no cover - dev ergonomics
        return f"ReflectionResult(importance={self.importance:.1f}, summary={self.summary!r})"


def _format_seed(
    *,
    contract: SprintContract,
    artifact: Artifact,
    judgement: VerifierJudgement,
) -> str:
    """Render the seed user-message for the reflection LLM.

    Truncating the artifact at 500 chars keeps the call cheap and avoids
    leaking the full chat-history-ish content of `artifact.content` (which is
    Generator-produced).
    """
    fc = judgement.failure_category.value if judgement.failure_category else "unknown"
    note = (judgement.reflection or "").strip() or "(no verifier note)"
    preview = (artifact.content or "").strip()[:500]
    return _REFLECTION_PROMPT.format(
        goal=contract.goal,
        failure_category=fc,
        verifier_note=note,
        artifact_preview=preview,
    )


async def reflect_on_failure(
    *,
    contract: SprintContract,
    artifact: Artifact,
    judgement: VerifierJudgement,
    model: str | None = None,
) -> ReflectionResult:
    """Run the reflection LLM call and return a `ReflectionResult`.

    Falls back to a deterministic synthesis on parse / network error: we'd
    rather write a degraded reflection than lose the failure signal entirely.

    Pre-condition: caller must already know `judgement.passed is False`. We
    don't re-check here — a successful run with `failure_category=None` would
    still produce a (less useful) reflection if forced through. The Executor
    is responsible for gating.
    """
    settings = get_settings()
    chosen_model = model or settings.verifier_model
    user = _format_seed(contract=contract, artifact=artifact, judgement=judgement)

    try:
        resp = await complete(
            model=chosen_model,
            system=_REFLECTION_PROMPT,
            user=user,
            role="reflection",
        )
    except Exception:
        # Network or parser blew up. Synthesise a minimal reflection from the
        # Verifier's own note so we still get a recall hit on the next plan.
        return _fallback_result(judgement)

    summary_raw = resp.parsed.get("summary")
    importance_raw = resp.parsed.get("importance")
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        return _fallback_result(judgement)
    try:
        importance = float(importance_raw)
    except (TypeError, ValueError):
        importance = MIN_REFLECTION_IMPORTANCE
    # Clamp to [MIN, MAX]: the LLM is told to emit 8-10 but we don't trust it
    # to stay in band.
    importance = max(MIN_REFLECTION_IMPORTANCE, min(MAX_IMPORTANCE, importance))
    return ReflectionResult(summary=summary_raw.strip(), importance=importance)


def _fallback_result(judgement: VerifierJudgement) -> ReflectionResult:
    """Build a ReflectionResult from the Verifier's seed text alone.

    The point is to never lose the failure signal: even a thin summary still
    gets injected as `forbidden` and helps the next Planner.
    """
    fc = judgement.failure_category.value if judgement.failure_category else "unknown"
    seed = (judgement.reflection or "").strip()
    base = f"[{fc}] {seed}" if seed else f"[{fc}] verifier rejected without details"
    return ReflectionResult(summary=base[:500], importance=MIN_REFLECTION_IMPORTANCE)
