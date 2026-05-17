# AgentLoom — Architecture

This document specifies the internal design. For a higher-level pitch, see [../README.md](../README.md).

---

## 1. Module Decomposition (AgentFlow-derived)

AgentLoom uses **four modules** that communicate via a single shared `MemoryHub`. This mirrors AgentFlow's planner / executor / verifier / generator decomposition (Yin et al., ICLR 2026 Oral) but adapts the role boundaries for general-purpose long-running tasks.

### 1.1 Planner
**Job**: Decompose the spec into discrete, self-contained tasks. Each task is a sprint contract.

**Inputs**:
- User goal (natural language)
- Memory recall (top-K episodes via R×I×R)
- Active Skill index (titles + 1-line summaries)
- Failure constraints (negative examples extracted from memory)

**Outputs**:
- `SprintContract { goal, non_goals, acceptance_criteria, target_files, forbidden, persona }`

**Key invariant**: The Planner is the **only** module that writes back to high-level memory. (Cognition's "writes single-threaded" applied at the memory layer.)

### 1.2 Generator
**Job**: Produce the artifact (code, text, plan, tool call sequence).

**Inputs**:
- The SprintContract (and nothing else from chat history)
- Tool sandbox (workspace-write, read-only filesystem, optional MCP servers)
- Active Skills loaded just-in-time based on the task type

**Outputs**:
- Structured artifact (diff, JSON, markdown)
- Trace events (one per tool call, one per LLM call)

**Cost discipline**: Generator runs on the cheapest viable model. Default: Claude Haiku 4.5 or GPT-5.3-codex.

### 1.3 Verifier (Holistic Evaluation)
**Job**: Grade the artifact along two axes simultaneously.

- **Bottom-up**: For each Trace event, classify as `ok | suboptimal | hallucinated | tool_error | missing_step`. Localize failures to specific spans. (Holistic Evaluation, arxiv 2605.14865.)
- **Top-down**: Score the artifact against the SprintContract rubric. Emit `{ pass: bool, score: float, failure_taxonomy: [...], rubric_breakdown: {...} }`.

**Clean Context invariant**: Verifier **never** receives the Generator's chat history. It receives only the SprintContract + the artifact + the trace events. (Cognition.)

**Cross-provider invariant**: For decisions with `score ≥ rubric.ship_threshold`, a second provider (the Judge) must independently re-score. Disagreement triggers a third opinion or human escalation.

### 1.4 Executor
**Job**: The actual run-loop orchestrator. Reads from MemoryHub, hands SprintContracts to Generators, routes Verifier feedback back to the Planner, decides retry vs. escalate vs. complete.

The Executor is the **only** stateful loop. Generator and Verifier are stateless invocations.

---

## 2. Memory Layer (MAGE-derived KG)

Memory is **not** a flat vector store. It is a **co-evolving knowledge graph** with four subgraphs (MAGE, arxiv 2605.10064):

### 2.1 The Four Subgraphs

| Subgraph | Node type | Edge type | Example node |
|---|---|---|---|
| **Experience** | Episode | `caused_by`, `resembles` | "Bugfix #41 — null check in date-math.ts" |
| **Task** | Task signature | `is_a`, `decomposes_to` | "fix failing test, type-error category" |
| **Skill** | Skill (SKILL.md) | `applicable_to`, `composed_of` | "skill:python-typeerror-pattern" |
| **Reasoning** | Reasoning trace | `derived_from` | "use git bisect when failure is post-rebase" |

### 2.2 Write Path

```
Generator finishes → trace events → Verifier scores
                                  ↓
                          Reflective Compaction (LLM call)
                                  ↓
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
     Experience node       Task signature       Skill candidate
     (always)              (always)             (only on success)
```

### 2.3 Read Path (R × I × R Scoring)

For each retrieval query, score each node:

```python
score(n) = recency(n) * importance(n) * relevance(n, query)
```

- **recency**: stepwise decay (24h: 1.0, 1w: 0.8, 1m: 0.5, 3m+: 0.1)
- **importance**: assigned at write time by an LLM (1-10), boosted on each subsequent reference
- **relevance**: cosine similarity over the 1536-dim embedding (OpenAI text-embedding-3-small)

Top-K is then traversed across the four subgraphs via 1-hop edges to pick up structurally adjacent context.

### 2.4 Promotion (Stanford-style)

- **Episode → Reasoning** when an episode is referenced ≥ 3 times. (Hippocampus → cortex consolidation analogy.)
- **Reasoning → Skill** when a Reasoning node is referenced across ≥ 3 distinct Task signatures.
- **Skill version bump** when the Verifier's score for the same Skill diverges by ≥ 1 std-dev between two windows.

### 2.5 Forgetting (TTL)

Background job runs nightly:
- Nodes with `recency * importance < 0.05` and no reference in 30 days → archived to cold storage.
- Hard deletion only happens for security-flagged content. Archive is searchable but excluded from default retrieval.

---

## 3. Skill Spec (Swarm Skills-compatible)

Each skill is a single Markdown file (Anthropic Skills format) with extended YAML frontmatter for Swarm Skills self-evolution (arxiv 2605.10052):

```markdown
---
name: python-typeerror-pattern
version: 3
description: Recognize and fix the most common Python TypeError patterns
applicable_when:
  - language: python
  - failure_type: typeerror
roles:
  - generator: applies the fix
  - verifier: confirms type narrows correctly
execution_bounds:
  max_llm_calls: 3
  max_tool_calls: 5
self_evolution:
  success_count: 17
  failure_count: 2
  last_patched: 2026-05-10
  multi_dim_score:
    correctness: 0.92
    cost: 0.78
    readability: 0.88
---

# Pattern: TypeError on optional field access

## When to use
...

## Procedure
1. ...
2. ...
```

### 3.1 Lifecycle

```
distill (Generator success + Verifier pass)
   ↓
draft Skill v1
   ↓
shadow-deploy (logged but not yet active)
   ↓
promote to active (after k successful shadow runs)
   ↓
patch (multi-dim score drift triggers re-distillation)
   ↓
deprecate (success rate falls below threshold)
```

### 3.2 Loading

Skills load **just-in-time** by name match on `applicable_when`. We never load the full library into context. This is the Anthropic Skills principle.

---

## 4. Trace & Telemetry

Every LLM call and tool call emits a `TraceEvent`:

```python
class TraceEvent(BaseModel):
    run_id: UUID
    parent_id: UUID | None
    module: Literal["planner", "generator", "verifier", "executor"]
    kind: Literal["llm_call", "tool_call", "memory_read", "memory_write", "skill_load"]
    started_at: datetime
    ended_at: datetime
    inputs: dict
    outputs: dict
    cost_usd: float
    tokens_in: int
    tokens_out: int
    provider: str
    model: str
    verifier_judgement: Literal["ok", "suboptimal", "hallucinated", "tool_error", "missing_step"] | None
```

Traces are stored in PostgreSQL (JSONB columns) and replayable via `agent_loom.replay`.

---

## 5. Failure Taxonomy

Failures are first-class citizens. Every failed run gets classified into the taxonomy, and the taxonomy itself is versioned (the Verifier learns new categories over time).

### v0.1 Taxonomy (initial)

| Category | Definition | Recovery |
|---|---|---|
| `spec_misread` | Generator solved the wrong problem | Refine SprintContract, re-issue |
| `partial_implementation` | Solution covers happy path but misses edge | Re-issue with edge constraint |
| `hallucinated_artifact` | References nonexistent function/file | Auto-rejected by Verifier with grep proof |
| `tool_error` | Tool returned error not handled | Add retry logic to skill |
| `context_rot` | Quality degraded mid-task | Context reset, sprint re-decomposed |
| `eval_drift` | Verifier was inconsistent across runs | Trigger Judge cross-check |
| `cost_budget_exceeded` | Hit per-task or per-session budget | Escalate to human |

---

## 6. Cost & Concurrency Model

### 6.1 Per-task budget
Each SprintContract carries a `max_cost_usd` field. Executor halts and escalates if exceeded.

### 6.2 Concurrency
- Multiple Generators may run in parallel **only** if their target files don't overlap.
- Overlapping target files → forced sequential, or git-worktree isolation if explicitly enabled.
- Verifier always runs in a separate process with isolated context.

### 6.3 Model routing (default)

| Module | Primary | Fallback | Judge (cross-provider) |
|---|---|---|---|
| Planner | Claude Opus 4.7 | GPT-5.5 thinking | — |
| Generator | Haiku 4.5 / GPT-5.3-codex | Sonnet 4.6 | — |
| Verifier | Sonnet 4.6 | GPT-5.3 | Claude Opus 4.7 *xor* GPT-5.5 thinking |

---

## 7. Storage

- **PostgreSQL 16 + pgvector** — episodic memory, traces, skill registry
- **In-memory `InMemoryEpisodicStore`** — Phase 0/1a default, used in tests and `MemoryHub.fake()`
- **Filesystem** — `skills/*.md` (so they're git-diffable), `runs/<run_id>/` (artifacts)

Schema migrations via Alembic.

### 7.1 Phase 1b — pgvector implementation (shipped)

The `PgvectorEpisodicStore` plugs into the same `EpisodicStore` Protocol as the
in-memory store, so `MemoryHub(store=...)` is the only swap. Cross-process
recall is proven by `tests/test_cross_process_recall.py`: two `python -m
agent_loom.examples.hello_harness --store pg` invocations share state through
Postgres, and the second run's Planner bumps `references_count` on the first
run's episode.

Implementation choices:

- **IVFFlat index** (`vector_cosine_ops`, `lists=100`) for ANN retrieval.
  Query path sets `ivfflat.probes=100` per transaction — small datasets need
  every list scanned; benchmark and unit tests confirm correctness, and 1k
  episodes still recalls in <200ms mean.
- **Two-stage rank**: ANN narrows to `top_k * 3` candidates, then `rir_score`
  in Python reorders by recency × importance × relevance. This keeps the
  scoring formula source-of-truth in one place (`memory/store.py::rir_score`).
- **One round-trip for the references-count update** via `UPDATE ... WHERE
  episode_id = ANY($1::uuid[])` — `top_k` rows in a single call.
- **ON CONFLICT UPDATE** writes so tests and reflection passes can re-write
  the same episode id without try/except gymnastics.

Phase 1b is the cut-off for memory plumbing changes. The KG layer (§2 four
subgraphs) lands in Phase 2 with a `graph_nodes` + `graph_edges` migration on
top of the same database.

---

## 8. API Surface (Phase 3+)

Optional FastAPI server exposes:

- `POST /runs` — submit a new goal
- `GET /runs/{run_id}` — fetch run status + artifacts
- `GET /runs/{run_id}/trace` — full TraceEvent stream
- `GET /memory/graph` — current KG snapshot (for dashboard)
- `GET /skills` — Skill registry
- `WS /runs/{run_id}/stream` — live trace events

---

## 9. What We Explicitly Do NOT Do (YAGNI)

To stay focused and avoid the "agent sprawl" failure mode (Varick, 2026):

- **No agent-to-agent free-form negotiation.** Manager-children only, map-reduce style.
- **No bespoke per-task fine-tuning.** Externalize via memory + skills, not weights.
- **No multi-region / multi-tenant** in v0.x. Single-user local + single-server hosted.
- **No proprietary protocol** — Swarm Skills format is the wire format. MCP for tools.

---

## 10. Open Design Questions (tracked in GitHub Issues at launch)

1. Should the KG use Neo4j or just PostgreSQL with explicit edge tables?
2. Should Skill self-evolution use a bandit (multi-dim scoring) or RL (Flow-GRPO from AgentFlow)?
3. How aggressively should we compact episodic memory? (Trade-off: recall fidelity vs. cost.)
4. Should the Judge always be cross-provider, or only above a stakes threshold?
