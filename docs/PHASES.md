# AgentLoom — Phase Roadmap

Four phases × roughly one week each. Each phase has a clear **Definition of Done** and ships a tangible artifact.

The user's available coding time is ~3 hours per day, so each phase budgets ~18–21 hours.

---

## Phase 0 — Foundation (Week 1, ~20h)

**Goal**: Smallest possible end-to-end run. No memory. No skills. Just Planner → Generator → Verifier.

### Tasks
- [ ] `pyproject.toml` with anthropic, openai, pydantic, pytest, structlog
- [ ] `core/types.py` — `SprintContract`, `TraceEvent`, `RunStatus` Pydantic models
- [ ] `core/planner.py` — single LLM call that produces a `SprintContract`
- [ ] `core/generator.py` — receives only the SprintContract; never sees prior chat
- [ ] `core/verifier.py` — Clean Context invariant enforced via dataclass-level access control
- [ ] `core/executor.py` — sequential loop, max 3 iterations per task
- [ ] `examples/hello_harness.py` — solves "add a function that returns Fibonacci(n)"
- [ ] First trace persistence (JSONL files under `runs/`)

### Definition of Done
- `python -m agent_loom.examples.hello_harness` produces a working `fib.py`
- The trace JSONL contains ≥ 4 events (1 planner call, 1 generator call, 1 verifier call, 1 final)
- Cost per run < $0.05 with default model routing
- Pytest passes with ≥ 80% coverage on `core/`

### Paper anchor
*AgentFlow* (4-module decomposition) + *Anthropic Harness Design* (sprint contract idea).

---

## Phase 1 — Memory MVP (Week 2, ~20h)

**Goal**: Memory exists, gets written, gets read. No KG structure yet — just flat episodic store with R×I×R retrieval.

### Tasks
- [ ] Postgres + pgvector setup via docker-compose (`scripts/dev_db_up.sh`)
- [ ] Alembic migration for `episodes` table (id, content, embedding, importance, recency_score, refs)
- [ ] `memory/store.py` — `write_episode()` and `recall(query, top_k)` with R × I × R scoring
- [ ] LLM-assigned importance at write time (single call, 1-10 scale)
- [ ] Embedding via OpenAI text-embedding-3-small (or local sentence-transformers as fallback)
- [ ] Wire Planner to call `recall()` before producing a SprintContract
- [ ] Wire Verifier to call `write_episode()` after every judgement

### Definition of Done
- A second run with a similar prompt retrieves the first run's episode at least sometimes
- `recall()` returns within < 200ms for stores up to 1k episodes
- Each episode includes `references_count` that increments on retrieval
- Pytest covers the R, I, R computations individually

### Paper anchor
*Stanford Generative Agents* (R × I × R) + early bits of *MAGE* (memory as first-class).

---

## Phase 2 — Reflexion Loop + KG (Week 3, ~20h)

**Goal**: Failure → reflection → memory growth → next-task improvement is observable.

### Tasks
- [ ] **Reflective Compaction**: after each Verifier rejection, an LLM produces a 1-paragraph "what went wrong and why" → written as an Episode with high importance
- [ ] **Failure constraint injection**: Planner reads top-K negative episodes for the current task signature and adds them to the SprintContract's `forbidden` field
- [ ] **KG edges**: add `edges` table with `(src_id, dst_id, edge_type)`; populate at write time via cheap rules (same task signature, same module, temporal succession)
- [ ] **Episode → Reasoning promotion** rule (≥ 3 references)
- [ ] **Failure taxonomy** v0.1: Verifier emits one of 7 categories per rejection
- [ ] **Repeat-failure-rate** metric: % of failures whose taxonomy category matches a previous failure in same task signature

### Definition of Done
- Run the same buggy task twice in a row; the second attempt has the failure constraint in its SprintContract
- Repeat-failure-rate drops measurably on the second batch of similar tasks
- KG has at least 50 nodes and 100 edges after running the example suite
- `agent_loom.memory.inspect` CLI shows the graph summary

### Paper anchor
*MAGE* (co-evolving KG) + *Reflexion* (Shinn 2023) + *Holistic Eval* (failure taxonomy).

---

## Phase 3 — Benchmark & Showcase (Week 4, ~20h) — **🎯 First Public Release**

**Goal**: Three-condition benchmark runs end-to-end, dashboards render, README claims are reproducible.

### Tasks
- [ ] **`benchmarks/bugfix_marathon.py`** — selects N tasks from a curated mini-set (SWE-bench Lite subset or our own seeded tasks). Runs Solo / Harness / Harness+Memory in sequence, logs cost, success, time
- [ ] **`dashboards/`** — simple FastAPI + vanilla JS or htmx that shows: live run, cost curves (Solo vs. Harness vs. Memory), KG snapshot, recent failures by taxonomy
- [ ] **`scripts/replay.py`** — replay any past run from its trace JSONL
- [ ] **`examples/`** — at least 3 demo scripts (Fib hello, "two-day research", "skill growth")
- [ ] **First Swarm Skill auto-distilled** by Phase 2's reflection loop. Verify it loads JIT in Phase 3 runs
- [ ] **README badges** updated with real status, real cost figures, real run counts
- [ ] **Demo video** (≤ 5 min) recorded and linked
- [ ] **First public release tag** `v0.1.0` on GitHub

### Definition of Done
- `python -m agent_loom.benchmarks.bugfix_marathon --tasks 20 --conditions solo,harness,memory` produces a comparative chart PNG and a CSV
- The chart visibly shows divergence in cost-per-task by problem 10
- Dashboard runs on `localhost:8000` and updates live
- README has at least one screenshot from the dashboard and one chart
- Repo is **public** and pushed to GitHub

### Paper anchor
*Anthropic Harness Design* ($9 vs. $200 reproduction) + *Holistic Evaluation* (top-down + bottom-up scoring).

---

## Beyond v0.1 (Future)

### v0.2 — Self-Evolving Harness
- *The Last Harness You'll Ever Build* (Sylph.AI, 2604.21003): Meta-Evolution Loop optimizes the harness blueprint itself
- *HARBOR* (2604.20938): Bayesian Optimization of prompts and rubrics

### v0.3 — Tool Ecosystem
- *ComplexMCP* (2605.10787) benchmark integration
- First-class MCP server registry

### v0.4 — Production Readiness
- Multi-user runs, RBAC for memory access
- *Mnemonic Sovereignty* (2604.16548): memory poisoning defenses
- Trace replay UI parity with wandb-style tools

### v0.5 — Multi-Agent Topology
- *Predictive Maps of Multi-Agent Reasoning* (2605.11453): successor-representation analysis of agent communication graphs
- Configurable topology presets (chain, star, hierarchical)

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase scope creep (especially Phase 2 KG) | High | Keep edges to simple rules in v0.1; full bandit-based MAGE deferred to v0.2 |
| API cost overruns during benchmarking | Medium | Per-task budgets enforced in `core/executor.py` from Phase 0 |
| SWE-bench Lite tasks too hard for Haiku-class Generator | Medium | Start with a hand-curated subset of "easy" issues; expand later |
| Dashboard frontend distracts from research substance | Medium | Phase 3 dashboard is **vanilla htmx** — no React, no build step |
| Trying to implement all 5 papers exhaustively | High | Each paper has a specific "role"; we cite + adopt the key idea, not the whole paper |
