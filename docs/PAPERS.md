# Papers Implemented

This document tracks the specific claim from each paper we adopt and the concrete file in AgentLoom that realizes it. Use this as the academic provenance trail.

---

## Core Five

### 1. AgentFlow: In-the-Flow Agentic System Optimization
- **arXiv**: [2510.05592](https://arxiv.org/abs/2510.05592)
- **Venue**: ICLR 2026 Oral (Top 1.1%)
- **Affiliation**: Stanford University
- **Key claim adopted**: Four-module decomposition (Planner / Executor / Verifier / Generator) coordinating through an evolving memory yields a 7B model that beats GPT-4o on reasoning tasks (+14.9% on search, +14.5% on math).
- **What we take**:
  - The four-module split is the structural backbone (see `src/agent_loom/core/`)
  - Memory is shared *across* modules, not per-module
  - Planner is the only writer to high-level memory
- **What we don't take (yet)**:
  - Flow-GRPO algorithm (deferred to v0.2 — too much engineering for v0.1)
  - Joint training of planner — we use frozen models throughout

### 2. MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs
- **arXiv**: [2605.10064](https://arxiv.org/abs/2605.10064)
- **Date**: May 14, 2026
- **Key claim adopted**: A four-subgraph knowledge graph (Experience / Task / Skill / Reasoning) that co-evolves from the same reward stream enables agent self-improvement *without* changing the backbone weights.
- **What we take**:
  - Four-subgraph structure (see `src/agent_loom/memory/graph.py`)
  - Co-evolution from a shared reward signal (the Verifier's score)
  - Frozen backbone — capability accumulates in the KG, not in weights
- **What we don't take (yet)**:
  - Bandit-based task and skill routing — deferred to v0.2
  - Teacher-written failure corrections — we use the Verifier itself as the teacher

### 3. Swarm Skills: A Portable Multi-Agent System Specification
- **arXiv**: [2605.10052](https://arxiv.org/abs/2605.10052)
- **Date**: May 15, 2026
- **Key claim adopted**: Extending the Anthropic Skills standard with multi-agent semantics (roles, workflows, execution bounds, self-evolution) makes workflows first-class distributable assets.
- **What we take**:
  - SKILL.md as the wire format (see `src/agent_loom/skills/spec.py`)
  - YAML frontmatter for roles, execution_bounds, self_evolution metadata
  - Multi-dimensional scoring (`correctness`, `cost`, `readability`) to patch skills
  - JIT loading — never load the whole library into context
- **What we don't take (yet)**:
  - Cross-organization skill marketplace (out of scope for portfolio)

### 4. Holistic Evaluation and Failure Diagnosis of AI Agents
- **arXiv**: [2605.14865](https://arxiv.org/abs/2605.14865)
- **Date**: May 2026
- **Key claim adopted**: Combining **top-down** agent-level metrics (planning quality, tool coverage) with **bottom-up** span-level diagnosis (per LLM call, per tool call) localizes failures to specific causes rather than just labeling the whole run as "bad."
- **What we take**:
  - Verifier emits per-span judgements + a top-down rubric score (see `src/agent_loom/core/verifier.py`)
  - The failure taxonomy itself is versioned and grows over time
  - Cross-provider Judge for high-stakes decisions
- **What we don't take (yet)**:
  - The paper's specific top-down metrics (planning quality, tool coverage) — we use simpler rubrics in v0.1

### 5. Externalization in LLM Agents: A Unified Review
- **arXiv**: [2604.08224](https://arxiv.org/abs/2604.08224)
- **Date**: April 2026
- **Key claim adopted**: Agent development is best understood as a stacked-layer trajectory — **Weights → Context → Harness** — with capabilities migrating outward over time (2022 → 2026).
- **What we take**:
  - This is the mental model in the README diagram
  - Every design decision is framed as "where in the stack does this live?"
- **What we don't take**:
  - This is a survey paper; no algorithmic claim to implement

---

## Background / Influences (cited in the README's "design principles")

### Anthropic Harness Design (2026-03 blog post)
- **Source**: Anthropic blog, Prithvi Rajasekaran
- **Key claim**: 3-agent harness (Planner / Generator / Evaluator) with context resets instead of compaction. $9 vs. $200 cost-quality reproduction.
- **Our reproduction target**: `benchmarks/bugfix_marathon.py` should reproduce the qualitative shape (Solo plateaus, Harness scales, Harness+Memory dominates after warmup).

### Cognition — "Multi-Agents: What's Actually Working" (2026-04)
- **Key claims adopted**:
  - **Writes single-threaded**: only the Planner writes to high-level memory
  - **Clean Context for Verifier**: Verifier never sees Generator's chat history
  - **Capability Router**: model selection per task, not "weakest → strongest" escalation

### Stanford Generative Agents (Park et al., 2023)
- **Key claim adopted**: Memory scored by Recency × Importance × Relevance.
- **Where**: `src/agent_loom/memory/store.py` — the `score()` function.

### Reflexion (Shinn et al., 2023)
- **Key claim adopted**: Verbalized self-reflection after failure becomes a memory artifact that improves the next attempt.
- **Where**: Phase 2's reflective compaction step.

### Voyager (Wang et al., 2023)
- **Key claim adopted**: An ever-growing skill library is a sustainable form of capability accumulation in frozen-model regimes.
- **Where**: The whole `src/agent_loom/skills/` module — but using Swarm Skills format rather than Voyager's raw code blocks.

### MemGPT (Packer et al., 2023)
- **Key claim adopted**: Cross-session memory makes long-horizon resumption possible.
- **Where**: Phase 3's "resume yesterday's project" demo.

---

## Watch list (papers we may pull in for v0.2+)

| Paper | Why we care |
|---|---|
| [The Last Harness You'll Ever Build (2604.21003)](https://arxiv.org/abs/2604.21003) | Meta-Evolution Loop — the harness evolves itself |
| [HARBOR: Automated Harness Optimization (2604.20938)](https://arxiv.org/abs/2604.20938) | Bayesian Optimization for prompt/rubric tuning |
| [A Survey on the Security of Long-Term Memory (2604.16548)](https://arxiv.org/abs/2604.16548) | Memory poisoning defenses for production |
| [Predictive Maps of Multi-Agent Reasoning (2605.11453)](https://arxiv.org/abs/2605.11453) | Successor-representation diagnosis of agent topologies |
| [ComplexMCP (2605.10787)](https://arxiv.org/abs/2605.10787) | Realistic MCP tool-sandbox benchmark |
| [SkillOS (2605.06614)](https://arxiv.org/abs/2605.06614) | Skill curation as a learned policy |
| [DeepVerifier (2601.15808)](https://arxiv.org/abs/2601.15808) | Test-time rubric scaling for the Verifier |

---

## How to cite AgentLoom

If you use this repository, please cite the papers above and link back. A `CITATION.cff` will be added at v0.1.0 release.
