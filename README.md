# AgentLoom

> A memory-augmented multi-agent harness that **weaves together** the latest research on long-running LLM agents.
> Inspired by **AgentFlow** (ICLR 2026 Oral), **MAGE** (May 2026), **Swarm Skills** (May 2026), **Holistic Evaluation** (May 2026), and **Externalization** (Apr 2026).

[![Status](https://img.shields.io/badge/status-WIP-yellow)]() [![License](https://img.shields.io/badge/license-MIT-blue)]() [![Papers](https://img.shields.io/badge/papers-5-purple)]()

---

## TL;DR

Long-running agents fail for three reasons: **context rot**, **repeated mistakes**, and **drift in evaluation**.

AgentLoom is a reference implementation that fixes all three by combining:

- A **4-module harness** (Planner / Executor / Verifier / Generator) — *AgentFlow, ICLR 2026 Oral*
- A **co-evolving knowledge graph memory** with R×I×R retrieval — *MAGE + Stanford Generative Agents*
- **Self-evolving Swarm Skills** that distill successful trajectories into reusable assets — *Swarm Skills, May 2026*
- **Top-down + bottom-up evaluation** with rubric-guided verification — *Holistic Evaluation, May 2026*

The result: **harness + memory + skills + eval as one coherent system**, not as siloed tricks.

---

## The Three Failure Modes We Fix

| Failure Mode | What it looks like | How AgentLoom fixes it |
|---|---|---|
| **Context Rot** | Quality degrades after 50+ turns; agent forgets earlier decisions | Episodic memory with R×I×R retrieval + planner-side context resets (AgentFlow style) |
| **Repeated Mistakes** | Agent makes the same error in 2nd, 3rd, 10th task | Reflective compaction promotes failures into the knowledge graph; future Planner reads them as constraints |
| **Eval Drift** | Verifier gets lenient over time; same bug judged "ok" then "not ok" | Holistic Evaluation: bottom-up span-level traces + top-down rubric, both cross-checked across providers |

---

## Architecture (3-Layer View)

Following the *Externalization* framework (Weights / Context / Harness):

```
              ┌──────────────────────────────────────────────┐
              │              HARNESS LAYER                   │
              │                                              │
              │   ┌────────┐   ┌──────────┐   ┌──────────┐  │
              │   │Planner │──▶│Generator │──▶│ Verifier │  │
              │   └────┬───┘   └────┬─────┘   └─────┬────┘  │
              │        │            │               │       │
              │        └────────────┴───────────────┘       │
              │                     │                       │
              └─────────────────────┼───────────────────────┘
                                    ▼
              ┌──────────────────────────────────────────────┐
              │             CONTEXT LAYER                    │
              │                                              │
              │   ┌──────────────────────────────────────┐   │
              │   │  Co-Evolving Knowledge Graph Memory  │   │
              │   │   ─ Experience subgraph              │   │
              │   │   ─ Task subgraph                    │   │
              │   │   ─ Skill subgraph (Swarm Skills)    │   │
              │   │   ─ Reasoning subgraph               │   │
              │   └──────────────────────────────────────┘   │
              │                     │                        │
              │   ┌──────────────────────────────────────┐   │
              │   │  Skill Library (SKILL.md format)     │   │
              │   │   ─ Auto-distilled from successes    │   │
              │   │   ─ Patched via multi-dim scoring    │   │
              │   └──────────────────────────────────────┘   │
              └──────────────────────────────────────────────┘
                                    ▼
              ┌──────────────────────────────────────────────┐
              │             WEIGHTS LAYER                    │
              │                                              │
              │   Frozen foundation models                   │
              │   (Claude Opus 4.7, GPT-5.3, Haiku 4.5)      │
              │   Cross-provider Judge for high-stakes calls │
              └──────────────────────────────────────────────┘
```

---

## Why This is Different

| Existing OSS | What they miss |
|---|---|
| LangGraph / CrewAI / AutoGen | Memory layer is bolt-on; no eval contract |
| MemGPT / Letta | Single-agent; no Verifier separation |
| Voyager / AutoSkill | No harness; skills don't connect to verification |
| Anthropic Cookbook | No reference implementation of the 2026 Harness pattern |

**AgentLoom is the first OSS reference implementation** that wires all four pieces — harness + memory + skills + eval — together with measurable trade-offs.

---

## Demo Scenarios (Roadmap)

### 1. Bug-Fix Marathon (the headliner)
Run 50 SWE-bench Lite problems sequentially with similar bugs sprinkled in. Compare three conditions: **Solo / Harness / Harness+Memory**.

> *Expected result: Solo plateaus around problem 15 (context rot), Harness holds steady, Harness+Memory gets cheaper per task as the skill library accumulates.*

### 2. Same Question, Two Days Apart
Day 1: research question. Day 2: follow-up. Show 4× token reduction via R×I×R retrieval of Day 1 episodic memory.

### 3. Skill Library Growth Visualization
Watch `skills/*.md` files appear and evolve over 100 tasks. Voyager-style, but in the Swarm Skills format compatible with Anthropic Skills standard.

### 4. Adversarial Verifier Catches Hallucination
Generator says "function X exists." Verifier (Clean Context, separate provider) runs `grep`, disproves it, sends back to Planner. Trace divergence visible in dashboard.

### 5. Resume Yesterday's Project
`agent-loom resume project_x` → "Yesterday you stopped at line 42 with 2 failing tests" → continues seamlessly.

---

## Papers Implemented

| # | Paper | arXiv | Role in AgentLoom |
|---|---|---|---|
| 1 | **AgentFlow: In-the-Flow Agentic System Optimization** (ICLR 2026 Oral) | [2510.05592](https://arxiv.org/abs/2510.05592) | 4-module harness baseline |
| 2 | **MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs** | [2605.10064](https://arxiv.org/abs/2605.10064) | KG-structured episodic memory |
| 3 | **Swarm Skills: A Portable Multi-Agent System Specification** | [2605.10052](https://arxiv.org/abs/2605.10052) | Skill format + self-evolution |
| 4 | **Holistic Evaluation and Failure Diagnosis of AI Agents** | [2605.14865](https://arxiv.org/abs/2605.14865) | Top-down + bottom-up Verifier |
| 5 | **Externalization in LLM Agents: A Unified Review** | [2604.08224](https://arxiv.org/abs/2604.08224) | 3-layer framework for the README |

See [docs/PAPERS.md](docs/PAPERS.md) for the full mapping between each paper's claims and our implementation choices.

---

## Quick Start

> **Note**: This is WIP. Phase 0 lands the minimal harness. Roadmap in [docs/PHASES.md](docs/PHASES.md).

```bash
# Clone and install
git clone https://github.com/<you>/agent-loom.git
cd agent-loom
uv sync   # or: pip install -e .

# Set keys
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY

# Run the smallest end-to-end demo
python -m agent_loom.examples.hello_harness

# Run the Bug-Fix Marathon (Phase 3+)
python -m agent_loom.benchmarks.bugfix_marathon --tasks 10 --conditions solo,harness,memory
```

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| **Phase 0** — Foundation | 🚧 In progress | Minimal Planner / Generator / Verifier with Clean Context |
| **Phase 1** — Memory MVP | 📋 Planned | R×I×R retrieval over pgvector |
| **Phase 2** — Reflexion Loop | 📋 Planned | Reflective compaction + KG (MAGE-style) |
| **Phase 3** — Benchmark & Showcase | 📋 Planned | 3-condition benchmark + dashboards + first public release |

Detailed roadmap → [docs/PHASES.md](docs/PHASES.md)
Architecture deep-dive → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
Papers mapping → [docs/PAPERS.md](docs/PAPERS.md)

---

## Design Principles

These are the rules AgentLoom follows. They are non-negotiable:

1. **Clean Context for Verifier** — Verifier never sees Generator's chat history. Only the diff/output. (*Cognition, "Multi-Agents: What's Actually Working"*)
2. **Writes single-threaded** — Parallel Generators run in git worktrees; merges go through the Planner alone.
3. **Cross-provider Judge for high-stakes calls** — Final ship/no-ship decisions require independent agreement between Claude and GPT.
4. **Externalize first** — When in doubt between training, prompting, and externalizing, externalize. Memory and Skills are first-class citizens.
5. **Measure everything** — Repeat-failure rate, memory hit rate, cost per solved task. If it isn't measured, it doesn't improve.

---

## Roadmap Beyond v0.1

- **v0.2** — Meta-Evolution Loop (Sylph.AI's *The Last Harness You'll Ever Build*) so the harness evolves itself
- **v0.3** — HARBOR-style Bayesian Optimization of prompts/rubrics
- **v0.4** — Production-grade traces + replay UI (think `wandb` for agents)
- **v0.5** — MCP-native tool sandbox (compatible with Anthropic Skills + Swarm Skills)

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If AgentLoom is useful for your research or product, please cite the underlying papers (see [docs/PAPERS.md](docs/PAPERS.md)) and link back to this repository.
