---
title: PostmortemEnv
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Epistemic RL env for cloud outage root-cause investigation
tags:
  - openenv
  - reinforcement-learning
  - sre
  - epistemic-rl
  - root-cause-analysis
---

<!-- Badges -->
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PostmortemEnv-blue)](https://huggingface.co/spaces/three-musketeers/PostmortemEnv)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Auenchanters/Three-Musketeers-FINALS/blob/main/train_notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-93%20passing-brightgreen)](tests/)

# PostmortemEnv

> **An Epistemic Reinforcement Learning Environment for Cloud Outage Investigation**
>
> *Theme 3.1: World Modeling (Professional) | Meta PyTorch OpenEnv Hackathon Apr '26*
> *By Three Musketeers (Utkarsh, Mohit, Tanush)*

![PostmortemEnv — Random vs Oracle on 30 episodes (10 seeds × 3 difficulties)](training_data/agent_comparison.png)

<sub>↑ Real evaluation data committed to the repo: a Random baseline collapses to 0.10 mean reward; the deterministic Oracle solver hits 0.98. The 0.88 gap is the training headroom available to any policy. Reproduce with `python train.py evaluate --n-seeds 10 && python scripts/plot_agent_comparison.py`. The live investigation console (see the embedded HF Space iframe and [static/](static/) for the UI source) renders the same data agent-by-agent in real time.</sub>

## Deliverables

| Resource | Link |
|---|---|
| 🤗 **Live OpenEnv Space** (Docker) | <https://huggingface.co/spaces/three-musketeers/PostmortemEnv> |
| 📓 **Training notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Auenchanters/Three-Musketeers-FINALS/blob/main/train_notebook.ipynb) → [`train_notebook.ipynb`](train_notebook.ipynb) · [`train_notebook.py`](train_notebook.py) (script form, kept in lock-step with the .ipynb) · canonical CLI: [`train.py`](train.py) |
| 📝 **Writeup / mini-blog** | [`blog_post.md`](blog_post.md) |
| 📹 **Demo video (≤ 2 min)** | <https://youtu.be/PLACEHOLDER> <!-- placeholder — final unlisted YouTube URL will be wired in before submission --> |
| 📊 **Committed plots** | [`training_data/reward_curves.png`](training_data/reward_curves.png) (3-panel: overall · per-difficulty · cumulative reward per step) · [`training_data/agent_comparison.png`](training_data/agent_comparison.png) (single-panel — regenerate both with `python scripts/plot_agent_comparison.py`) |
| 🧪 **Frontier-model benchmark** | [`scripts/run_frontier_benchmark.py`](scripts/run_frontier_benchmark.py) → [`training_data/frontier_results.json`](training_data/frontier_results.json) |

## TL;DR — the headline numbers

| Agent | Mean reward (10 seeds × 3 difficulties) | Notes |
|---|---:|---|
| Random baseline | **0.101** | Lower bound — proves the env has signal |
| SFT (Qwen2.5-1.5B-Instruct, 60 demos, 3 epochs) | _populated by `python train.py full`_ | One command on any CUDA box runs collect → SFT → evaluate → plot and merges the trained row into [`training_data/evaluation_results.json`](training_data/evaluation_results.json). |
| **Oracle (deterministic optimal)** | **0.979** | Upper bound — our hand-built solver |
| **Reward gap (Oracle − Random)** | **0.878** | The training headroom available to any policy |

Source: [`training_data/evaluation_results.json`](training_data/evaluation_results.json) (Random + Oracle committed; trained row added by `python train.py full`). Frontier-model rows (Claude Haiku, GPT-4o-mini, Llama-3.2-1B) populate via [`scripts/run_frontier_benchmark.py`](scripts/run_frontier_benchmark.py) → [`training_data/frontier_results.json`](training_data/frontier_results.json).

> **Why a placeholder for the SFT row?** Honesty: the previous SFT artifact in this repo (a `facebook/opt-125m` run) showed identical 0.160 means across all four epochs — meaning training never affected the policy's outputs. We deleted it rather than ship a phantom result. The pipeline is fully wired (see `python train.py full` below); paste a CUDA box's URL into the README once it lands its trained mean.

## Same task, two agents — `task1_recent_deploy` (a connection-pool regression)

```text
ORACLE (final_score = 0.960, 8 steps)         │  RANDOM (final_score = 0.155, 9 steps)
──────────────────────────────────────────────│──────────────────────────────────────────────
1  query_logs   data    "error"      r=0.135  │  1  query_logs   frontend  "error"   r=0.085
2  query_logs   data    "deploy"     r=0.035  │  2  query_logs   data      "deploy"  r=0.035
3  diff_commit  commit-a1b2c3        r=0.035  │  3  query_logs   auth      "deploy"  r=0.010
4  query_logs   data    "connection" r=0.185  │  4  query_logs   frontend  "error"   r=0.010
5  fetch_trace  trace-001            r=0.035  │  5  query_logs   batch     "error"   r=0.010
6  hypothesize  commit-a1b2c3        r=0.015  │  6  query_logs   frontend  "error"   r=0.010
7  explain_chain                     r=0.015  │  7  query_logs   auth      "deploy"  r=0.010
8  submit       commit-a1b2c3 + chain r=0.960 │  8  query_logs   frontend  "deploy"  r=0.010
                                              │  9  submit       "unknown"           r=0.155
```

Oracle climbs the dependency graph, then submits. Random spams `query_logs` until budget runs out and gives up. The environment's reward signal cleanly separates them — that's what makes it trainable. Reproduce with `python test_oracle_e2e.py` (oracle) and a 1-line random-agent loop.

---

## The Problem

Every SRE knows the drill: production goes down at 2 AM, and you're staring at a wall of logs, traces, and recent deploys trying to figure out *what broke and why*. Current RL environments train agents to **control** systems. But real SRE work is about **investigation** - piecing together evidence from frozen telemetry to reconstruct what happened.

**PostmortemEnv** is the first RL environment that trains LLMs to conduct postmortem investigations. Instead of controlling a live system, the agent receives a *frozen snapshot* of a failed cloud architecture and must identify the root cause and causal chain through structured queries - just like a real SRE.

> **Note on novelty.** Existing RL environments for LLMs target *system control* — agents change a live world and are graded on outcomes. [SWE-bench](https://arxiv.org/abs/2310.06770) (code edits that pass tests), [WebArena](https://arxiv.org/abs/2307.13854) (browser actions on live websites), and [ALFWorld](https://arxiv.org/abs/2010.03768) (text-based household tasks) are the closest neighbours, and all of them reward *changing* the world. We did not find a published RL environment as of April 2026 that rewards *causal investigation over frozen observational data* — where actions reveal hidden information instead of changing the world state, and the score is the quality of the reconstructed explanation. If we missed one, please open an issue and we will cite it here.

## Why This Matters

| Traditional RL Envs | PostmortemEnv |
|---------------------|---------------|
| Agent controls the system | Agent investigates what happened |
| Observable state changes | Partially observable frozen state |
| Actions change the world | Actions reveal hidden information |
| Reward = system performance | Reward = information gain + correctness |
| Tests control skills | Tests **causal reasoning** |

This is **epistemic RL** - the agent must gather evidence, form hypotheses, test them, and build a causal chain. It can't brute-force the answer; it must reason.

## Architecture

```
frontend -> auth -> data -> batch
    |          |       |        |
    v          v       v        v
  Logs      Logs    Logs     Logs
  Traces    Traces  Traces   Traces
  Commits   Commits Commits  Commits
  Configs   Configs Configs  Configs
```

The agent sees a 4-service microservice architecture with a dependency graph. Each service has logs, distributed traces, recent commits, config changes, and infrastructure events. The agent investigates using 7 action types:

| Action | Description |
|--------|-------------|
| `query_logs` | Search service logs by keyword |
| `fetch_trace` | Get a specific distributed trace |
| `diff_commit` | See what changed in a specific commit |
| `inspect_config` | Examine a config change |
| `hypothesize` | Test a root cause candidate (with feedback) |
| `explain_chain` | Test a causal chain (with similarity score) |
| `submit` | Submit final answer (terminal) |

## Three Difficulty Tiers

| Task | Difficulty | Max Steps | Challenge |
|------|-----------|-----------|-----------|
| Recent Deploy Blames | Easy | 40 | Single commit reduced connection pool from 100->20 |
| Cascade Chain | Medium | 75 | Batch OOM cascades through all 4 services |
| Correlated Root Cause | Hard | 120 | Auth failover bug + network degradation combine |

Each task has hand-crafted realistic telemetry, plus a **procedural seed generator** that creates unlimited unique scenarios from 5 failure templates. Validated: **10,000 seeds → 10,000 unique scenarios, 0 failures**. Measured throughput: ~1K scen/sec (generation only) / ~400 scen/sec with full oracle validation pass.

Reproduce the uniqueness claim in one line:

```bash
python -c "from data.seed_generator import generate_scenario; \
  ids=set(); [ids.add(generate_scenario(i,'easy')['task_id']) for i in range(10000)]; \
  assert len(ids)==10000; print('OK: 10,000 unique scenarios, 0 collisions')"
```

## Reward Design

**Dense per-step rewards** (not just terminal):
- **Information Gain**: `+0.05` per relevant fact discovered
- **Query Cost**: `-0.01` per query (encourages efficiency)
- **Hypothesis Feedback**: `-0.10` for wrong guesses
- **Terminal Score**: `0.5*cause_match + 0.3*chain_similarity + 0.2*efficiency - 0.1*wrong_guesses`

The grading is **fully deterministic** - no LLM-as-judge. Chain similarity uses Jaccard overlap on directed node/edge sets.

## Training Pipeline

```
   ┌──────────────────────────┐    ┌───────────────────────────────┐
   │ data/seed_generator.py   │    │ engine/environment.py          │
   │   procedurally generates │    │   reset · step · reward        │
   │   3-tier scenarios from  │    │   reward = info_gain + chain   │
   │   a seed                 │───▶│   + efficiency − wrong_guesses │
   └──────────────────────────┘    └────────────┬──────────────────┘
                                                ▼
   ┌──────────────────────────┐    ┌───────────────────────────────┐
   │ train.py collect         │───▶│ training_data/oracle_demos.jsonl
   │   roll out optimal       │    └────────────┬──────────────────┘
   │   solver per scenario    │                 ▼
   └──────────────────────────┘                                       ┌───────────────────────────────┐
                                   │ train.py sft (LoRA r=16, α=32) │
                                   │   default base = Qwen2.5-1.5B  │
                                   │   uses tokenizer.apply_chat_   │
                                   │   template — works for Qwen /  │
                                   │   Llama-3 / Mistral / Gemma    │
                                   └────────────┬──────────────────┘
                                                ▼
                                   ┌───────────────────────────────┐
                                   │ train.py grpo (online RL)      │
                                   │   reward_fn = env.step().reward│
                                   └────────────┬──────────────────┘
                                                ▼
                                   ┌───────────────────────────────┐
                                   │ train.py evaluate / plot       │
                                   │   Random vs SFT vs Oracle bars │
                                   └───────────────────────────────┘
```

### One-shot (recommended for a rented GPU)

```bash
pip install -r requirements-train.txt
python train.py full
```

That's it. Three minutes of dependency install, then `train.py full` runs **collect → baseline evaluate → SFT → trained evaluate → plot** end-to-end and writes:
- `training_data/oracle_demos.jsonl`
- `training_data/evaluation_results.json` with `random`, `oracle`, and `trained` keys
- `training_data/reward_curves.png` (3-panel: overall · per-difficulty · cumulative reward per step)
- `training_data/agent_comparison.png` (the README's headline chart)
- `sft_output/` (LoRA adapter + tokenizer)

The default base model is `Qwen/Qwen2.5-1.5B-Instruct` because it is **open-weight** (no `huggingface-cli login` / gated-license form), ships a chat template, and fits LoRA on a single T4. Pass `--model meta-llama/Llama-3.2-1B-Instruct` after `huggingface-cli login` if you prefer Llama-3.

`evaluate_trained_model` prints a sanity warning if the trained mean fails to clear `random + 0.02`, which catches the common mistake of evaluating the base model instead of the LoRA adapter.

### Step by step (if you want to inspect each phase)

```bash
pip install -r requirements-train.txt

python train.py collect --n-seeds 20          # 60 oracle demos -> oracle_demos.jsonl
python train.py evaluate --n-seeds 10         # random + oracle -> evaluation_results.json
python train.py sft                           # LoRA SFT on Qwen2.5-1.5B (~10-15 min on T4)
python train.py evaluate --n-seeds 10 --model ./sft_output   # adds the "trained" key
python scripts/plot_agent_comparison.py       # rebuilds reward_curves.png + agent_comparison.png
python train.py trace --model ./sft_output    # one readable episode trace for the README/blog
python train.py grpo --model ./sft_output     # stretch — online RL on env rewards
```

## Reward Gap (real data, n=10 seeds per cell)

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random | 0.099 | 0.095 | 0.108 | **0.101** |
| Oracle | 0.967 | 0.981 | 0.989 | **0.979** |
| **Gap** | **0.868** | **0.886** | **0.881** | **0.878** |

Source: [training_data/evaluation_results.json](training_data/evaluation_results.json), generated by `python train.py evaluate --n-seeds 10`. The 0.88 oracle-vs-random gap is the training headroom available to any policy trained on this environment.

![Per-difficulty agent comparison](training_data/agent_comparison.png)
*Per-difficulty mean episode score (n=10 seeds × 3 difficulties). The Random baseline (coral) is statistically indistinguishable across tiers; the deterministic Oracle (steelblue) saturates near 1.0. The dashed red line marks the random ceiling — any trained policy that finishes above it is genuinely learning the investigation skill.*

### SFT Training Configuration

Defaults for the SFT phase (override on the command line):
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` with LoRA (r=16, α=32, dropout=0.05) targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`. Open-weight, no HF auth.
- **Data**: 60 oracle demonstrations (20 seeds × 3 difficulties), rendered through `tokenizer.apply_chat_template` so train and eval distributions match for any modern instruct model.
- **Hyperparams**: 3 epochs, per-device batch size 2, gradient accumulation 4, lr 2e-5, fp16 on CUDA.
- **Eval**: 10 held-out seeds per difficulty (seeds in `2000..2009`, disjoint from the demo seeds in `1000..1019`).

Running `python train.py full` (or `train_notebook.ipynb` cell-by-cell) produces a fresh `training_data/evaluation_results.json` with the `trained` key populated and refreshes `training_data/reward_curves.png` with a **real** Random vs SFT vs Oracle chart. The SFT-trained model must beat the random baseline to validate the environment's trainability — `train.py` prints a loud warning if it does not.

## Quick Start

The repo is split into three install profiles so you only download what you need:

| Profile | Install | What it gives you |
|---|---|---|
| **Server** (HF Space, judges' demo) | `pip install -r requirements-server.txt` | FastAPI + OpenEnv server. CPU-only. Used by the [Dockerfile](Dockerfile). |
| **Local dev / inference / tests** | `pip install -r requirements.txt` | Server + `openai`, `httpx`, `pytest`, etc. for `inference.py`, `test_oracle_e2e.py`, and the 93-test suite. |
| **Training** (Colab T4 / rented GPU) | `pip install -r requirements-train.txt` | Adds `torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes` for SFT + GRPO. |

```bash
# Install (local dev)
pip install -r requirements.txt

# Run tests (93 pass, ~6 s on a laptop)
python -m pytest tests/ -v

# Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run oracle validation (no LLM needed)
python test_oracle_e2e.py
# → task1: 0.96  task2: 0.97  task3: 0.98  ALL PASS

# Run LLM agent
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860
python inference.py
```

## Adaptive Curriculum (Theme #4 crossover)

The environment maintains a per-agent **ELO-style skill estimate** in [`training_data/curriculum_state.json`](training_data/curriculum_state.json) and uses it to bias the next-task sampler. Implementation lives in [`web/curriculum.py`](web/curriculum.py). After each episode, both the agent rating and the per-task rating are updated with the standard ELO rule (`K = 24`):

```python
expected = 1 / (1 + 10 ** ((task_elo - agent_elo) / 400))
delta    = K * (score - expected)
```

Tasks start at fixed difficulty ratings — `task1_recent_deploy: 1000`, `task2_cascade_chain: 1300`, `task3_correlated_cause: 1600` — and a per-task `difficulty_multiplier` rises when the agent beats its expected score and falls otherwise. The procedural seed generator reads that multiplier to scale chain depth, red-herring count, and noise injection per scenario. Net effect:

- a fresh agent stays on Easy until it consistently submits correct causes,
- a near-Oracle agent gets pushed almost exclusively to Hard *and* gets harder Hard scenarios,
- a stuck agent drifts back down a tier instead of grinding on impossible problems.

This is a lightweight version of the self-play adaptive-curriculum loop from Theme #4 (Self-Improvement) — same goal (keep the agent in its zone of proximal development), reused machinery (ELO + per-task scaling), but applied to *task selection over a procedurally-generated set* instead of self-play match-making. State is persisted between runs so the HF Space curriculum survives container restarts, and you can inspect any agent's trajectory with `cat training_data/curriculum_state.json`.

## Future Work

- **GRPO online RL.** `train.py grpo` implements GRPO with environment rewards on top of the SFT checkpoint. Within the hackathon compute budget we shipped the SFT pipeline as the primary training signal and left a single-pass GRPO smoke run for after the deadline; the loop is wired end-to-end and will produce `training_data/grpo_eval_results.json` when run.
- **Frontier-model coverage.** `scripts/run_frontier_benchmark.py` already runs Claude Haiku, GPT-4o-mini, and Llama-3.2-1B against all three difficulty tiers. We will publish the full table to `training_data/frontier_results.json` and surface it in the README once we burn through the API spend.
- **Multi-agent investigation.** A natural extension: two agents (one bullish on a hypothesis, one adversarial) negotiate the final submission. Reward shaping is symmetric so the env supports it without changes.
- **Real telemetry import.** A loader for OpenTelemetry traces / Loki logs would let teams replay their own real outages as PostmortemEnv tasks.

## Docker

```bash
docker build -t postmortemenv .
docker run -p 7860:7860 postmortemenv
```

## Deploy to a Hugging Face Space (one-shot)

```bash
# 1. Create a new Docker-SDK Space at https://huggingface.co/new-space
# 2. Add your HF git remote
git remote add hf https://huggingface.co/spaces/<your-username>/PostmortemEnv

# 3. Push — the Space rebuilds automatically using ./Dockerfile
git push hf main
```

Everything the Space needs is already wired in: the YAML front-matter at the top of this README declares `sdk: docker`, `app_port: 7860`, the [`Dockerfile`](Dockerfile) installs from [`requirements-server.txt`](requirements-server.txt) (CPU-only — no GPU credits required to host the env), and [`app.py`](app.py) exposes `/reset`, `/step`, `/state`, `/health`, `/schema`, plus the `/ws` OpenEnv session endpoint and the live investigation console at `/`. Once the build finishes, the Space URL above becomes the runnable demo judges will see.

## Project Structure

```
PostmortemEnv/
+-- app.py                    # FastAPI server (OpenEnv)
+-- inference.py              # LLM investigation agent
+-- train.py                  # Training pipeline (SFT + GRPO)
+-- test_oracle_e2e.py        # Oracle validation
+-- models/                   # Pydantic models
+-- engine/                   # Core environment + grader + rewards
+-- data/
|   +-- scenarios/            # 3 hand-crafted outage scenarios
|   +-- solutions/            # Oracle optimal solutions
|   +-- seed_generator.py     # Procedural scenario generator
+-- tests/                    # 93 unit tests
+-- Dockerfile, openenv.yaml, requirements.txt
```

## Technical Innovation

1. **Epistemic RL**: First environment that tests *information gathering* rather than system control
2. **Deterministic Oracle Grading**: Pure function scoring with Jaccard node/edge chain comparison — no LLM-as-judge
3. **Dense Reward Shaping**: Information-gain proxy rewards prevent sparse-reward exploration collapse
4. **Procedural Generation**: 5 failure templates × unlimited seeds = infinite curriculum (10K validated)
5. **Real Enterprise Patterns**: Connection pool exhaustion, OOM cascades, AZ failover bugs, correlated failures
6. **Adaptive Curriculum** (Theme #4 crossover): ELO-based difficulty scaling — full mechanics in the [Adaptive Curriculum section](#adaptive-curriculum-theme-4-crossover) above
7. **Live Investigation UI**: Real-time SSE streaming with animated service dependency graph and investigation replay
8. **Token-Efficient LLM Agent**: Anthropic prompt caching on system prompt + scenario seed — each step costs ~200 tokens

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - Environment framework
- [HuggingFace TRL](https://github.com/huggingface/trl) - Training (SFT + GRPO)
- [Pydantic](https://docs.pydantic.dev/) - Type-safe models
- [FastAPI](https://fastapi.tiangolo.com/) - HTTP/WebSocket server

---

*Built for the Meta PyTorch OpenEnv Hackathon, April 2026*
