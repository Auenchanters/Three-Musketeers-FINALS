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

<!-- Badge -->
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PostmortemEnv-blue)](https://huggingface.co/spaces/three-musketeers/PostmortemEnv)

# PostmortemEnv

> **An Epistemic Reinforcement Learning Environment for Cloud Outage Investigation**
>
> *Theme 3.1: World Modeling (Professional) | Meta PyTorch OpenEnv Hackathon Apr '26*
> *By Three Musketeers (Utkarsh, Mohit, Tanush)*

| 📹 [Demo Video](#) | 📝 [HF Blog Post](#) | 🤗 [Live Space](https://huggingface.co/spaces/three-musketeers/PostmortemEnv) |
|---|---|---|

---

## The Problem

Every SRE knows the drill: production goes down at 2 AM, and you're staring at a wall of logs, traces, and recent deploys trying to figure out *what broke and why*. Current RL environments train agents to **control** systems. But real SRE work is about **investigation** - piecing together evidence from frozen telemetry to reconstruct what happened.

**PostmortemEnv** is the first RL environment that trains LLMs to conduct postmortem investigations. Instead of controlling a live system, the agent receives a *frozen snapshot* of a failed cloud architecture and must identify the root cause and causal chain through structured queries - just like a real SRE.

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

Each task has hand-crafted realistic telemetry, plus a **procedural seed generator** that creates unlimited unique scenarios from 5 failure templates. Validated: **10,000 seeds → 10,000 unique scenarios, 0 failures, 1,371 scenarios/sec**.

## Reward Design

**Dense per-step rewards** (not just terminal):
- **Information Gain**: `+0.05` per relevant fact discovered
- **Query Cost**: `-0.01` per query (encourages efficiency)
- **Hypothesis Feedback**: `-0.10` for wrong guesses
- **Terminal Score**: `0.5*cause_match + 0.3*chain_similarity + 0.2*efficiency - 0.1*wrong_guesses`

The grading is **fully deterministic** - no LLM-as-judge. Chain similarity uses Levenshtein edit distance.

## Training Pipeline

```bash
# 1. Collect oracle demonstrations
python train.py collect --n-seeds 20

# 2. Evaluate the reward gap (random vs oracle)
python train.py evaluate --n-seeds 10

# 3. Plot reward curves
python train.py plot

# 4. SFT on oracle demonstrations
python train.py sft --model meta-llama/Llama-3.2-1B-Instruct

# 5. GRPO with environment rewards
python train.py grpo --model ./sft_output
```

## Reward Gap (real data, n=10 seeds per cell)

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random | 0.099 | 0.094 | 0.108 | **0.101** |
| Oracle | 0.967 | 0.981 | 0.989 | **0.979** |
| **Gap** | **0.868** | **0.887** | **0.881** | **0.878** |

Source: [training_data/evaluation_results.json](training_data/evaluation_results.json), generated by `python train.py evaluate --n-seeds 10`. The 0.88 oracle-vs-random gap is the training headroom available to any policy trained on this environment.

### SFT Training (Colab T4)

The full SFT pipeline runs on a free Colab T4 GPU via [train_notebook.py](train_notebook.py):
- **Model**: `meta-llama/Llama-3.2-1B-Instruct` with LoRA (r=16, α=32)
- **Data**: 60 oracle demonstrations (20 seeds × 3 difficulties)
- **Epochs**: 3, batch size 2, lr 2e-5
- **Eval**: 10 held-out seeds per difficulty (separate seed namespace)

After training, `training_data/reward_curves.png` is overwritten with a **real** Random vs SFT vs Oracle chart from `training_data/sft_eval_results.json`. The SFT-trained model must beat the random baseline to validate the environment's trainability.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests (92 pass, <0.4s)
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

## Docker

```bash
docker build -t postmortemenv .
docker run -p 7860:7860 postmortemenv
```

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
+-- tests/                    # 92 unit tests
+-- Dockerfile, openenv.yaml, requirements.txt
```

## Technical Innovation

1. **Epistemic RL**: First environment that tests *information gathering* rather than system control
2. **Deterministic Oracle Grading**: Pure function scoring with edit-distance chain comparison — no LLM-as-judge
3. **Dense Reward Shaping**: Information-gain proxy rewards prevent sparse-reward exploration collapse
4. **Procedural Generation**: 5 failure templates × unlimited seeds = infinite curriculum (10K validated)
5. **Real Enterprise Patterns**: Connection pool exhaustion, OOM cascades, AZ failover bugs, correlated failures
6. **Adaptive Curriculum** (Theme #4 crossover): ELO-based difficulty scaling adjusts scenarios to agent skill
7. **Live Investigation UI**: Real-time SSE streaming with animated service dependency graph and investigation replay
8. **Token-Efficient LLM Agent**: Anthropic prompt caching on system prompt + scenario seed — each step costs ~200 tokens

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - Environment framework
- [HuggingFace TRL](https://github.com/huggingface/trl) - Training (SFT + GRPO)
- [Pydantic](https://docs.pydantic.dev/) - Type-safe models
- [FastAPI](https://fastapi.tiangolo.com/) - HTTP/WebSocket server

---

*Built for the Meta PyTorch OpenEnv Hackathon, April 2026*
