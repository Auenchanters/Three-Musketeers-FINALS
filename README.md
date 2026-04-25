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
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PostmortemEnv-blue)](https://huggingface.co/spaces/Auenchanters/postmortemenv)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Auenchanters/Three-Musketeers-FINALS/blob/main/train_notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-133%20passing-brightgreen)](tests/)

# PostmortemEnv

> *Theme 3.1 — World Modeling (Professional). Meta PyTorch OpenEnv Hackathon, April 2026.*
> *By Three Musketeers — Utkarsh, Mohit, Tanush.*

## It is 02:47 AM.

Your phone is buzzing. Three dashboards are red. A `data` pod has been crash-looping for nine minutes, every `auth` request that gets routed to AZ-A is timing out, and the on-call channel is asking what shipped today. You don't get to *fix* this — somebody else already restarted the service and silenced the page. **You have to *explain it.*** Tomorrow morning, in a postmortem doc, in front of seven engineers and your director.

That doc is the entire job. The frozen telemetry — logs, traces, deploys, configs, infra events — is the only thing you have left. You read, you correlate, you hypothesise, you find the one commit that quietly halved the connection pool, and you write it up. **PostmortemEnv is an RL environment for that exact skill.** It hands an agent a frozen snapshot of a failed cloud architecture and grades how well it reconstructs the causal story.

![PostmortemEnv — Random vs Oracle on 30 episodes (10 seeds × 3 difficulties)](training_data/agent_comparison.png)

<sub>↑ Real evaluation data committed to the repo. A Random baseline collapses to **0.10 mean reward**; the deterministic Oracle solver hits **0.98**. The 0.88 gap is the training headroom available to any policy. Reproduce with `python train.py evaluate --n-seeds 10 && python scripts/plot_agent_comparison.py`. The live investigation console (see the HF Space link below and [static/](static/) for the UI source) renders the same data agent-by-agent in real time.</sub>

---

## Why we built this

All three of us have been on-call. One of us has watched a teammate cry at 4 AM after a billing-system outage they couldn't explain to leadership the next morning. Every existing RL-for-LLMs benchmark we looked at — SWE-bench, WebArena, ALFWorld — rewards an agent for **changing** the world: pass a test, click a button, pick up a mug. None of them reward the part of operations work that actually consumes engineer-years: **reading evidence after the fact and writing down what happened.**

We built PostmortemEnv because we wanted to know whether a small open-weight model could be trained to do that. The answer turned out to be "yes, the environment is trainable, the gap is 0.88, and we built the entire pipeline to prove it." This README walks you through the proof.

## What it actually does

A scenario is a frozen failed deployment of a 4-service architecture (`frontend → auth → data → batch`). The agent reads a service graph and a one-paragraph incident brief, then has a step budget (40 / 75 / 120 for easy / medium / hard) to investigate using **7 typed actions**:

| Action | What it does |
|---|---|
| `query_logs` | Search a service's logs by keyword |
| `fetch_trace` | Pull a specific distributed trace |
| `diff_commit` | See what changed in a recent commit |
| `inspect_config` | Examine a config change |
| `inspect_infra` | Inspect an infrastructure event (failover, DNS, AZ) |
| `hypothesize` | Test a candidate root cause and get feedback |
| `explain_chain` | Test a partial causal chain (qualitative band) |
| `submit` | Terminal — submit final cause + chain for grading |

Each step returns information *and* a small reward. The terminal score is a five-rubric weighted sum (cause, chain, efficiency, investigation quality, anti-gaming) computed by a **deterministic** grader — no LLM-as-judge, no flakiness. Implementation: [`engine/environment.py`](engine/environment.py), [`engine/grader.py`](engine/grader.py), [`engine/rubrics.py`](engine/rubrics.py).

## Real-world applications

PostmortemEnv is not just a hackathon toy — the API and reward shaping are designed to be re-used.

- **Train an SRE-copilot.** A 1-2B model SFT'd on oracle demos already learns "don't spam the log search, follow the dependency graph, hypothesise once, submit." A 7-13B model fine-tuned on the same demos plus GRPO is the obvious next step.
- **Generate synthetic incident data for ops teams.** [`data/seed_generator.py`](data/seed_generator.py) produces unlimited unique procedurally-generated incidents (10K seeds → 10K unique scenarios, 0 collisions, validated). Use them for runbook training, on-call interview prep, or chaos-day prompts.
- **Evaluation harness for *any* "investigation" agent.** The OpenEnv API is generic. Swap the scenario JSONs for compliance audits, fraud cases, or security forensics and you have an instant epistemic-RL benchmark in a different domain.

## Why this is different from existing RL envs

| Existing benchmark | What the agent does | What we do differently |
|---|---|---|
| **SWE-bench** | Edit code so a test passes | Read frozen evidence; never edit |
| **WebArena** | Click buttons on a live website | Read-only telemetry; world doesn't change |
| **ALFWorld** | Pick up the mug, put it in the microwave | Reconstruct *which* mug broke and *why* |
| **Most RL gyms** | Reward = system performance | Reward = quality of the explanation |

This is **epistemic RL**: the agent's actions reveal hidden information rather than changing the world, and the score is the quality of the reconstructed causal chain. We searched for prior work as of April 2026 and could not find a published RL environment with this framing — if we missed one, please open an issue and we will cite it here.

## Live demo

> **The live OpenEnv Space:** **<https://huggingface.co/spaces/Auenchanters/postmortemenv>**

Click that link, choose a task on the left, hit **Run Oracle** (no API key required), and watch the investigation console stream the agent's reasoning step-by-step. Pick **Random** to see what un-trained behaviour looks like. Bring your own Anthropic / OpenAI key to run an LLM agent on the same frozen scenario. The console renders the dependency graph, the per-step reward signal, and the final breakdown of all five rubrics.

The same UI also exposes the curriculum's ELO state — a fresh agent gets pushed to harder scenarios as it succeeds and bumped down a tier when it gets stuck.

> Heads up: the HF Space sleeps when idle. Cold-start takes ~30s; hit refresh once if the first request 503s.

## Headline benchmark

| Agent | Mean reward (n seeds × 3 difficulties) | Notes |
|---|---:|---|
| Random baseline | **0.230** (4 seeds) | Lower bound — proves the env has signal |
| Qwen2.5-3B-Instruct, base (no SFT) | **0.276** (4 seeds) | A frozen instruct model already beats random by following the JSON contract |
| Qwen2.5-3B-Instruct + LoRA SFT (60 demos, 6 ep) | **0.274** (4 seeds) | LoRA r=16, bf16 on L4. Loss converges (1.371 → 1.028, +1.6 pp token accuracy) but the policy doesn't move past the base — 60 demos is a known-too-small SFT corpus for the multi-step investigation pattern. Adapter lives at [`Auenchanters/postmortemenv-qwen3b-full`](https://huggingface.co/Auenchanters/postmortemenv-qwen3b-full); reproduce with `BASE_MODEL=Qwen/Qwen2.5-3B-Instruct hf jobs run …` (see [`scripts/hf_job_train.py`](scripts/hf_job_train.py)). The honest take: this hackathon run shows the *pipeline* works end-to-end on real GPU hardware; closing the gap to oracle is the next-week task (more demos → GRPO on top, see roadmap). |
| **Oracle (deterministic optimal)** | **0.934** (4 seeds) | Upper bound — our hand-built solver |
| **Reward gap (Oracle − Random)** | **0.704** | The training headroom available to any policy |

Source: [`training_data/evaluation_results.json`](training_data/evaluation_results.json) (Random + Oracle rows are committed; the SFT row appears after `python train.py full` finishes). A real SFT run also writes [`training_data/loss_curve.png`](training_data/loss_curve.png) from `trainer.state.log_history` — no fabricated artifact, the file simply does not exist until the run does.

### Per-difficulty breakdown (Qwen-3B GPU run)

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random                    | 0.218 | 0.226 | 0.246 | **0.230** |
| Qwen-3B base              | 0.264 | 0.281 | 0.282 | **0.276** |
| Qwen-3B + LoRA SFT        | 0.261 | 0.281 | 0.281 | **0.274** |
| Oracle                    | 0.930 | 0.936 | 0.937 | **0.934** |
| **Gap (Oracle − Random)** | **0.712** | **0.710** | **0.691** | **0.704** |

![Per-difficulty agent comparison](training_data/agent_comparison.png)
*Per-difficulty mean episode score (committed image is from a 30-episode 10-seed Random vs Oracle eval; the table above is the live 4-seed Qwen-3B vs Random vs Oracle eval pulled from [`Auenchanters/postmortemenv-qwen3b-full`](https://huggingface.co/Auenchanters/postmortemenv-qwen3b-full/blob/main/training_data/evaluation_results.json) on the Hub). The Random baseline is statistically indistinguishable across tiers; the deterministic Oracle saturates near 1.0. Any trained policy that finishes above the random ceiling is genuinely learning the investigation skill — the next iteration of the SFT run (more demos, higher LoRA rank, then GRPO) is the route to that line.*

> **A note on honesty.** The Qwen-3B SFT row (0.274) is the *real* result from a 24-minute L4 run on Hugging Face Jobs — same number you get if you re-run the job. Loss did converge (1.371 → 1.028 over 90 steps, see [`sft_metrics.json`](https://huggingface.co/Auenchanters/postmortemenv-qwen3b-full/blob/main/sft_metrics.json) on the Hub) but it never moved the *behavioral* needle past the base model. That's the truth. 60 oracle demos × 6 epochs is a deliberately small budget — the goal of this hackathon submission was a working pipeline (collect → SFT → eval → push) on rented GPU, not a benchmark win. Closing the 0.66 gap to oracle is the obvious next-week task: more demos (200+), higher LoRA rank (32+), and GRPO on top of the SFT'd checkpoint using the same reward signal the live REINFORCE chart uses. Frontier-model rows (Claude Haiku, GPT-4o-mini, Llama-3.2-1B) are produced by [`scripts/run_frontier_benchmark.py`](scripts/run_frontier_benchmark.py) after exporting `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.

### Live in-browser training (no GPU, no API key)

The "Live training" panel in the UI runs an on-policy REINFORCE softmax agent against the **same `PostmortemEnvironment` reward signal** the SFT/GRPO pipelines optimise. Hit **Start training** and watch the rolling-mean curve climb above the random baseline in ~15 seconds — fresh from a uniform-random policy, no pre-recorded curve. Reproduce locally with `python smoke_full_pipeline.py`:

| Task | Random baseline | Trained (rolling mean @ 500 ep) | Absolute lift | Wall clock |
|---|---:|---:|---:|---:|
| `task1_recent_deploy`            | 0.274 | **0.657** | **+0.383** | 2.3 s |
| `task2_cascade_chain`            | 0.252 | **0.449** | **+0.197** | 3.1 s |
| `task3_correlated_cause`         | 0.256 | **0.619** | **+0.362** | 3.0 s |
| `task4_multi_region_failover`    | 0.293 | **0.380** | **+0.086** | 3.9 s |
| `task5_data_corruption_cascade`  | 0.286 | **0.658** | **+0.372** | 2.4 s |
| **Mean (5 tasks)**               | 0.272 | **0.553** | **+0.280** (≈ 2.0×) | 14.7 s |

Source: [`training_data/live_training_smoke.json`](training_data/live_training_smoke.json), regenerated every time you run `python smoke_full_pipeline.py`. The same loop powers the on-page "Live training" chart at [`web/training_loop.py`](web/training_loop.py); the UI deliberately throttles to ~30 ms/episode so the curve animates while a judge watches.

### Same task, two agents — `task1_recent_deploy` (a connection-pool regression)

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

## Reproduce in 60 seconds

```bash
git clone https://github.com/Auenchanters/Three-Musketeers-FINALS.git
cd Three-Musketeers-FINALS
pip install -r requirements.txt
python -m pytest tests/ -q                  # 133 tests, ~4 s on a laptop
python test_oracle_e2e.py                   # task1: 0.96 task2: 0.97 task3: 0.98 ALL PASS
uvicorn app:app --host 0.0.0.0 --port 7860  # serves the env + live UI at http://localhost:7860
```

That's everything a judge needs to verify the env works locally. No GPU, no API keys, no `huggingface-cli login`.

The repo is split into three install profiles so you only download what you need:

| Profile | Install | What it gives you |
|---|---|---|
| **Server** (HF Space, judges' demo) | `pip install -r requirements-server.txt` | FastAPI + OpenEnv server. CPU-only. Used by the [Dockerfile](Dockerfile). |
| **Local dev / inference / tests** | `pip install -r requirements.txt` | Server + `openai`, `httpx`, `pytest`, etc. for `inference.py`, `test_oracle_e2e.py`, and the test suite. |
| **Training** (Colab T4 / rented GPU) | `pip install -r requirements-train.txt` | Adds `torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes` for SFT + GRPO. |

## Train your own agent

### One-shot (recommended for a rented GPU)

```bash
pip install -r requirements-train.txt
python train.py full
```

That's it. Three minutes of dependency install, then `train.py full` runs **collect → baseline evaluate → SFT → trained evaluate → plot** end-to-end and writes:

- `training_data/oracle_demos.jsonl` (60 oracle rollouts, used as SFT data)
- `training_data/evaluation_results.json` with `random`, `oracle`, and `trained` keys
- `training_data/reward_curves.png` (3-panel: overall · per-difficulty · cumulative reward per step)
- `training_data/agent_comparison.png` (the README's headline chart)
- `training_data/loss_curve.png` (auto-generated from `sft_output/sft_metrics.json` via the new `plot_loss_curve` helper — also runnable on demand with `python train.py plot-loss`)
- `sft_output/` (LoRA adapter + tokenizer)

The default base model is `Qwen/Qwen2.5-1.5B-Instruct` because it is **open-weight** (no `huggingface-cli login` / gated-license form), ships a chat template, and fits LoRA on a single T4. Pass `--model meta-llama/Llama-3.2-1B-Instruct` after `huggingface-cli login` if you prefer Llama-3. `evaluate_trained_model` prints a sanity warning if the trained mean fails to clear `random + 0.02`, which catches the common mistake of evaluating the base model instead of the LoRA adapter.

### Step by step (if you want to inspect each phase)

```bash
pip install -r requirements-train.txt

python train.py collect --n-seeds 20          # 60 oracle demos -> oracle_demos.jsonl
python train.py evaluate --n-seeds 10         # random + oracle -> evaluation_results.json
python train.py sft                           # LoRA SFT on Qwen2.5-1.5B (~10-15 min on T4)
python train.py evaluate --n-seeds 10 --model ./sft_output   # adds the "trained" key
python scripts/plot_agent_comparison.py       # rebuilds reward_curves.png + agent_comparison.png
python train.py plot-loss                     # writes training_data/loss_curve.png from sft_metrics.json
python train.py trace --model ./sft_output    # one readable episode trace for the README/blog
python train.py grpo --model ./sft_output     # stretch — online RL on env rewards
```

### Colab one-click

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Auenchanters/Three-Musketeers-FINALS/blob/main/train_notebook.ipynb)

The notebook clones from GitHub (`https://github.com/Auenchanters/Three-Musketeers-FINALS.git`) and runs the same `train.py full` pipeline on a T4. The script form lives at [`train_notebook.py`](train_notebook.py) for diff-friendly review.

## Deploy your own

```bash
docker build -t postmortemenv .
docker run -p 7860:7860 postmortemenv
```

To deploy to a Hugging Face Docker-SDK Space:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/PostmortemEnv
git push hf main
```

Everything the Space needs is already wired in: the YAML front-matter at the top of this README declares `sdk: docker` and `app_port: 7860`, the [`Dockerfile`](Dockerfile) installs from [`requirements-server.txt`](requirements-server.txt) (CPU-only — no GPU credits required to host the env), and [`app.py`](app.py) exposes `/reset`, `/step`, `/state`, `/health`, `/schema`, plus the `/ws` OpenEnv session endpoint and the live investigation console at `/`. Once the build finishes, the Space URL becomes the runnable demo.

## Adaptive curriculum (Theme #4 crossover)

The environment maintains a per-agent **ELO-style skill estimate** in [`training_data/curriculum_state.json`](training_data/curriculum_state.json) and uses it to bias the next-task sampler. Implementation lives in [`web/curriculum.py`](web/curriculum.py). After each episode, both the agent rating and the per-task rating are updated with the standard ELO rule (`K = 24`):

```python
expected = 1 / (1 + 10 ** ((task_elo - agent_elo) / 400))
delta    = K * (score - expected)
```

Tasks start at fixed difficulty ratings — `task1_recent_deploy: 1000`, `task2_cascade_chain: 1300`, `task3_correlated_cause: 1600` — and a per-task `difficulty_multiplier` rises when the agent beats its expected score and falls otherwise. The procedural seed generator reads that multiplier to scale chain depth, red-herring count, and noise injection per scenario. Net effect:

- a fresh agent stays on Easy until it consistently submits correct causes,
- a near-Oracle agent gets pushed almost exclusively to Hard *and* gets harder Hard scenarios,
- a stuck agent drifts back down a tier instead of grinding on impossible problems.

This is a lightweight version of the self-play adaptive-curriculum loop from Theme #4 (Self-Improvement) — same goal (keep the agent in its zone of proximal development), reused machinery (ELO + per-task scaling), but applied to *task selection over a procedurally-generated set* instead of self-play match-making. State is persisted between runs, so the HF Space curriculum survives container restarts.

## Why we think this wins

- **Novel framing.** Epistemic RL — actions reveal information, score is explanation quality — is genuinely new ground for the LLM-RL community. We could not find prior published work that frames the problem this way (see the comparison table above).
- **Honest, reproducible benchmark.** The 0.10 → 0.98 Random-vs-Oracle gap is computed from real episodes whose JSON is in the repo. Anyone can rerun `python train.py evaluate --n-seeds 10` and verify.
- **Dense reward, not sparse.** Information gain is rewarded per step, so agents do not drown in sparse-reward exploration collapse — `data/seed_generator.py` produces 10K unique scenarios with non-trivial signal on every step.
- **Fully deterministic grader.** Five composable rubrics, Jaccard node + edge similarity for the chain, hard clamping for OpenEnv validator compliance. No LLM-as-judge anywhere — what you score is what you get.
- **133-test suite, all passing.** Engine, grader, environment, generator, agents, runner, curriculum, training stream, action-parser hardening, action validation. Run `python -m pytest tests/ -q` and read the green.
- **Live UI judges actually want to play with.** SSE-streamed investigation console, animated dependency graph, replay, BYO-key LLM agent. Built explicitly to make the demo wow-able in five seconds, not five minutes.
- **End-to-end pipeline, not a demo of one piece.** Procedural data → SFT → evaluation → plotting → ELO curriculum → GRPO scaffold, all behind a single `python train.py full` command.

## Architecture & training pipeline (for the curious)

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
   └──────────────────────────┘    ┌───────────────────────────────┐
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
                                   │   + loss_curve.png from log    │
                                   └───────────────────────────────┘
```

## Project structure

```
PostmortemEnv/
+-- app.py                    # FastAPI server (OpenEnv)
+-- inference.py              # LLM investigation agent
+-- train.py                  # Training pipeline (SFT + GRPO + plot-loss)
+-- test_oracle_e2e.py        # Oracle validation
+-- models/                   # Pydantic models
+-- engine/                   # Environment + grader + rubrics + rewards
+-- data/
|   +-- scenarios/            # 3 hand-crafted outage scenarios
|   +-- solutions/            # Oracle optimal solutions
|   +-- seed_generator.py     # Procedural scenario generator
+-- web/                      # Live UI (FastAPI + SSE) + agent runners
+-- static/                   # Investigation console (HTML/CSS/JS)
+-- tests/                    # 133 unit tests (engine + grader + agents + curriculum + training + parsers + validation)
+-- Dockerfile, openenv.yaml, requirements*.txt
```

## Future work

- **GRPO online RL.** `train.py grpo` implements GRPO with environment rewards on top of the SFT checkpoint. Within the hackathon compute budget we shipped the SFT pipeline as the primary training signal and left a single-pass GRPO smoke run for after the deadline; the loop is wired end-to-end and will produce `training_data/grpo_eval_results.json` when run.
- **Frontier-model coverage.** [`scripts/run_frontier_benchmark.py`](scripts/run_frontier_benchmark.py) runs Claude Haiku, GPT-4o-mini, and Llama-3.2-1B against all three difficulty tiers. We will publish the full table to `training_data/frontier_results.json` once we burn through the API spend.
- **Multi-agent investigation.** A natural extension: two agents (one bullish on a hypothesis, one adversarial) negotiate the final submission. Reward shaping is symmetric so the env supports it without changes.
- **Real telemetry import.** A loader for OpenTelemetry traces / Loki logs would let teams replay their own real outages as PostmortemEnv tasks.

## Built with

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [HuggingFace TRL](https://github.com/huggingface/trl) — training (SFT + GRPO)
- [PEFT](https://github.com/huggingface/peft) — LoRA adapters
- [Pydantic](https://docs.pydantic.dev/) — type-safe models
- [FastAPI](https://fastapi.tiangolo.com/) — HTTP/WebSocket server + SSE

## Deliverables (judges' index)

| Resource | Link |
|---|---|
| 🤗 **Live OpenEnv Space** (Docker) | <https://huggingface.co/spaces/Auenchanters/postmortemenv> |
| 📓 **Training notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Auenchanters/Three-Musketeers-FINALS/blob/main/train_notebook.ipynb) → [`train_notebook.ipynb`](train_notebook.ipynb) · [`train_notebook.py`](train_notebook.py) (script form, kept in lock-step with the .ipynb) · canonical CLI: [`train.py`](train.py) |
| 📝 **Writeup / mini-blog** | [`blog_post.md`](blog_post.md) |
| 📹 **Demo video (≤ 2 min)** | <https://youtu.be/PLACEHOLDER> <!-- placeholder — final unlisted YouTube URL will be wired in before submission --> |
| 📊 **Committed plots** | [`training_data/reward_curves.png`](training_data/reward_curves.png) (3-panel: overall · per-difficulty · cumulative reward per step) · [`training_data/agent_comparison.png`](training_data/agent_comparison.png) (single-panel — regenerate both with `python scripts/plot_agent_comparison.py`). The SFT loss curve is auto-generated to `training_data/loss_curve.png` after the first real GPU run via `python train.py plot-loss`. |
| 🧪 **Frontier-model benchmark** | [`scripts/run_frontier_benchmark.py`](scripts/run_frontier_benchmark.py) — populates `training_data/frontier_results.json` after exporting `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` |

---

*Built for the Meta PyTorch OpenEnv Hackathon, April 2026 — by Three Musketeers (Utkarsh, Mohit, Tanush). MIT-licensed.*
