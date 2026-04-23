# PostmortemEnv: Teaching LLMs to Investigate, Not Just Control

**Theme 3.1: World Modeling (Professional) | Meta PyTorch OpenEnv Hackathon Apr '26**
**Team: Three Musketeers (Utkarsh, Mohit, Tanush) | Scaler AI Labs Sub-Prize**

---

## The Inversion: Control RL vs Epistemic RL

Every RL environment in the landscape teaches agents the same thing: *control*. Move a robot. Play a game. Manage a fleet. The agent acts, the world changes, and the reward measures how well the agent steered reality. But real SRE work is the opposite. When production goes down at 2 AM, the outage has *already happened*. The system state is frozen. Nobody is asking you to fix it in real-time — they're asking you to figure out *what went wrong and why*. That is an **investigation**, not a control problem. PostmortemEnv is the first RL environment that trains this skill: epistemic reasoning over frozen telemetry.

## The Task: What the Agent Sees

The agent receives a frozen snapshot of a 4-service microservice architecture after an outage:

```
frontend -> auth -> data -> batch
    |          |       |        |
    v          v       v        v
  Logs      Logs    Logs     Logs
  Traces    Traces  Traces   Traces
  Commits   Commits Commits  Commits
  Configs   Configs Configs  Configs
```

Each service has logs, distributed traces, recent git commits, configuration changes, and infrastructure events. The agent must investigate using 7 structured actions — `query_logs`, `fetch_trace`, `diff_commit`, `inspect_config`, `hypothesize`, `explain_chain`, and `submit` — within a bounded step budget.

Three difficulty tiers test progressively harder causal reasoning:

| Difficulty | Scenario | Challenge |
|---|---|---|
| **Easy** | Recent Deploy Blames | Single commit reduced connection pool 100→20 |
| **Medium** | Cascade Chain | Batch OOM cascades through all 4 services |
| **Hard** | Correlated Root Cause | Auth failover bug + network degradation combine |

A procedural seed generator built on 5 failure templates creates unlimited unique scenarios for training.

## The Reward: Deterministic Grading

No LLM-as-judge. The scoring formula is a pure function:

```python
r_terminal = 0.5 * is_correct_cause
           + 0.3 * chain_similarity
           + 0.2 * efficiency_bonus
           - 0.1 * n_wrong_hypotheses
```

Where `chain_similarity` uses Jaccard overlap on directed node/edge sets between the predicted and actual causal chains, and `efficiency_bonus = max(0, 1 - steps_used / max_steps)`. All scores are clamped to (0.01, 0.99) for validator compliance.

Dense per-step rewards prevent sparse-reward exploration collapse:
- **+0.05** per relevant fact discovered (information gain)
- **−0.01** per query (encourages efficiency)
- **−0.10** per wrong hypothesis
- **−0.005** per step (time pressure)

## The Numbers: Real Evaluation Data

Evaluated on 10 seeds per difficulty (`python train.py evaluate --n-seeds 10`):

| Agent | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Random | 0.099 | 0.095 | 0.108 | **0.101** |
| Oracle | 0.967 | 0.981 | 0.989 | **0.979** |
| **Gap** | **0.868** | **0.886** | **0.881** | **0.878** |

The 0.88 gap between random and oracle is the training headroom. SFT on `meta-llama/Llama-3.2-1B-Instruct` with LoRA on 60 oracle demonstrations is run on a Colab T4 to produce a real reward curve — the trained model must beat random to validate the environment.

Source: [`training_data/evaluation_results.json`](training_data/evaluation_results.json)

## Reproduce It

```bash
pip install -r requirements.txt
python train.py collect --n-seeds 20     # oracle demos
python train.py evaluate --n-seeds 10    # Random vs Oracle
python train.py plot                     # reward curves
uvicorn app:app --port 7860             # live demo
```

Or via Docker:
```bash
docker build -t postmortemenv .
docker run -p 7860:7860 postmortemenv
```

The full SFT training notebook is in [`train_notebook.py`](train_notebook.py), designed for Google Colab T4.

---

*PostmortemEnv. The first epistemic RL environment. Because the hardest part of an outage isn't fixing it — it's figuring out what happened.*
