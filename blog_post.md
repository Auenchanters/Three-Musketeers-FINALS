# PostmortemEnv: Teaching LLMs to Investigate, Not Just Control

**Theme 3.1: World Modeling (Professional) | Meta PyTorch OpenEnv Hackathon Apr '26**
**Team: Three Musketeers (Utkarsh, Mohit, Tanush) | Scaler AI Labs Sub-Prize**

---

## 03:02 UTC

The pager goes off. Checkout is throwing 5xx errors. Twelve dashboards. A hundred recent commits. Eight services upstream. You have two minutes before someone in the group chat asks "what's the impact?" — and twenty before someone wants a root cause. You are not being asked to *fix* anything. The outage already happened. You are being asked to *explain* it.

Today's tooling tells you **what** is broken — error rates, p99 latencies, alarm states — but almost nothing about **why**. Every "AI for ops" demo we've seen is the same: predict the failure, autoscale the cluster, restart the pod. None of them teach the actual hard skill, which is reading a frozen pile of telemetry and reconstructing the chain of events. That skill is *epistemic*, not *control*: it's about gathering evidence, ranking hypotheses, and stopping when the story is consistent. RL hasn't had an environment that rewards it.

**PostmortemEnv is that environment.** The agent receives a frozen 4-service architecture after an outage and a budget of investigation steps. It picks a service to query logs for. It diffs a commit. It traces a request. It hypothesizes a root cause and gets a binary "warmer / colder." When it's confident, it submits — and the deterministic grader scores both the cause it named and the causal chain it reconstructed. The reward signal cleanly separates a flailing agent from a thoughtful one (Random: 0.10, Oracle: 0.98 — see numbers below). That gap is what makes the environment trainable.

> *"This is the first benchmark we've found where 'guess the answer' literally cannot beat 'do the work,' because the work is what produces information."* — our team at hour 30 of the build.

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

## One investigation, end-to-end

Here is what the reward signal actually looks like over a single episode of `task1_recent_deploy` — a connection-pool regression introduced by a recent deploy. The Oracle (left) climbs the dependency graph, then submits. A random agent (right) sprays queries and times out.

```text
ORACLE  (final_score = 0.960, 8 steps)               │  RANDOM  (final_score = 0.155, 9 steps)
─────────────────────────────────────────────────────│─────────────────────────────────────────────
1  query_logs   data    "error"          r = 0.135   │  1  query_logs   frontend  "error"   r = 0.085
   → finds 3 connection-pool ERRORs in `data`        │     → finds noise in frontend (irrelevant)
2  query_logs   data    "deploy"         r = 0.035   │  2  query_logs   data      "deploy"  r = 0.035
   → confirms a recent deploy on `data`              │     → lucky hit but doesn't follow up
3  diff_commit  commit-a1b2c3            r = 0.035   │  3  query_logs   auth      "deploy"  r = 0.010
   → sees `max_conns: 100 → 20`                      │     → no signal, query already exhausted
4  query_logs   data    "connection"     r = 0.185   │  4  query_logs   frontend  "error"   r = 0.010
   → discovers the pool-exhaustion fact              │     → repeated query, info already known
5  fetch_trace  trace-001                r = 0.035   │  5  query_logs   batch     "error"   r = 0.010
   → confirms the cascade pattern                    │     → batch isn't even involved
6  hypothesize  commit-a1b2c3            r = 0.015   │  6  query_logs   frontend  "error"   r = 0.010
   → "warmer": grader says correct cause             │  7  query_logs   auth      "deploy"  r = 0.010
7  explain_chain  data → auth → frontend r = 0.015   │  8  query_logs   frontend  "deploy"  r = 0.010
   → chain similarity 1.00                           │     → at this point it's pattern-matching
8  submit                                r = 0.960   │  9  submit         "unknown"          r = 0.155
   ✓ correct cause, correct chain, fast              │     ✗ no cause, no chain, time exhausted
```

Two things to notice:

1. **Information-gain reward is non-flat.** Steps 1 and 4 pay 0.135 and 0.185 respectively because they each surface a hidden fact relevant to the root cause. Steps 2, 3, 5 pay only 0.035 — they're useful context but don't reveal new information. Repeated identical queries pay nothing. This is why the random agent collapses to floor reward after step 2: it keeps re-asking questions whose answers are already in `known_facts`.
2. **The terminal score dominates.** Even an Oracle that nails every intermediate query only earns ~0.45 in step rewards. The remaining ~0.5 comes from the final `submit` — the deterministic grader rewards (i) cause match, (ii) chain Jaccard similarity, (iii) efficiency, and penalises wrong hypotheses. A model that *only* maximises step reward without committing to a hypothesis would saturate at ~0.5; a model that submits early with low confidence is penalised twice (0.0 cause match × 0.5 weight). The shape of the reward forces the agent to *finish the job*.

You can replay any oracle episode locally with `python test_oracle_e2e.py` and any random episode with a 10-line loop over `env.step(Action(action_type=ActionType.QUERY_LOGS, service=...))`. We've also packaged a frontier-model benchmark — `python scripts/run_frontier_benchmark.py` — that runs Claude Haiku 4.5, GPT-4o-mini, and Llama-3.2-1B against the same three tasks and writes a comparable score table. Even partial frontier coverage is informative: anything in the 0.30–0.60 band proves that *real reasoning* (not memorisation) is what the environment is selecting for.

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
