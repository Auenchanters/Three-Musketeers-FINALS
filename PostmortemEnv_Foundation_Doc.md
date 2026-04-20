# PostmortemEnv — Foundation Document

**Team:** Three Musketeers (Utkarsh Singh Yadav — Lead; L Mohit Jain; Tanush Deepak)
**Event:** Meta PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale
**Dates:** 25–26 April 2026, Scaler SoT Electronic City, Bangalore
**Status:** Round 2 concept — locked in on 19 April 2026. Build window: 6 days pre-onsite + 2 days onsite.

---

## 0. TL;DR

PostmortemEnv is an OpenEnv-native reinforcement learning environment where an LLM agent investigates a cloud outage that has **already happened**. The agent receives a frozen telemetry snapshot (logs, traces, commits, configs, infra events) and must identify the root cause and the causal chain of downstream failures within a limited query budget. Grading is deterministic: exact-match on root cause, edit distance on chain, efficiency bonus.

The innovation is structural, not thematic. Almost every other team in the 800-team field is building *control* environments where the agent acts on the world. PostmortemEnv is an *epistemic* environment where the agent investigates what the world did. That inversion of the RL loop is the pitch hook and the defensible technical novelty.

We reuse roughly 70% of our Round 1 CloudWarEnv/CloudFinOpsEnv codebase (service DAG, chaos engine, Pydantic models) by inverting the chaos engine: instead of injecting failures for the agent to prevent in real time, it scripts completed outage traces plus ground-truth cause/chain for the agent to reconstruct.

---

## 1. Team & Event Context

### Team
- Utkarsh Singh Yadav — Team Lead, primary pitch deliverer, architecture owner.
- L Mohit Jain — Engineer.
- Tanush Deepak — Engineer.

Roles inside the build will be assigned in the Day 0 kickoff, but recommended split is: Utkarsh on environment + grader + pitch; Mohit on training pipeline + Unsloth/TRL integration; Tanush on demo + HF Spaces deploy + video.

### Event constraints (non-negotiable)
- **OpenEnv latest release** — REST API in Docker. Three endpoints: `reset()`, `step(action)`, `state()`.
- **Compute limits:** 2 vCPU, 8 GB RAM, inference must complete in under 20 minutes.
- **LLM calls via OpenAI client** — required interface.
- **Structured stdout logs** — `[START]` / `[STEP]` / `[END]` markers required.
- **Fully offline execution** — no external APIs, no external datasets at runtime.
- **Minimum submission:** OpenEnv-compliant environment, minimal training script in Unsloth or HF TRL on Colab, mini-blog on Hugging Face or mini-video on YouTube under 2 minutes.

### Round 2 judging rubric
| Criterion | Weight |
|---|---|
| Environment Innovation (novel, creative, meaningful test) | 40% |
| Storytelling (clear problem/environment/agent, engaging demo) | 30% |
| Showing Improvement in Rewards (observable training progress) | 20% |
| Reward and Training Script/Pipeline Setup | 10% |

Pitch format: 3 minutes + 2 minutes Q&A. Each judge evaluates 10–15 teams.

### Field context
Top 800 of ~70,000 registered teams. Nearly all finalist teams are senior engineers from top companies. Our edge is not raw execution quality — it is idea selection in a field where ~100 teams will build cloud/SRE control environments and very few will build investigation environments.

---

## 2. What PostmortemEnv Is

### Plain-English description
A production cloud service just experienced a cascading outage. Four services were involved. Logs, traces, recent commits, config changes, and infrastructure events from the incident window are preserved. Somewhere in that haystack is the true root cause — a specific commit, config change, or infra event — and a specific causal chain explaining how the failure propagated through the service dependency graph.

The agent, playing the role of an on-call SRE writing the postmortem, must identify both within a bounded number of queries. It cannot see the full haystack at once. It must query selectively — `query_logs(service, keyword)`, `fetch_trace(id)`, `diff_commit(hash)` — to narrow the hypothesis space, then submit a final answer.

Grading is deterministic against scripted ground truth. A random-query baseline performs terribly (the haystack is large). A trained policy that learns to start from recent deploys and narrow via traces performs substantially better. The reward curve bends visibly. That visibility is the 20% of the rubric locked in by design.

### The one-line pitch
*"Production just burned for 12 minutes. Four services cascaded. Someone has to write the RCA. We built the first RL environment that teaches an agent to do it."*

### The core framing
**Epistemic RL, not control RL.** Every claim of novelty we make in the pitch traces back to this single framing. Memorize it. Repeat it in Q&A.

---

## 3. Why This Idea Wins (Strategic Rationale)

### Rubric alignment, criterion by criterion

**Environment Innovation (40%).** The novelty is the investigation-vs-control inversion. Almost every team builds a control environment because that is what RL tutorials teach. An epistemic environment — where actions are queries and the reward is tied to information gain — is structurally rare. We are not claiming a novel domain (AIOps exists); we are claiming a novel problem setup applied to a familiar domain. That is exactly the kind of claim serious judges respect because it is defensible under technical questioning.

**Storytelling (30%).** Whodunit is the oldest narrative format humans have. Every judge instantly understands "find the commit that killed prod." The demo has built-in tension — a cascading red dashboard, a hidden culprit — and a reveal moment when the trained agent identifies the right commit hash. Most teams will show a bar chart climbing. We show a mystery being solved.

**Showing Improvement in Rewards (20%).** Random-query baselines are genuinely bad at this task because the haystack is large and most queries yield no signal. A trained policy is genuinely good because the task has exploitable structure (deploys tend to cluster near incidents, traces reveal blast radius). The gap between the two is large, robust, and photogenic on a reward curve.

**Reward and Training Pipeline (10%).** Scripted ground truth + exact-match on cause + edit distance on chain + efficiency bonus = a cleaner, more coherent reward than any shaping-heavy control environment can offer.

### Field positioning
In a field of 800 senior-engineer teams, approximately:
- 200 will build customer support / triage environments.
- 150 will build coding / PR-review / debugging environments.
- 100 will build cloud / SRE / DevOps *control* environments.
- 80 will build sales / CRM workflows.
- ~270 will spread across the remaining obvious lanes (personal assistant, SQL agents, negotiation sims, research agents).

PostmortemEnv sits in a lane I estimate fewer than 10 teams will enter: investigation/RCA agents with deterministic oracle grading. That density gap is our primary moat.

### Sub-prize fit
Our primary sub-prize target is **Scaler AI Labs** (Multi-App RL Environment for Enterprise Workflows). Postmortem authoring is literally an enterprise workflow with business rule nuance (severity classification, blast radius estimation, attribution decisions). We frame the pitch accordingly.

Secondary defensive angle: **Theme 2 (Long-Horizon Planning & Instruction Following)** — the hard task has 120 steps, sparse terminal reward, requires hypothesis revision after early wrong turns.

### Code-reuse advantage
We are not starting from scratch. Our existing CloudWarEnv/CloudFinOpsEnv artifacts — service DAG, chaos engine, Pydantic observation/action/state models, inference.py skeleton, Dockerfile, openenv.yaml, HF Spaces deployment scripts — carry over with modifications. A team starting fresh on any idea of comparable ambition will be shipping duct-taped work. We will not.

---

## 4. Theme & Sub-Prize Mapping

| Lane | Fit | Notes |
|---|---|---|
| Theme 2 — Long-Horizon Planning | Strong | 120-step hard task, sparse terminal reward, requires recovery from early wrong hypotheses |
| Theme 3.1 — World Modeling (Professional) | Strong | Partially observable world, belief updating, structured enterprise workflow |
| Theme 1 — Multi-Agent Interactions | Do not claim | We are single-agent. Claiming multi-agent would be the same mistake we were about to make with CloudWar. |
| Theme 4 — Self-Improvement | Possible stretch | If we add auto-generated outages as curriculum, this opens up. Not priority. |
| **Scaler AI Labs sub-prize** | **Primary target** | Frame as enterprise postmortem workflow |
| Mercor sub-prize (capped/uncapped reward scaling with token output) | Possible | Could apply if we let the agent write the final postmortem narrative, with length-scaled reward. Stretch goal only. |

---

## 5. Environment Design Specification

### 5.1 Setting

Four-service cloud architecture with fixed dependency DAG: `frontend → auth → data → batch`. Each service has its own log stream, its own deploy history, and its own config store. The service graph, service names, and structural metadata are observable. The incident's true cause, true causal chain, and blast radius weights are hidden and must be inferred via queries.

An outage has already occurred and is now frozen. The agent's job is forensic.

### 5.2 Observation space

The agent does not receive the full haystack as one observation. Instead, each `step()` returns the result of whatever query the agent submitted. A session-level "state summary" observation is available via `state()` for debugging but is not part of the agent's policy input during training.

The hidden haystack underlying each episode contains approximately:

| Data type | Volume per episode |
|---|---|
| Log lines | ~5,000 across 4 services (mostly noise, ~50 actually relevant) |
| Distributed traces | 50 (~5 show the cascade clearly) |
| Recent commits (24h window) | 10 (1 or 2 are the culprit) |
| Config changes (24h window) | 20 (1 or 2 are the culprit) |
| Infrastructure events (24h window) | 5 (0–1 are contributory) |
| Service dependency graph | Always fully visible |

Query responses are text blobs (log excerpts, commit diffs, etc.), returned in a Pydantic `Observation` model with a `query_result: str` field, a `remaining_budget: int` field, and a `known_facts: List[str]` field that accumulates facts the agent has confirmed so far.

### 5.3 Action space

Discrete action set, 7 primary action types:

| Action | Parameters | Purpose |
|---|---|---|
| `query_logs` | service, time_window, keyword | Retrieve matching log lines |
| `fetch_trace` | trace_id | Retrieve a specific distributed trace |
| `diff_commit` | commit_hash | See what changed in a specific commit |
| `inspect_config` | config_id | See a specific config change |
| `hypothesize` | cause_entity_id | Submit a candidate root cause for feedback (correct/incorrect) without ending the episode |
| `explain_chain` | ordered list of (service, effect) | Submit a causal chain hypothesis for feedback |
| `submit` | final_cause, final_chain | Terminate episode with final answer |

`hypothesize` and `explain_chain` are intermediate checks, not terminal. Wrong intermediate hypotheses cost small penalty; correct ones confirm progress. Only `submit` terminates and triggers final grading.

### 5.4 Reward function

**Dense per-step component (for training signal):**
```
r_step = +alpha * information_gain_proxy - beta * query_cost
```

`information_gain_proxy` is a rule-based approximation — the number of new "relevant facts" the query uncovered, where relevant facts are pre-labeled by the oracle during episode generation. This keeps the reward shaping deterministic and cheap to compute. `query_cost` is a small constant per step to push the agent toward efficiency.

**Terminal component (on `submit`):**
```
r_terminal = 0.5 * is_correct_cause 
           + 0.3 * chain_similarity 
           + 0.2 * efficiency_bonus 
           - 0.1 * n_wrong_hypothesize_calls
```

Where:
- `is_correct_cause`: 1 if submitted cause entity matches ground truth exactly, else 0.
- `chain_similarity`: `1 - normalized_edit_distance(predicted_chain, true_chain)`, clamped to [0, 1].
- `efficiency_bonus`: `max(0, 1 - steps_used / max_steps)`.
- `n_wrong_hypothesize_calls`: count of incorrect intermediate hypotheses submitted.

Final episode score reported to the grader is `r_terminal`, clamped to [0.0, 1.0].

**Fallback reward if information-gain shaping proves unstable:** Binary "relevant query" reward — +1 if the query returned oracle-labeled relevant content, 0 otherwise. Less elegant but more robust. Keep this ready as a drop-in by Day 4.

### 5.5 Tasks (easy → medium → hard)

| Task | Scenario | Step budget | Ground-truth cause archetype |
|---|---|---|---|
| Task 1 — *Recent Deploy Blames* | Single-service outage caused by one of the last 10 commits to that service | 40 | Recent commit |
| Task 2 — *Cascade Chain* | Cross-service cascade: a dependency bug in an upstream service propagates through the DAG | 75 | Commit OR config change in upstream service, plus 2–3-hop chain |
| Task 3 — *Correlated Root Cause* | A recent deploy and a concurrent infra event (AZ degradation) combine non-obviously — neither alone would have caused the outage | 120 | Tuple of (commit, infra_event), plus full-depth chain |

Each task exposes **at least 100 distinct seeds**. Seeds determine which entity is the true cause, the chain shape, and the noise distribution. This prevents memorization.

### 5.6 Oracle grader

The grader is a pure function of (agent's final `submit` payload, scripted ground truth). It does not invoke any LLM and contains no stochasticity. Pseudocode:

```python
def grade(submission, ground_truth):
    is_correct_cause = (submission.final_cause == ground_truth.cause)
    chain_similarity = 1 - normalized_edit_distance(
        submission.final_chain, ground_truth.chain
    )
    efficiency_bonus = max(0, 1 - submission.steps_used / ground_truth.max_steps)
    score = (
        0.5 * is_correct_cause
        + 0.3 * chain_similarity
        + 0.2 * efficiency_bonus
        - 0.1 * submission.n_wrong_hypotheses
    )
    return max(0.0, min(1.0, score))
```

The exact weights above are a reasonable first pass. **Finalize these weights on Day 2 after a 500-seed calibration sweep** that confirms: (a) random agent scores near 0, (b) a hand-crafted heuristic agent scores in the 0.3–0.5 range, (c) ceiling of 1.0 is achievable by an oracle-aware agent. If any of those three fails, adjust weights before coding the rest.

---

## 6. Technical Architecture

### 6.1 OpenEnv compliance
- Pydantic models for `Observation`, `Action`, `Reward`, `State`.
- REST endpoints: `POST /reset`, `POST /step`, `GET /state`.
- Dockerfile targets Python 3.11 slim, installs OpenEnv, bundles outage seed data.
- `openenv.yaml` declares the environment name, task IDs, max_steps per task, and scoring metadata.

### 6.2 inference.py contract
Must:
- Use `OpenAI` client for LLM calls.
- Emit structured stdout: `[START]` at episode begin, `[STEP N] action=... observation=... reward=...` at each step, `[END] final_score=...` at termination.
- Run fully offline (no external network calls beyond the OpenAI client if configured with a local endpoint or bundled credentials).
- Complete full evaluation suite in under 20 minutes on 2 vCPU / 8 GB RAM.

### 6.3 Deployment
Hugging Face Spaces, Docker SDK. Single-container deploy. Reuse Round 1 Space template — only the environment logic changes.

### 6.4 Training pipeline (Round 2 minimum requirement)
Colab notebook demonstrating measurable improvement:
- **Baseline A (random):** random action selection within the action space.
- **Baseline B (heuristic):** hand-crafted policy that checks recent commits first, then traces, then configs. Represents a "competent junior engineer."
- **Trained policy:** Unsloth or HF TRL. Likely PPO or GRPO on a small open-weight LLM (Qwen 2.5 7B or Llama 3.2 3B), fine-tuned on collected successful trajectories from the heuristic baseline plus self-play.

Reward curves to show at pitch: Baseline A (flat near 0), Baseline B (flat around 0.3–0.5), Trained (rising from Baseline B to 0.7+). **Decide exact model + training algorithm on Day 0 kickoff** — this is a real open decision, see Section 10.

---

## 7. 8-Day Build Plan

This plan assumes 6 days pre-onsite (20–23 April) and 2 days onsite (25–26 April). Today is 19 April; kickoff is Day 0 (20 April).

### Day 0 (20 April) — Kickoff
- Full team read this document.
- Finalize role split.
- Finalize oracle weights via the 500-seed calibration sweep (see Section 5.6).
- Decide training model and algorithm (Section 10 open decision).
- Fork Round 1 repo into `postmortem-env` branch; prune unused code.

### Days 1–2 (21–22 April) — Environment core
- Invert the chaos engine: from *real-time injection* to *scripted outage generator*. Output per seed: frozen telemetry bundle + ground-truth cause + ground-truth chain + relevant-fact labels.
- Define the hypothesis space (enumerable list of cause entity types).
- Implement query interface for all 7 action types.
- Generate 100 seeds per task × 3 tasks = 300 total seeds, committed to the repo.

### Days 3–4 (23–24 April) — Reward and grader
- Implement deterministic oracle grader (Section 5.6).
- Implement per-step information-gain reward shaping.
- Implement binary-reward fallback as a feature flag.
- Validate: random agent runs 100 seeds, heuristic agent runs 100 seeds, scores match the calibration targets from Day 0.

### Day 5 (25 April, onsite Day 1) — Pipeline and training
- `inference.py`, Dockerfile finalization, `openenv.yaml`.
- HF Spaces deploy — confirm the container runs under the 20-min limit.
- Colab notebook: baseline evaluations + training run kickoff using the HF compute credits.

### Day 6 (26 April, onsite Day 2) — Polish and pitch
- Training run completion. Cut reward curves for pitch.
- Record the 2-minute mini-video.
- Write the HF mini-blog post.
- **20+ rehearsals** of the 3-minute pitch. Not 2. Twenty.
- Q&A kill-list: write the 15 hardest questions, rehearse answers.
- Sleep properly the night before pitch day. Do not pull an all-nighter.

### Buffer protocol
If anything runs behind on Days 1–4, cut the hard task (Task 3) first. Ship with 2 tasks rather than a broken 3-task environment. The pitch story still holds with 2 tasks.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Information-gain reward shaping is unstable or gameable | Binary "relevant query" reward fallback ready by Day 4, feature-flagged |
| Agent memorizes seeds instead of generalizing | 100+ seeds per task with randomized noise distributions; evaluation on held-out seeds |
| Query budget is too loose (random agent lucks into answer) or too tight (trained agent can't escape zero) | 500-seed calibration sweep on Day 0 locks budgets per task |
| Training doesn't produce a visibly improving curve | Use SFT on heuristic baseline trajectories first; if PPO/GRPO doesn't converge in time, ship SFT-only and frame as "foundation for future RL work" |
| Judges ask "how is this different from existing AIOps tools?" | Rehearsed Q&A answer: existing tools are rule-based or supervised; this is an RL benchmark for causal-attribution reasoning with a deterministic oracle — no such benchmark exists |
| Judges ask "isn't this just supervised classification?" | Rehearsed answer: No — the action space includes which information to gather, not just which label to emit. The agent must learn both an investigation policy and a final-answer policy. |
| Onsite compute limits block training | Submit SFT results; training can be demonstrated in Colab during the pitch if needed |
| One teammate gets sick | Utkarsh as lead can deliver the full pitch solo; Mohit and Tanush's work is checkpointed in git nightly |

---

## 9. Pitch Strategy

### The 3-minute structure

| Time | Content |
|---|---|
| 0:00–0:20 | **Hook.** Cold open: "Production just burned for 12 minutes. Four services cascaded. Someone has to write the RCA." |
| 0:20–0:50 | **The inversion.** State the one sentence that defines the project: "Every RL environment we looked at teaches agents to control the world. We built one that teaches agents to figure out what the world did." Contrast: control vs epistemic RL. |
| 0:50–1:40 | **Live demo.** Show one specific seed. Naive agent flounders, exhausts budget, submits wrong cause. Trained agent narrows to the culprit commit in ~15 steps. Reward curve overlay. This is the 50-second chunk that wins or loses the pitch. |
| 1:40–2:20 | **Design.** Three slides: observation (haystack), action (7 query types), reward (oracle grader formula). Keep it tight. |
| 2:20–2:50 | **Why it matters.** Frame as a benchmark for causal-attribution RL. Direct sub-prize fit — Scaler AI Labs. |
| 2:50–3:00 | **Close.** "We inverted the RL loop. The agent isn't acting on the world — it's figuring out what the world did." |

### Q&A kill-list (15 hardest anticipated questions)

To be fully authored by Day 6. Seed list:
1. How is this different from existing AIOps tools like Datadog or PagerDuty AI?
2. Isn't root cause identification just supervised classification?
3. Why RL instead of a rule-based search heuristic?
4. How do you prevent the agent from memorizing the 300 seeds?
5. What prevents reward hacking on the information-gain proxy?
6. How does the oracle handle ambiguous causes (multiple valid root causes)?
7. Why 4 services and not more realistic numbers?
8. How does your reward curve compare to a strong prompt-engineered baseline with no training?
9. What does the trained policy actually learn — do you have interpretability?
10. How would this scale to real production systems with millions of log lines?
11. Why is your chain-similarity metric edit-distance rather than graph-structural?
12. What happens if the agent submits a partially correct chain?
13. How did you pick the efficiency weight (0.2) vs correctness weight (0.5)?
14. Could you extend this to multi-agent (investigator + adversary)?
15. What is the one thing you would improve with another week?

Each team member should be able to answer any of these in 20 seconds cold. Practice by drilling each other.

### Delivery rules
- Utkarsh leads the pitch. Mohit handles the technical Q&A. Tanush drives the demo terminal.
- No reading from notes during the 3 minutes.
- Practice with a timer. Aim for 2:50 so there is no risk of being cut off.
- Eye contact with each judge at least once during the pitch.
- If Q&A goes wrong, do not get defensive. "Good question — here's our thinking" + honest answer, even if the honest answer is "we chose not to tackle that tradeoff given the time budget."

---

## 10. Open Decisions (to be made by Day 0 / 20 April)

These are real open calls, not placeholders. Document the decision and rationale in this section once made.

1. **Training model.** Qwen 2.5 7B vs Llama 3.2 3B vs a smaller option. Trade-off: 7B has more headroom but slower; 3B trains faster but may underperform on long contexts. Default lean: Qwen 2.5 7B with LoRA via Unsloth.
2. **Training algorithm.** SFT on heuristic trajectories → DPO → optionally PPO/GRPO. Or SFT-only if time is tight. Default lean: SFT + DPO, skip PPO unless Day 5 shows we have budget.
3. **Observation format.** Do we give the agent text blobs as raw strings, or pre-structured JSON with fields? Default lean: structured JSON for training stability; raw strings would force more capability but may hurt learning signal.
4. **Hypothesize penalty vs reward.** Current design: wrong hypothesize is -0.1, correct hypothesize is a confirmation signal with no score effect. Should correct hypothesize give positive reward? Risk: agent spams hypothesize. Default lean: keep current design, revisit after Day 4 evaluation.
5. **Stretch target — Mercor sub-prize.** Should we let the agent author the final postmortem narrative with length-scaled reward? Default lean: **no**, it adds complexity and risks diluting the focus. Revisit only if Days 1–4 go faster than planned.

---

## 11. What Carries Over From Round 1 (CloudWarEnv / CloudFinOpsEnv)

Direct reuse (copy-paste with minor edits):
- Service dependency DAG definition.
- Pydantic models for observation, action, reward, state.
- Chaos scheduler logic (inverted: scripts outages instead of injecting them live).
- Dockerfile, `openenv.yaml` skeleton.
- `inference.py` OpenAI-client scaffolding and `[START]/[STEP]/[END]` logging.
- HF Spaces deployment pipeline.

Needs rebuild:
- Task definitions (new tasks, new horizons, new grader).
- Action space (new action types — queries rather than operational actions).
- Reward function (new structure — information gain + terminal score).
- Oracle (new — did not exist in Round 1 form).
- Seed generator (new — must produce full frozen telemetry + ground truth).

Throw away:
- Round 1 budget/cost-pressure mechanics.
- Round 1 scaling/failover/reroute actions.
- Any real-time control logic.

---

## 12. Fallback Plan

If PostmortemEnv hits a hard wall in Days 1–2 (e.g., the seed generator proves intractable, or the hypothesis space blows up), the fallback is **APIDriftEnv** — a multi-step consumer workflow environment where API schemas drift mid-episode, targeted directly at the Patronus sub-prize. This sacrifices most Round 1 code reuse but enters an even lower-density lane with a near-uncontested sub-prize. Decision cutoff: end of Day 2. After that, we commit to finishing PostmortemEnv regardless.

---

## 13. Appendix A — Round 2 Themes (for reference)

| Theme | Fit for PostmortemEnv |
|---|---|
| 1 — Multi-Agent Interactions | Do not claim |
| 2 — Long-Horizon Planning & Instruction Following | Strong secondary |
| 3.1 — World Modeling, Professional | Strong primary |
| 3.2 — World Modeling, Personalized | Do not claim |
| 4 — Self-Improvement | Possible stretch |
| 5 — Wild Card | Do not claim (we fit Theme 3.1 cleanly) |

Sub-prize targets:
- **Primary: Scaler AI Labs** (Theme 3.1) — Multi-App RL Environment for Enterprise Workflows.
- Defensive fallback: Mercor (Theme 2) — if we add length-scaled postmortem narrative authoring.

---

## 14. Appendix B — One-Paragraph Version

For pasting into Slack, Discord, emails, and future Claude sessions:

> **PostmortemEnv** is an OpenEnv-native RL environment where an LLM agent investigates a cloud outage that has already happened. It receives a frozen telemetry snapshot (logs, traces, commits, configs, infra events across a 4-service DAG) and must identify the true root cause and causal chain within a bounded query budget. Actions are information-gathering operations (`query_logs`, `fetch_trace`, `diff_commit`, etc.) rather than operational controls. Grading is deterministic: exact-match on cause + edit distance on chain + efficiency bonus, clamped to [0.0, 1.0]. Three tasks escalate in difficulty from single-service recent-deploy blame (40 steps) to cross-service cascades (75 steps) to correlated root cause with concurrent infra events (120 steps). The core innovation is structural: it inverts the RL control loop into an epistemic loop where the agent's job is to figure out what the world did, not to act on it. Built by Team Three Musketeers for the Meta PyTorch OpenEnv Hackathon × Scaler SoT Grand Finale, targeting the Scaler AI Labs sub-prize under Theme 3.1 (World Modeling — Professional Tasks).

---

*Document version 1.0 — 19 April 2026. Revisions tracked in git.*
