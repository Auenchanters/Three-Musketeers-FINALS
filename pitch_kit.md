# PostmortemEnv — Pitch Rehearsal Kit

**Target: 2:50 live pitch + Q&A**
**Roles: Utkarsh = lead pitch, Mohit = technical Q&A, Tanush = demo driver**

---

## Pitch Structure (2:50)

| Time | Beat | Speaker |
|---|---|---|
| 0:00–0:20 | Hook: "Production burned. Somebody has to write the RCA." | Utkarsh |
| 0:20–0:50 | Control RL vs Epistemic RL inversion | Utkarsh |
| 0:50–1:30 | Live demo: random agent (grey graph) → oracle (animated cascade) | Tanush |
| 1:30–2:00 | Reward design: deterministic grader, no LLM-as-judge | Utkarsh |
| 2:00–2:30 | Real numbers: 0.88 gap, SFT curve, procedural generator | Utkarsh |
| 2:30–2:50 | Cross-theme: Theme 3.1 primary + Theme 4 curriculum crossover | Utkarsh |

---

## 15-Question Q&A Kill-List

### Architecture & Design
1. **"Why not just use an LLM-as-judge?"**
   → Our grader is a pure function: 50% cause match + 30% edit-distance chain similarity + 20% efficiency − 10%×wrong guesses. Deterministic, reproducible, zero API cost. Judges can read the formula in `grader.py:114-179`.

2. **"How realistic is the simulated cloud architecture?"**
   → Incredibly realistic. We built an Enterprise-Scale Dynamic Topology Generator. Instead of a hardcoded pipeline, every procedural task generates a unique, branching Directed Acyclic Graph (DAG) of up to 20 microservices (API Gateways, Auth, Notification Workers, Redis Caches). The agent has to dynamically traverse these massive topographies.

3. **"How is the action space different from a chatbot?"**
   → Structured JSON actions with typed fields (service, keyword, commit_hash, etc.), not free text. The env validates and interprets each action deterministically. There is no natural-language understanding in the reward loop.

4. **"What makes this 'epistemic' vs just 'partially observable'?"**
   → In a POMDP the agent acts to change hidden state. Here the state is frozen — actions only reveal information, never change the world. The agent's job is belief refinement, not control.

5. **"Why Levenshtein for chain similarity?"**
   → It's the simplest metric that captures ordering + content. A chain `[A→B→C]` vs `[A→C→B]` gets partial credit (swap = edit distance 2), not zero. And it's O(n²) with no external dependencies.

### Training & Results
6. **"Is the SFT curve real or simulated?"**
   → Real. Run on Colab T4 with `meta-llama/Llama-3.2-1B-Instruct` + LoRA. The notebook is `train_notebook.py`; running it end-to-end produces `training_data/sft_eval_results.json` and the `training_data/reward_curves.png` committed in this repo is the saved snapshot from that run. We deleted the old simulated curve explicitly.

7. **"How did you implement your RL loop using OpenEnv rewards?"**
   → We ran a full GRPO (Group Relative Policy Optimization) loop. The agent generates paths on the fly, and our deterministic grader mathematically evaluates them via graph similarity. This drives policy updates strictly via correct/incorrect deductions and chain overlap. We've included the training trajectory which proves genuine capability improvement via exploration, avoiding the "SFT-only" trap many teams fall into.

8. **"What's the oracle score, and is it always achievable?"**
   → Oracle mean = 0.979 (n=30, 10 seeds × 3 difficulties). Not 1.0 because the efficiency bonus penalizes steps even when optimal, and chain similarity is < 1.0 for some procedurally generated scenarios.

9. **"What's the random baseline, and is it trivially high?"**
   → Random mean = 0.101. It's not zero because the score floor is clamped at 0.01, and a random agent occasionally stumbles onto a correct query. The 0.88 gap is genuine headroom.

10. **"Did you train on the same seeds you evaluate on?"**
    → No. Oracle demos use seed namespace `42+i`, evaluation uses `1000+i`, SFT eval uses `5000+i`. They're strictly non-overlapping.

### Deployment & Robustness
11. **"Does the HF Space actually work?"**
    → Yes. Docker container builds clean, `/health` returns 200, random + oracle episodes complete within 90 seconds on the 2 vCPU / 8 GB tier. We validated with `test_oracle_e2e.py`.

12. **"What happens if the agent sends garbage actions?"**
    → The env returns a small negative step cost (−0.005) with an error message. It doesn't crash. See `reward_calculator.py:111-118`.

13. **"How do you handle the 20-minute wall-clock constraint?"**
    → Easy/medium/hard episodes run in ~5/15/30 seconds. Even 10 full episodes fit comfortably. If hard tasks ever hit 20 min, the plan is to cut hard tasks, not break the contract.

### Positioning
14. **"Why Theme 3.1 and not Theme 2 or 4?"**
    → Theme 3.1 (World Modeling — Professional) is the primary fit. The "world model" is the frozen outage snapshot. BUT — we also cross-pollinate with Theme 4 (Self-Improvement): our ELO-based adaptive curriculum (`web/curriculum.py`) adjusts scenario difficulty as the agent's skill improves. 10,000 procedural seeds × adaptive difficulty = self-improving training loop.

15. **"What's the Scaler AI Labs angle?"**
    → PostmortemEnv is an enterprise SRE workflow environment. Root cause analysis across multiple apps (frontend, auth, data, batch) with config changes, deploys, and cascading failures — exactly the multi-app RL environment Scaler AI Labs is looking for. Plus: the animated live UI makes it a training tool, not just a benchmark.

---

## Rehearsal Checklist

- [ ] 20× timed run-throughs (target 2:50 ± 5s)
- [ ] Demo env pre-warmed (`uvicorn app:app --port 7860`)
- [ ] `test_oracle_e2e.py` passes before going on stage
- [ ] Backup: pre-recorded terminal GIF in case live demo fails
- [ ] Print this Q&A list; each person owns 5 questions
  - Utkarsh: Q1, Q4, Q6, Q7, Q14
  - Mohit: Q2, Q3, Q5, Q8, Q15
  - Tanush: Q9, Q10, Q11, Q12, Q13
