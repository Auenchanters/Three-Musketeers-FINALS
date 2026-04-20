# PostmortemEnv — Round 2 Refinement Plan

**Goal:** Close the gap between the current build and the winning criteria for the Meta PyTorch OpenEnv Hackathon Round 2 (25–26 April 2026).

**Scope (locked with user):**
- Theme target: **Theme 3.1 (World Modeling — Professional)** + **Scaler AI Labs sub-prize** only. Do not claim Theme 2 or Theme 4. The pitch stays laser-focused exactly as the Foundation Doc §4 argues.
- Training: **Real SFT run on Colab T4** (not simulated). No GRPO this round — SFT-only is enough to ship a real curve; GRPO is stretch only if Day 5 has slack.

---

## 1. Context — Why This Refinement Exists

The repo already has the core environment, grader, scenarios, procedural generator, tests, live demo UI, and training scaffolding. The strategic framing (epistemic RL vs control RL) is locked in and defensible.

But an honest audit against the four rubric items and the three minimum requirements surfaces **five concrete gaps** that will cost us scores if we pitch today:

| # | Gap | File evidence | Rubric item hit |
|---|---|---|---|
| 1 | The "learning curve" plot is **simulated** with `math.exp` | [train.py:695-707](train.py#L695-L707) | 20% "Showing Improvement in Rewards" |
| 2 | Colab SFT/GRPO cells are **commented out** — no real training ever ran | [train_notebook.py:78-100](train_notebook.py#L78-L100) | Minimum requirement + 10% pipeline |
| 3 | GRPO reward function is a heuristic on action types, not real env reward | [train.py:385-413](train.py#L385-L413) | 10% reward coherence |
| 4 | No 2-minute video, no HuggingFace mini-blog | n/a | Explicit minimum requirement |
| 5 | README's "reward gap" table (0.90 vs 0.03) is claimed without a fresh evaluation artifact tied to it | [README.md:111-115](README.md#L111-L115) | Judge trust |

The fix is not a rewrite. It is tightening honesty, adding one real training run, and producing the two missing deliverables (video + blog). Everything else in the current codebase is already at or above the bar.

---

## 2. Outcome Targets (measurable)

By end of Day 6 (26 April):
- `training_data/reward_curves.png` shows a **real** SFT training curve from a Colab run — loss down, eval score up.
- `training_data/sft_eval_results.json` exists, contains evaluation of the SFT-trained model on 10 held-out seeds per difficulty, score strictly higher than the random baseline.
- `<=2:00` YouTube video exists, linked in README.
- HuggingFace mini-blog exists, linked in README.
- HF Space runs the container end-to-end inside 20 minutes on 2 vCPU / 8 GB.
- 3-minute pitch has been rehearsed ≥20 times (per Foundation Doc §7).

---

## 3. Work Items (in execution order)

### WI-1 — Fix the GRPO/SFT reward path so it uses the environment
*(small code change; prerequisite for real training)*

**Files to modify:**
- [train.py](train.py) — replace the heuristic `reward_fn` at lines 385-413 with a function that parses the completion as an Action, calls `env.step()`, and returns the resulting reward. Reuse the existing `_reset_with_scenario` helper at [train.py:41-74](train.py#L41-L74).
- [train.py:289-292](train.py#L289-L292) — `SFTConfig` currently passes `max_seq_length` and `dataset_text_field` as kwargs; verify compatibility with the installed TRL version on Colab (TRL 0.11+ changed the API). Pin the version in [requirements-train.txt](requirements-train.txt).

**Why:** Judges will open the training script. A reward function that doesn't call the environment is the single most obvious red flag in an RL submission.

### WI-2 — Delete the simulated learning curve
*(trivial; high credibility win)*

**Files to modify:**
- [train.py:688-725](train.py#L688-L725) — remove the `math.exp`-synthesized "learning curve" plot entirely. Keep only the Random-vs-Oracle comparison plot (which uses real data).
- [train_notebook.py:66-75](train_notebook.py#L66-L75) — remove the `display(Image("learning_curve.png"))` line for the simulated curve.

**Why:** A fabricated curve is worse than no curve. Replace it with the real SFT loss curve from WI-4.

### WI-3 — Uncomment and execute the SFT Colab cells
*(the real training run)*

**Files to modify:**
- [train_notebook.py:78-100](train_notebook.py#L78-L100) — uncomment SFT cell; set `model_name="meta-llama/Llama-3.2-1B-Instruct"` (per Foundation Doc §10 default), `epochs=3`, `batch_size=2`.
- Add a Cell 9: load the saved SFT adapter, run a fresh `evaluate_agents`-style loop but with the SFT model as the third agent alongside random and oracle. Save to `training_data/sft_eval_results.json`.
- Add a Cell 10: plot three curves on one figure — Random, SFT-trained, Oracle — using real mean scores per difficulty. Save as `training_data/reward_curves.png` (overwriting the old one).

**Execution:** Run the notebook on Colab T4 (Day 5 per Foundation Doc §7). Expected runtime 30-60 min for SFT + eval on ~60 demos (20 seeds × 3 difficulties).

**Fallback:** If SFT diverges or time runs out, ship Random-vs-Oracle only and honestly label the plot "Training headroom" rather than "Training progress." Do not fabricate.

### WI-4 — Write the 2-minute video script + record it
*(minimum requirement)*

**Deliverable:** `<=120 second` YouTube video, unlisted is fine. No file in repo — just a link in README.

**Script beats (aligned to Foundation Doc §9 pitch):**
- 0:00–0:15 — Cold open: "Production burned for 12 minutes. Four services cascaded. Someone has to write the RCA."
- 0:15–0:40 — Every other RL env teaches agents to control. This one teaches them to investigate.
- 0:40–1:20 — Demo cut: random agent flails on `seed_X_medium` → trained agent identifies the commit. Show the terminal UI from [static/index.html](static/index.html) live.
- 1:20–1:45 — Reward curve (the real one from WI-3).
- 1:45–2:00 — Close: "The first epistemic RL environment. PostmortemEnv."

**Who:** Tanush per Foundation Doc §1 recommended split.

### WI-5 — Write the HuggingFace mini-blog
*(minimum requirement)*

**Deliverable:** One blog post on the HF profile / Space page. ~500-800 words.

**Structure:**
1. The inversion (control RL vs epistemic RL) — 1 paragraph.
2. The task (what the agent sees, what it must produce) — 1 paragraph + service-graph diagram from README.
3. The reward (deterministic grader formula) — 1 code block from [engine/grader.py:114-179](engine/grader.py#L114-L179).
4. The numbers (real Random vs SFT vs Oracle from WI-3) — 1 table + 1 plot.
5. Reproduce instructions — 3 commands from README Quick Start.

**Who:** Utkarsh as lead, per Foundation Doc §1.

### WI-6 — Tighten README to match reality
*(small cleanup, high trust signal)*

**Files to modify:**
- [README.md:109-117](README.md#L109-L117) — replace the claimed "~0.95 oracle / ~0.03 random" table with the real numbers from `training_data/evaluation_results.json` once WI-3 completes. No rounding up.
- Add a "Video" and "Blog" link section under the header pointing to WI-4 and WI-5 artifacts.
- Add a one-line HF Space URL badge at the top.

### WI-7 — HF Space end-to-end validation
*(minimum requirement — environment must actually run)*

**Steps:**
- `docker build -t postmortemenv .` locally; confirm clean build.
- Push to the HF Space; confirm the container starts and `/health` returns 200.
- Run a scripted validation: random agent on 3 tasks, oracle agent on 3 tasks, all within 20-minute wall clock on the 2 vCPU / 8 GB tier. Reuse [test_oracle_e2e.py](test_oracle_e2e.py) as the harness.
- If the container exceeds 20 min, cut the hard task (per Foundation Doc §7 buffer protocol) rather than breaking the contract.

### WI-8 — Pitch rehearsal kit
*(per Foundation Doc §7 and §9)*

Not a code change, but part of the winning plan:
- Print the 15-question Q&A kill-list from Foundation Doc §9.
- 20+ rehearsals with a timer targeting 2:50.
- Assign Utkarsh = lead pitch, Mohit = technical Q&A, Tanush = demo driver (unchanged from Foundation Doc §1).

---

## 4. What We Are **Not** Doing

Explicit non-goals. Do not let scope creep eat training time.

- **Not claiming Theme 2 or Theme 4.** The single-theme pitch is stronger. Foundation Doc §4 is right.
- **Not building a 300-step long-horizon task.** Tempting but risks the core story.
- **Not claiming the ELO curriculum as "self-improvement"** in the pitch. It is a nice backend polish that judges see in the live UI; it is not the hook.
- **Not running GRPO this round.** SFT-only is sufficient for a real reward curve. GRPO is stretch-only if Day 5 finishes early.
- **Not rewriting the Colab-notebook format or the SFT prompt template.** The current format works.
- **Not adding more scenarios or more failure templates.** 3 hand-crafted + 5 templates is plenty.

---

## 5. Critical Files (for implementation reference)

| File | Role | Touch in this plan? |
|---|---|---|
| [engine/environment.py](engine/environment.py) | Core env loop, all 7 action handlers | No — solid |
| [engine/grader.py](engine/grader.py) | Deterministic scoring formula | No — solid |
| [engine/reward_calculator.py](engine/reward_calculator.py) | Dense per-step reward shaping | No — solid |
| [data/seed_generator.py](data/seed_generator.py) | 5-template procedural scenario generator | No — solid |
| [app.py](app.py) | FastAPI + OpenEnv create_app | No — solid |
| [openenv.yaml](openenv.yaml) | Task metadata | No — solid |
| [train.py](train.py) | Training pipeline | **Yes — WI-1, WI-2** |
| [train_notebook.py](train_notebook.py) | Colab notebook | **Yes — WI-3** |
| [README.md](README.md) | Front door | **Yes — WI-6** |
| [inference.py](inference.py) | Live LLM agent (OpenAI client contract) | No — solid |
| [web/runner.py](web/runner.py), [web/agents.py](web/agents.py), [static/](static/) | Live demo UI for pitch | No — solid |
| [tests/](tests/) | 881 lines of tests | No — passing |

---

## 6. Verification (how we know we're done)

End-to-end smoke test, run in this order, all green before pitch:

1. `python -m pytest tests/ -v` — all tests pass.
2. `python test_oracle_e2e.py` — oracle scores > 0.9 on all 3 tasks.
3. `python train.py collect --n-seeds 20` — produces `training_data/oracle_demos.jsonl`.
4. `python train.py evaluate --n-seeds 10` — produces real Random-vs-Oracle numbers, oracle_mean ≥ 0.7, random_mean ≤ 0.15 (per Foundation Doc §5.6 calibration targets).
5. Colab: SFT run completes, `sft_output/` exists, eval script produces `sft_eval_results.json` with SFT_mean > random_mean + 0.2.
6. `docker build -t postmortemenv .` succeeds. `docker run -p 7860:7860 postmortemenv`; curl `http://localhost:7860/health` → 200; open `http://localhost:7860/` → live console renders.
7. HF Space URL returns the live console; running a random-agent episode from the UI completes in under 90 seconds.
8. YouTube video is uploaded, ≤ 2:00, linked in README.
9. HF blog is published, linked in README.
10. README numbers match `training_data/evaluation_results.json` exactly.

---

## 7. Schedule Snapshot (aligned to Foundation Doc §7)

| Day | Date | Work items |
|---|---|---|
| Day 1 | 21 Apr | WI-1, WI-2 (code cleanup) |
| Day 2 | 22 Apr | WI-6 README tightening after fresh `evaluate` run; start WI-5 blog draft |
| Day 3 | 23 Apr | WI-7 HF Space deploy + 20-min validation; start WI-4 video script |
| Day 4 | 24 Apr | Buffer + WI-8 rehearsal prep |
| Day 5 | 25 Apr onsite | WI-3 real SFT on Colab (uses HF credits); finish WI-4 video recording |
| Day 6 | 26 Apr onsite | Finalize WI-5 blog, update plots from SFT run, 20× WI-8 rehearsals, pitch |

If Day 5 SFT diverges: ship SFT-only with a shorter curve and honest labeling. Do not skip the video or blog — those are minimum requirements.
