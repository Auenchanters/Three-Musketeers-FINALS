# PostmortemEnv — Spec Compliance Audit & Optimization Recommendations

> Full analysis of the `Revised_event_Context_Restructured.md` hackathon spec against the Three-Musketeers codebase.

---

## Part 1: Spec Compliance Checklist

### ✅ Fully Implemented Requirements

| # | Spec Requirement | Implementation | Files |
|---|---|---|---|
| §4 | Environment with `reset()`, `step()`, `state()`, `reward` | `PostmortemEnvironment` extends `Environment[Action, Observation, EnvironmentState]` | [environment.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/engine/environment.py) |
| §5 | OpenEnv-compliant environment (action/observation/state dataclasses, FastAPI wrapper) | Full Pydantic models + `create_app()` wiring | [models/](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/models/__init__.py), [app.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/app.py) |
| §6 | Easy → Medium → Hard curriculum | 3 hand-crafted tiers + procedural generation with difficulty-scaled telemetry volume | [seed_generator.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/data/seed_generator.py) |
| §7 | Multiple independent reward components | 5 reward signals: info-gain, query-cost, step-cost, hypothesis penalty, terminal score | [reward_calculator.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/engine/reward_calculator.py) |
| §8 | Anti-reward-hacking protections | Budget limits, wrong-hypothesis penalty, deterministic oracle grading (no LLM-as-judge) | [grader.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/engine/grader.py) |
| §10 | TRL + Unsloth/GRPO + OpenEnv stack | SFT via TRL `SFTTrainer` + LoRA, GRPO via `GRPOTrainer` with env-based reward_fn | [train.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/train.py) |
| §11 | GRPO/RLVR with verifiable rewards | GRPO reward function executes completions inside the env and returns `obs.reward` — verifiable, not learned | [train.py#L387-L433](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/train.py#L387-L433) |
| §13 | Deploy on HF Spaces | Dockerfile on port 7860, `openenv.yaml`, Docker health check, HF Space badge in README | [Dockerfile](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/Dockerfile), [openenv.yaml](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/openenv.yaml) |
| §14 | Stability before scale | 92 unit tests, oracle E2E validation, reward clamping (0.01–0.99) | [tests/](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/tests/), [test_oracle_e2e.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/test_oracle_e2e.py) |
| §16 | LoRA save path correctness | `trainer.save_model()` + `tokenizer.save_pretrained()` — uses TRL's proper save path | [train.py#L311-L312](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/train.py#L311-L312) |
| §19 | Sharp demo (baseline → trained → improvement) | Random vs Oracle evaluation, reward curves, 4-panel visualization | [run_real_training.py](file:///Users/tanushdeepak07/Downloads/Scaler%20new/Three-Musketeers-FINALS/run_real_training.py) |

### Minimum Requirements (§ Judging Criteria)

| Requirement | Status | Notes |
|---|---|---|
| Usage of OpenEnv (latest release) | ✅ | `openenv-core>=0.1.0` in requirements, extends `Environment` base class |
| Minimal training script (Unsloth or TRL) in Colab | ✅ | `train.py` (SFT + GRPO), `train_notebook.py`, `run_real_training.py` |
| Mini-blog on HuggingFace | ✅ | `blog_post.md` present |
| Mini-video on YouTube | ⚠️ | `video_script.md` exists but README links to `https://youtu.be/PLACEHOLDER` — **needs a real URL** |
| OpenEnv environment hosted on HF Spaces | ✅ | Dockerfile + README YAML metadata for HF Spaces |

### Judging Criteria Coverage

| Criterion | Weight | Status & Evidence |
|---|---|---|
| **Environment Innovation** | 40% | ✅ Strong — "Epistemic RL" is genuinely novel; frozen telemetry + causal reasoning isn't in any existing benchmark |
| **Storytelling** | 30% | ⚠️ Good but needs polish — `blog_post.md`, `pitch_kit.md`, `video_script.md` exist; video link is placeholder |
| **Showing Improvement in Rewards** | 20% | ✅ Real training curves exist in `training_data/reward_curves.png`, random-vs-oracle gap documented at 0.878 |
| **Reward & Training Pipeline Setup** | 10% | ✅ Full pipeline: `collect → sft → grpo → evaluate → plot` |

---

## Part 2: Gaps & Issues Found

### 🔴 Critical

1. **Video link is still a placeholder** (`https://youtu.be/PLACEHOLDER` in README line 29) — This is a minimum submission requirement.

2. **`_reset_with_scenario` duplicated across 3 files** — `train.py`, `run_real_training.py`, and `web/agents.py` all bypass `env.reset()` by directly setting private attributes. If `PostmortemEnvironment.__init__` ever changes shape, these will silently break. This is a maintenance bomb.

3. **`time_window` parameter on `query_logs` is accepted but never used** — `Action.time_window` is defined in the model but `_handle_query_logs` ignores it entirely. This is a dead feature that could confuse the LLM agent.

### 🟡 Medium

4. **No execution sandbox** — The spec (§8, §13) repeatedly warns about reward hacking through "editing timers, caching results, abusing globals, mutating protected state." The environment runs in the same Python process as the agent during GRPO training (`train.py`). While this is fine for this particular env (frozen read-only data), it's an anti-pattern worth noting for judges.

5. **`run_real_training.py` dataset prep is broken** — `prepare_dataset()` (line 148-166) reads `.jsonl` and tries to access `demo.get("observation")` and `demo.get("action")`, but the actual JSONL format from `collect_demonstrations()` stores data as `{"conversation": [...], "final_score": ...}` — the observation/action keys don't exist at the top level. This means `run_real_training.py` trains on garbage data if run after `collect`.

6. **Grader partial credit logic is unreachable** — In `grader.py:154-159`, partial credit for correlated causes is only checked when `is_correct == False`, but `check_cause_match` at line 95-98 *already* returns `True` for any contributing cause. So `cause_score` will be 1.0 (not 0.5 partial credit) when a contributing cause matches. These lines are dead code.

7. **No infra event handler** — There are `_handle_query_logs`, `_handle_fetch_trace`, `_handle_diff_commit`, `_handle_inspect_config`, but no action type for inspecting infrastructure events. The infra events are shown in the observation but the agent can't drill into them.

8. **`blog_post.md` and `pitch_kit.md` reference placeholder data** — Should be verified they contain final numbers.

### 🟢 Minor

9. **`seed_generator.py` uses `md5` (line 185)** — Cryptographically irrelevant here, but `hashlib.md5()` can throw in FIPS-compliant environments. Use `hashlib.md5(usedforsecurity=False)`.

10. **Duplicate import in `train.py`** — At line 355, `generate_oracle_solution` is imported but never used inside `run_grpo()`.

11. **`requirements.txt` includes `openai` and `numpy`** — These are only needed for inference/training, not for the env server itself. This bloats the Docker image.

12. **Train.py line 800+ cut off** — The file has 810 lines; the CLI `main()` ends with a print statement for quickstart that's never fully visible in the view.

---

## Part 3: Optimization Recommendations (Non-Breaking)

### A. Performance & Efficiency

| # | Optimization | Impact | Effort |
|---|---|---|---|
| A1 | **Add `__slots__`** to `PostmortemEnvironment` for ~15% faster attribute access | Speed | Low |
| A2 | **Pre-index logs by ID** in `reset()` → `self._log_id_index = {log["id"]: log for svc_logs in self._logs.values() for log in svc_logs}`. Currently each `_handle_query_logs` call does a linear scan for relevant-fact checking. | Speed | Low |
| A3 | **Cap observation serialization** — `_build_observation` builds complete commit/config lists every step. For hard tasks (18 commits, 22 configs), this serializes ~2KB of unchanged data per step. Cache the static portions at `reset()`. | Memory/Speed | Medium |
| A4 | **Lazy scenario loading** — `load_scenario` reads and parses JSON from disk every call. Add an LRU cache (e.g., `@functools.lru_cache`) for the 3 hand-crafted scenarios since they never change. | Speed | Low |
| A5 | **Batch GRPO reward computation** — The current `reward_fn` in `train.py` resets the env once per completion. For a batch of 4 completions on the same task_id, you could share the reset. | Training speed | Medium |

### B. Reward Design Improvements

| # | Optimization | Rationale |
|---|---|---|
| B1 | **Add exploration bonus for novel services queried** — Currently the agent gains the same 0.0 reward for querying irrelevant services repeatedly. A small bonus (e.g., +0.005) for first-time service queries would encourage broader investigation. | Prevents the agent from learning to spam `query_logs("data", "error")` repeatedly |
| B2 | **Time-decay reward for information gain** — Currently α=0.05 per relevant fact regardless of when it's discovered. Add a small bonus for discovering facts earlier (e.g., `α * (1 + 0.5 * remaining_budget / max_steps)`). | Encourages efficient investigation |
| B3 | **Add a "diminishing returns" penalty for repeated identical queries** — The env allows the same `query_logs(service, keyword)` to be called repeatedly without penalty beyond the flat cost. | Direct anti-hack measure |

### C. Environment Robustness

| # | Optimization | Description |
|---|---|---|
| C1 | **Add `INSPECT_INFRA` action type** — Infrastructure events are visible in the observation but the agent has no way to drill into them. Add a handler similar to `_handle_inspect_config` for infra events. | Completes the investigation toolkit |
| C2 | **Implement `time_window` filtering** — `_handle_query_logs` should respect `action.time_window` to filter logs by temporal proximity to the incident. | Makes the feature real instead of dead |
| C3 | **Consolidate `_reset_with_scenario` into environment** — Add `PostmortemEnvironment.reset_from_scenario(scenario: dict)` as a public method. Remove the 3 duplicated private-attribute-setting functions in train.py, run_real_training.py, and agents.py. | Eliminates maintenance risk |
| C4 | **Add episode-level logging** — Emit structured logs (JSON lines) for each episode with `task_id`, per-step actions/rewards, final score. This would be invaluable for the "monitor right things during training" requirement (§15). | Observability |
| C5 | **Fix the grader partial credit logic** — In `compute_final_score`, the partial credit branch (line 154-159) is unreachable. If a contributing cause should give partial credit (0.5), then `check_cause_match` should NOT return True for individual contributing causes — handle them differently. | Correctness |

### D. Demo & Storytelling

| # | Optimization | Impact |
|---|---|---|
| D1 | **Replace the YouTube placeholder URL** in README | Submission requirement |
| D2 | **Add a "before/after" section to the blog post** showing random agent transcript vs oracle transcript on the same task — this is exactly what§19 says judges find compelling | Judging score |
| D3 | **Highlight the 0.878 reward gap** more prominently in the README and pitch kit — this is your strongest metric | Judging score |
| D4 | **Add an SVG service dependency graph** that highlights the investigation path — this was discussed in past conversations as a "wow factor" feature | Demo impact |

### E. Code Quality

| # | Optimization | Description |
|---|---|---|
| E1 | **Split requirements.txt** — Create `requirements-server.txt` (fastapi, uvicorn, pydantic, openenv-core, websockets, httpx) and keep `requirements.txt` for everything. The Dockerfile should install only `requirements-server.txt`. | Smaller image |
| E2 | **Add type annotations** to all `Dict[str, Any]` returns in `seed_generator.py` with `TypedDict` or dataclasses — this would make the generated scenario shape explicit and catchable at dev time. | Maintainability |
| E3 | **Fix `run_real_training.py` dataset preparation** — The `prepare_dataset` function expects `observation` and `action` keys that don't exist in the JSONL. It should iterate through the `conversation` array to build instruction/response pairs. | Correctness |

---

## Part 4: Priority Action Items

> [!IMPORTANT]
> These are ordered by impact on the judging criteria:

### Must-Do Before Submission
1. **Fix the YouTube video placeholder** (README line 29) — minimum requirement
2. **Fix `run_real_training.py` dataset parsing** (line 148-166) — broken training artifact
3. **Consolidate `_reset_with_scenario`** into the environment class — prevents silent breakage

### Should-Do (High Impact)
4. Add `INSPECT_INFRA` action type to complete the investigation toolkit
5. Implement `time_window` filtering to make the parameter real
6. Fix the unreachable partial credit logic in the grader
7. Add LRU cache for scenario loading performance

### Nice-to-Have (Polish)
8. Add exploration/novelty bonus to prevent query spam
9. Implement repeated-query penalty
10. Split requirements for smaller Docker image
11. Cache static observation data at reset time
