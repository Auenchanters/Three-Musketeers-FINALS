# PostmortemEnv — Project Review

> **Audience:** the team. **Status:** internal. Not shipped to judges.
>
> **Coverage:** (A) security & robustness audit, (B) theme-alignment + judging-criteria scorecard, (C) prioritised next-steps checklist to win the Scaler × Meta OpenEnv Hackathon (April 2026, Theme 3.1 — World Modeling, Professional).
>
> **Anchored to:** `analysis_results (1).md`, `analysis_results (2).md`, `brutal_rating (1).md`, and the `.claude/-External- Apr ‘26 OpenEnv Hackathon Themes & Judging Criteria.md` rubric. Reflects the state of the repo *after* Phase 1–3 of the finalisation plan landed (HF Space URL fixes, loss-curve wiring, grader partial-credit fix, lru_cache on `load_scenario`, `web/runner.py` per-episode summary JSONL, README rewrite).

---

## A. Security & robustness audit

The environment is publicly exposed via the HF Space (`https://huggingface.co/spaces/Auenchanters/postmortemenv`) with FastAPI under it. The findings below are ordered by severity. **None of these block submission**, but two are worth fixing before judges actively start scripting against the Space.

### A1 — HIGH — Path traversal in `data/generator.py:load_scenario`

**Where:** [`data/generator.py`](data/generator.py) `load_scenario(task_id)`. The `task_id` value is read straight from the public `POST /reset` body in [`app.py:68`](app.py).

**Issue:** When the `task_id` is not in the small allow-listed `SCENARIO_FILES` map, the function falls through to:

```python
gen_path = GENERATED_DIR / "scenarios" / f"{task_id}.json"
if gen_path.exists():
    with open(gen_path, "r", encoding="utf-8") as f:
        return json.load(f)
```

There is no validation of `task_id` here. A request body of `{"task_id": "../../../../etc/passwd"}` resolves to `data/generated/scenarios/../../../../etc/passwd.json`. Exploitability is *limited* because (a) the resolved path must end in `.json`, and (b) the file must exist, and (c) `json.load` will then try to parse it. So leaking `/etc/passwd` directly is unlikely. But:

- An attacker who can plant any `.json` file inside the container's writable directories (e.g. via a future write endpoint, log injection, or another bug) can read it.
- The attack surface includes any HF-Space-side `.json` files — model cards, config dumps, build manifests — that the runtime puts in predictable locations.
- This is still **CWE-22 path traversal** by the book, and a security-aware judge will flag it.

**Recommended fix:** strict regex validation before any disk read.

```python
import re

_TASK_ID_RE = re.compile(r"^(seed_\d+_(easy|medium|hard)|[a-z0-9_]+)$")

def load_scenario(task_id: str) -> Dict[str, Any]:
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"Invalid task_id format: {task_id!r}")
    # ... existing body
```

Apply the same regex inside `load_solution`. Add unit tests asserting that `"../../../etc/passwd"` and `"foo/bar"` raise `ValueError`.

**Effort:** 10 minutes. **Recommendation:** **fix before submission.**

### A2 — HIGH — Concurrency race on the global `_http_env` in `app.py`

**Where:** [`app.py:42`](app.py) instantiates a single `_http_env = PostmortemEnvironment()` and all of `/reset`, `/step`, `/state`, `/metadata` mutate it.

**Issue:** Two judges hitting the Space at the same time will silently corrupt each other's episode state. Concretely: judge A calls `/reset` with `task_id="task1_recent_deploy"`, judge B calls `/reset` with `task_id="task3_correlated_cause"`, judge A then calls `/step` and gets back an observation from judge B's `task3` scenario. There is no error — just wrong data, and a confused judge who walks away unsure if the env works.

The WebSocket `/ws` endpoint handed off to `openenv.core.env_server.create_app` is fine; it spins up a per-session env. The issue is specifically the legacy HTTP contract preserved for the smoke tests and `inference.py`.

**Recommended fix (in order of effort):**

1. *Quickest:* slap a single `asyncio.Lock` around the three handlers and document "single-tenant HTTP contract; use `/ws` for concurrency."
2. *Right answer:* maintain a `dict[str, PostmortemEnvironment]` keyed by the `episode_id` (already a parameter on `/reset`). Auto-evict envs older than N minutes.
3. *Cleanest:* delete the bespoke HTTP routes and rely entirely on OpenEnv's `/ws` session model. Update `inference.py` and `test_oracle_e2e.py` to talk WS.

**Effort:** option (1) is 5 minutes; option (2) is ~30 minutes. **Recommendation:** ship option (1) before submission, file an issue for option (2)/(3).

### A3 — MEDIUM — Unbounded LLM-vendor traffic relayed via `/api/runs`

**Where:** [`web/runner.py:99-104`](web/runner.py) (`StartRunBody.api_key`) and [`web/agents.py`](web/agents.py) (`LLMAgent.run`).

**Issue:** Anyone who scripts `POST /api/runs` with `agent="llm"` and a valid Anthropic / OpenAI key can use the Space as a proxy to anthropic.com / openai.com / a Llama provider. Headers and request body originate from the Space's IP, not the caller's. There's no rate limit and no per-IP cap. The `httpx.AsyncClient(timeout=60.0)` does cap each *individual* call, but a malicious script can still issue thousands of concurrent LLM runs to amplify cost on the API key (and, less seriously, to use the Space as an outbound proxy).

This is a documented BYO-key pattern, so it's defensible — the user supplies their own credentials, so cost falls on them, not us. But "defensible" is not "ignore."

**Recommended fix:**

- Add an in-process token-bucket rate limit on `/api/runs` (e.g. 3 requests/min per IP, max 5 concurrent). 20 lines with `slowapi`.
- Reject `api_key` lengths > 256 chars and non-printable bytes.
- Cap the per-step LLM response body at e.g. 64 KB before parsing JSON in `web/agents.py`.

**Effort:** ~30 minutes. **Recommendation:** medium-priority, fix during/after submission.

### A4 — MEDIUM — No CORS / no auth on the live UI endpoints

**Where:** [`web/runner.py`](web/runner.py) routes (`/api/tasks`, `/api/runs`, `/api/stream/{run_id}`, …).

**Issue:** Any browser tab on any origin can `fetch()` these endpoints. Acceptable for a hackathon demo Space — the entire point is "judges hit it from a logged-out browser." But once we move toward production we'll want a CORS allow-list and at minimum an `X-API-Key` header for the cost-incurring LLM agent route.

**Recommended fix:** none for the hackathon. File an issue for post-event hardening.

### A5 — LOW — Guessable run IDs on the public SSE endpoint

**Where:** [`web/runner.py:67`](web/runner.py) — `run.run_id = uuid.uuid4().hex[:12]` (48 bits).

**Issue:** 48 bits of randomness is theoretically guessable. The only thing leaked from cross-stream subscription is the public SSE event log of someone else's run, which already contains nothing sensitive (no API keys, no scenarios beyond the public ones). But "low impact" ≠ "no impact" if a judge thinks adversarially.

**Recommended fix:** `uuid.uuid4().hex` (full 128 bits). One-character change.

### A6 — LOW — No body-size cap on per-step LLM responses

**Where:** [`web/agents.py`](web/agents.py) `_call_anthropic` / `_call_openai` / `_call_*`. Reads the entire HTTP response into memory.

**Issue:** A misconfigured / malicious LLM endpoint could return an arbitrarily large body and exhaust container memory. `httpx`'s 60 s timeout doesn't help if the data streams fast.

**Recommended fix:** `response.read()` followed by an explicit `len(content) > MAX_BYTES` check, or `response.aiter_bytes(chunk_size=...)` with a running budget. ~10 minutes.

### A7 — LOW — Pickle / arbitrary-code unsafe paths

**Audit:** searched for `pickle.load`, `eval(`, `exec(`, `subprocess.run` with `shell=True`. The only call to `subprocess` outside `train.py` is in `web/curriculum.py` at startup to git-rev a state file (no shell). `train.py` shells out to `python scripts/plot_agent_comparison.py` with a list arg, no shell. No pickle in any user-supplied path. Clean.

### A8 — Already-mitigated items (logged for completeness)

- `hashlib.md5` is called with `usedforsecurity=False` in `data/seed_generator.py` (FIPS-safe).
- `test_oracle_e2e.py` does not hit the network.
- `requirements-server.txt` is stripped of training deps — the public Docker image doesn't ship `torch`/`transformers`.
- `_handle_inspect_infra` is implemented; `time_window` filtering is real.

---

## B. Theme alignment & judging-criteria scorecard

### B0 — Theme #3.1 (World Modeling — Professional) alignment

**Score: 9.0 / 10.**

The judging criteria for Theme #3.1 reads (verbatim):

> Here you will develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts to arrive at the desired outcome. Learning from these environments will enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows. The goal is to strengthen causal reasoning and persistent world models.

PostmortemEnv hits this rubric line by line:

| Theme #3.1 phrase | How PostmortemEnv answers it |
|---|---|
| *real interaction with tools, APIs, or dynamic systems* | 7 typed tools (`query_logs`, `fetch_trace`, `diff_commit`, `inspect_config`, `inspect_infra`, `hypothesize`, `explain_chain`) over a frozen but typed telemetry haystack |
| *real hard work instead of exploiting short-cuts* | The grader is deterministic, the cause IDs are not in the prompt, anti-gaming penalises wrong-hypothesis spam, and Random collapses to 0.10 |
| *maintain consistent internal state* | The agent must accumulate `known_facts`, track `service_graph` causality, and assemble a chain |
| *update beliefs based on outcomes* | `hypothesize` + `explain_chain` provide structured feedback the agent must update on |
| *orchestrate multi-step workflows* | Step budget is 40 / 75 / 120; Oracle uses 8 steps; Random burns 9–120 to floor |
| *strengthen causal reasoning and persistent world models* | The terminal grade is **explanation quality** — the score *is* the world model the agent built |

**Why not a 10:** the topology is 4 services. Real outages span 20–200 services. The procedural generator can scale up — but the hand-crafted showcase tasks don't.

### B1 — Environment Innovation (40 % weight)

**Score: 8.3 / 10. Weighted contribution: 3.32.**

**Strengths.** "Epistemic RL" is a genuinely fresh framing. Every other team will reward an agent for *changing* the world. We reward it for *explaining* a frozen one. The 7-action API maps cleanly to real SRE workflows. Procedural generation with 10 K validated unique seeds is solid infrastructure. The composable `Rubric` system uses OpenEnv's primitives the way they were meant to be used.

**Why not a 10.**

- Topology is 4 services (see B0).
- Failure templates are realistic but not surprising — connection-pool exhaustion, OOM cascades, config drift, AZ failover, memory leaks. No exotic mode (race, clock skew, multi-region failover, data corruption).
- Squarely Theme #3.1 — the ELO curriculum does cross-pollinate Theme #4 (Self-Improvement), but there's no Theme #1 (multi-agent) or Theme #2 (super-long-horizon) interaction.

**Bumped from 8.2 → 8.3 because** the Phase 2 grader fix made the partial-credit branch reachable. Hard-tier scoring is now genuinely 3-way (full-correct / contributing-only / wrong) instead of binary, which is a small but real win on rubric expressiveness.

### B2 — Storytelling & Presentation (30 % weight)

**Score: 7.8 / 10. Weighted contribution: 2.34.**

This was the brutal-rating Achilles heel at 6.5 / 10. The Phase 3 README rewrite addresses every named issue:

| `brutal_rating (1).md` complaint | Phase 3 fix |
|---|---|
| README starts with "PostmortemEnv is an epistemic RL environment" | Now opens with *"It is 02:47 AM. Your phone is buzzing."* |
| No "before/after" walkthrough | Oracle vs Random side-by-side trace is in §"Same task, two agents" |
| The live UI is buried | Live demo has its own H2 with explicit "click here, the Oracle button works without an API key" instruction |
| Blog reads like a spec | `blog_post.md` intro now leads with the 03:02 outage hook + Try-it-live callout |
| First impression is text-heavy | Hero plot (`agent_comparison.png`) is in the second screenful |

**Why not 8.5+ yet.**

- **No demo video.** This is the single biggest remaining lever and it is gated on the team recording it. The judging spec calls a < 2-minute video a *minimum requirement*. Until the YouTube link resolves, we are at risk of an auto-fail flag, not just a lower score.
- **No animated GIF / hero screenshot of the live UI in the README**. The textual oracle/random trace is good, but a 5-second GIF of the dependency graph animating during an investigation would be the wow moment.
- **Real-world-applications section is bullet-list-tight, not story-tight.** Three concrete use cases is fine, but a sentence-level case study ("Acme SRE team trained an internal copilot on 200 of their own real incidents replayed through PostmortemEnv") would make it a 9.

### B3 — Showing Improvement in Rewards (20 % weight)

**Score: 7.0 / 10. Weighted contribution: 1.40.**

**Strengths.** Random-vs-Oracle gap of **0.878** is real, reproducible (`python train.py evaluate --n-seeds 10`), and committed to the repo. 4-panel reward-curve visualisation exists. Held-out evaluation seeds (2000–2009) are disjoint from training seeds (1000–1019).

**Honest weaknesses.**

- **The SFT trained row in [`training_data/evaluation_results.json`](training_data/evaluation_results.json) is empty.** The previous `facebook/opt-125m` artifact was a 0.160-flat-across-epochs ghost; we deleted it rather than ship a phantom. The pipeline is wired (`python train.py full` runs collect → SFT → evaluate → plot end-to-end and now auto-emits `loss_curve.png` from `sft_metrics.json`), but until the row lands the criterion is "we *could* train" not "we trained."
- **No GRPO improvement evidence.** GRPO scaffold is in `train.py grpo` and reward-grounded. No published delta yet.
- **No frontier-model benchmarks.** [`scripts/run_frontier_benchmark.py`](scripts/run_frontier_benchmark.py) is wired and ready; needs `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` and a budget.

This score moves to 8.0+ the moment the SFT row lands, and to 8.5+ once one frontier model is benchmarked.

### B4 — Reward & Training Pipeline (10 % weight)

**Score: 8.7 / 10. Weighted contribution: 0.87.**

**Strengths.**

- Full pipeline `collect → sft → evaluate → plot → grpo` exists and is one-command (`python train.py full`).
- GRPO `reward_fn` calls into the live env (textbook RLVR, not surrogate-from-static-dataset).
- Composable `RubricSet` (cause / chain / efficiency / investigation / anti-gaming) — this is exactly the OpenEnv primitive the spec asks for.
- Deterministic Jaccard node+edge chain similarity. Zero LLM-as-judge.
- Score clamping to (0.01, 0.99) for OpenEnv validator compliance.
- LoRA r=16, α=32 on Qwen2.5-1.5B (open-weight, no gated form).
- Dense per-step rewards (`+0.05` info gain, `−0.01` query, `−0.10` wrong hypothesis) prevent sparse-reward exploration collapse.

**Bumped from 8.5 → 8.7 by Phase 2 fixes:**

- Partial-credit branch in `compute_final_score` is now reachable (was dead code per `analysis_results (2).md` C5). Hard-tier scoring is more discriminating.
- `@functools.lru_cache(maxsize=8)` on the file-read of the 3 hand-crafted scenarios cuts ~2 KB JSON reparse per `/reset`.
- Removed unused `generate_oracle_solution` import from `run_grpo` (audit item 10).
- `web/runner.py` now emits a one-line summary JSONL per episode for downstream observability.

**Why not 9+.**

- GRPO `reward_fn` evaluates a single completion as one action, not a full episode rollout. Multi-turn GRPO is acknowledged as a gap in TRL's recipe.
- No Unsloth integration despite the spec mentioning it. Vanilla TRL + PEFT works fine but won't earn bonus points.

### B5 — Overall weighted score

| Criterion | Weight | Score | Weighted |
|---|---|---|---|
| Environment Innovation | 40 % | **8.3** | **3.32** |
| Storytelling & Presentation | 30 % | **7.8** | **2.34** |
| Showing Improvement in Rewards | 20 % | **7.0** | **1.40** |
| Reward & Training Pipeline | 10 % | **8.7** | **0.87** |
| **Total** | **100 %** | | **7.93 / 10** |

**Delta from `brutal_rating (1).md` baseline (7.48 / 10): +0.45.**

That delta is essentially "we did the README work and the analysis-2 fixes." The Phase 1–3 work is done; everything left to capture is in §C below.

### B6 — Honest placement estimate

If there are ~20 finalist teams, **we are sitting in the top 5–8**. With the demo video + a real SFT row + one frontier-model benchmark, we move to **top 3** confidently. Hitting first place requires either (a) a single jaw-dropping demo moment in the video, or (b) a published GRPO improvement delta. Both are within reach.

---

## C. Concrete checklist of next steps to win

Ordered by **impact-per-hour**. The "must-do" items move us from "credible submission" to "credible top-3 submission" and are roughly 2–4 hours of additional team work.

### C-MUST — required before submission

1. **🎬 Record the 2-minute demo video.** This is a *minimum submission requirement* per the judging spec, not a nice-to-have. Suggested shot list:
   1. 5 s — cold-open with `02:47 AM` text on black, then cut to red dashboards.
   2. 15 s — narrate "you don't get to fix this, you have to explain it."
   3. 30 s — open the live HF Space. Click *Run Oracle* on the hard task. Show the investigation streaming.
   4. 30 s — switch to *Run Random*. Show the same task collapsing to 0.15.
   5. 20 s — cut to `agent_comparison.png` headline plot. "Random 0.10 vs Oracle 0.98 — that gap is the training headroom."
   6. 15 s — cut to `python train.py full` running, then `loss_curve.png`. "Same env trains a 1.5 B model end-to-end in one command on a T4."
   7. 5 s — closing card with the Space URL.
   - Upload as **unlisted YouTube**, paste the URL into the README's `Demo video` row.
   - Owner: Tanush. ETA: 60–90 minutes.

2. **💤 Wake the HF Space and add a healthcheck.** `huggingface-cli` ping the Space within 24 hours of submission so it's not in *Sleeping* state when judges click. Add a tiny `tools/keep_warm.sh` that hits `/health` every 6 hours via a free uptime monitor (UptimeRobot has a free tier). **Owner: anyone. ETA: 15 minutes.**

3. **🔒 Apply A1 + A2 security fixes.** Both are < 30 minutes total and remove the only HIGH findings:
   - Add the `_TASK_ID_RE` regex check on `load_scenario` / `load_solution`.
   - Wrap the `_http_env` HTTP handlers in an `asyncio.Lock` (option-1 from §A2).
   - Add 3 unit tests: traversal rejected, two concurrent resets serialise, lock is released on exception.
   - **Owner: Mohit. ETA: 30 minutes.**

4. **🏋️ Run `python train.py full` on a real T4.** This produces:
   - The trained row in `evaluation_results.json` (the empty SFT row goes away).
   - `loss_curve.png` (auto-generated by the new `plot_loss_curve` helper from `sft_metrics.json`).
   - `reward_curves.png` and `agent_comparison.png` refreshed with the *trained* bar present.
   - Commit all four artifacts to `training_data/`.
   - If trained mean fails to clear `random + 0.02`, the script will print a loud warning — re-run with a different `--epochs` or a slightly larger model (`Qwen/Qwen2.5-3B-Instruct`).
   - **Owner: Utkarsh on the rented GPU. ETA: 30–60 minutes wall-clock.**

### C-HIGH — high-leverage if there's time before the deadline

5. **🧪 Run one frontier-model benchmark.** `export ANTHROPIC_API_KEY=...; python scripts/run_frontier_benchmark.py --models claude-haiku-4-5 --n-seeds 5`. Even one model on 5 seeds is enough to populate `frontier_results.json` and flip the README's "produced by …" line into a real link. Cost: ~$2–5 of API spend. **Owner: any team-mate with a key. ETA: 20 minutes.**

6. **🖼️ Add an animated GIF of the live UI to the top of the README.** Use [Kap](https://getkap.co/) on macOS or [ScreenToGif](https://www.screentogif.com/) on Windows. 5–8 seconds is enough — show the dependency graph animating as Oracle steps. Compress to < 2 MB. Commit as `docs/demo.gif` and embed under the H1 in the README. (We currently do not reference this file, so adding it is purely upside.) **Owner: anyone. ETA: 30 minutes.**

7. **📊 Add a "first time visiting?" banner pointing at the live demo.** Top of README, above the `02:47 AM` hook:
   `> 👉 **In a hurry?** Click [here](https://huggingface.co/spaces/Auenchanters/postmortemenv), choose a task, hit *Run Oracle*. No setup. No API key. ~30 seconds to "got it."`
   This one line is the difference between a judge bouncing and a judge clicking. **Owner: anyone. ETA: 1 minute.**

8. **🤖 Run a smoke GRPO for 200 steps.** Even +0.03 of trained-vs-SFT-baseline movement is the single most valuable artifact in the submission per `brutal_rating (1).md`. Start from the SFT checkpoint produced in C-MUST #4. Commit `training_data/grpo_eval_results.json` with the comparison. **Owner: Utkarsh. ETA: 90 minutes wall-clock + 15 minutes wiring.**

### C-NICE — nice-to-have polish

9. **📐 Increase scenario topology in one hand-crafted task** to 10–15 services. Brutal-rating's #1 innovation complaint was "this is a toy topology." A single scaled-up `task4_multi_region_failover` would lift Innovation from 8.3 to 8.6. Procedural generator already supports it; just author one fixture file + solution. **Owner: anyone. ETA: 60 minutes.**

10. **🧪 Add tests for the new helpers** added during Phase 1–2:
    - `plot_loss_curve` no-op when `sft_metrics.json` is missing
    - `Grader.is_contributing_cause_match` (already added — see `tests/test_grader.py`)
    - `web/runner.py` summary JSONL is written exactly once per run.
    - **Owner: anyone. ETA: 30 minutes.**

11. **🔐 A3 + A6 LLM rate-limit + body-size cap.** Post-event hardening. Not blocking submission. **Owner: future-us.**

12. **🧹 Remove the `agent_transcripts` references and any stray analysis docs from the repo's public surface.** Move `analysis_results (1).md`, `analysis_results (2).md`, `brutal_rating (1).md`, and this file (`PROJECT_REVIEW.md`) into a `.dev/` directory excluded from the HF Space upload (or `.gitattributes` mark them as `export-ignore`). Judges shouldn't see our internal scoring documents. **Owner: anyone. ETA: 5 minutes.**

### C-DO-NOT-DO

- Do **not** fabricate a `loss_curve.png` or a frontier-model results JSON. The pipeline is honest; ship honest artifacts.
- Do **not** force-push to `main` close to the deadline. HF Space rebuilds take ~3 minutes; a broken push 10 minutes before submission is the worst-case failure mode.
- Do **not** add a CORS allow-list, auth, or rate-limit *during* the submission window. We've verified the BYO-key model is stable; introducing new middleware now risks breaking judges' ability to use the live demo.
- Do **not** rename `Auenchanters/postmortemenv` after submission. The submitted URL is the only one judges will use.

---

## D. Summary

**Where we are:** 7.93 / 10 weighted, top-5-to-8 of ~20 finalists. Codebase is the most mature in the competition (`brutal_rating (1).md`'s call). Phase 1–3 of the finalisation plan landed cleanly: HF Space URL is correct everywhere; loss-curve plotting is wired and will produce a real artifact on the first GPU run; the grader partial-credit branch is no longer dead code; `data.generator.load_scenario` caches file reads; `web/runner.py` emits a one-line per-episode JSONL summary; the README is now story-led with a hook, motivation, real-world apps, differentiation, live demo, benchmarks, and a why-this-wins section; the blog post intro is aligned and links the live Space.

**What stands between us and first place:** a 90-second video, a real SFT row in the eval JSON, one frontier-model benchmark, and the two HIGH security fixes. ~3–4 hours of focused team work. Everything is wired end-to-end; we just have to press the buttons.

— *Three Musketeers · April 2026*
