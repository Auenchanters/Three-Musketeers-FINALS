# PostmortemEnv — Brutal Judging Criteria Rating

> Assumes all 12 identified issues are fixed. No mercy.

---

## 1. Environment Innovation — 40% weight

### Score: **8.2 / 10**

**What's genuinely strong:**
- "Epistemic RL" is a real conceptual differentiator. Every other team will build environments where agents *act on* systems. You built one where agents *investigate* systems. That framing is novel, defensible, and immediately understandable.
- Frozen telemetry is a clever constraint. It removes the need to simulate a live distributed system while still testing causal reasoning — the hard part of SRE work.
- The 7 action types (query, trace, diff, config, hypothesize, explain_chain, submit) map cleanly to real SRE workflows. This isn't a toy.
- Procedural generation from 5 failure templates with unlimited seeds is solid infrastructure. 10K validated seeds is a strong number.
- The "information gain" reward model (where queries reveal hidden facts) is a well-designed information-theoretic setup.

**Where it falls short of a 10:**
- The "4-service microservice architecture" is small. Real outages involve 20-200 services. Even the "hard" hand-crafted task only has 4 services (the procedural generator scales to 15-20, which is better, but the hand-crafted showcase tasks don't show this). A judge might think "this is a toy topology."
- The failure templates (connection_pool, oom_cascade, config_drift, failover_bug, memory_leak) are realistic but not *surprising*. A judge who's seen production outages will nod but won't be wowed. There's no "novel failure mode" — no race condition, no clock skew, no data corruption cascade, no multi-region failover.
- The environment is **single-turn per action**. There's no multi-agent collaboration (Theme #1), no super-long-horizon planning (Theme #2 mentions 300 scattered instructions), no self-improvement loop (Theme #4). You're squarely in Theme #3.1, which is fine, but you're not cross-pollinating themes in a way that would make judges say "this goes beyond what we asked."
- Partial observability is *stated* but weak in practice. The agent sees the full service graph, all commit hashes, all trace IDs, all config IDs upfront. The only hidden information is log contents, trace details, and commit diffs. A truly partially observable env would hide which services even exist, or reveal the graph incrementally.

**Net assessment:** Strong innovation for a hackathon. Top 15-20% of what judges will see. Not top 5% — that would require either a bigger topology, more exotic failure modes, or a genuinely novel interaction paradigm.

---

## 2. Storytelling — 30% weight

### Score: **6.5 / 10**

**This is your weakest area. Be honest with yourself.**

**What works:**
- The "every SRE knows the 2 AM drill" framing is immediately relatable. Good hook.
- The README is well-structured with clear tables, code examples, and architecture diagrams.
- The reward gap table (Random: 0.101 vs Oracle: 0.979) is compelling at a glance.
- `pitch_kit.md` and `blog_post.md` exist — many teams won't have these.

**What's brutally wrong:**
- **No video.** The YouTube link is a placeholder. This isn't a "nice to have" — it's a *minimum submission requirement*. Even if you fix this with a real video, the quality matters enormously. A 90-second screen recording of the live UI running Oracle vs Random side-by-side would be 10x more compelling than any markdown document. Judges will watch the video first, read the blog second, and look at code last.
- **The blog post reads like documentation, not storytelling.** Good storytelling for judges follows: *"Here's a real outage → here's why current tools fail → here's what our env teaches → here's proof it works."* Your blog reads more like *"Here's our architecture → here's our API → here's our reward formula."* That's a spec, not a story.
- **No concrete "before/after" walkthrough.** The spec (§19) explicitly says judges want to see: baseline attempt → reward output → trained attempt → measurable improvement → safeguards. You have aggregate numbers (Random: 0.101 vs Oracle: 0.979) but no *narrative* of a single investigation playing out step by step.
- **The live UI exists but isn't highlighted.** You built a real-time SSE-streaming investigation console with animated service graph — that's demo gold. But it's buried. This should be the centerpiece of your video and the first thing judges see.
- **No "wow" screenshot or GIF** in the README. The first visual impression matters. Your README is text-heavy.

**What you need to do to get this to 8+:**
1. Record a 90-second video: Open the live UI → run Oracle on the hard task → show the investigation unfolding live → cut to the reward curves → 3-sentence voiceover explaining why this matters.
2. Add a single animated GIF of the live UI to the top of the README.
3. Rewrite the blog intro to start with a real outage story, not with "PostmortemEnv is an epistemic RL environment."

---

## 3. Showing Improvement in Rewards — 20% weight

### Score: **7.0 / 10**

**What's solid:**
- Random vs Oracle gap of 0.878 is enormous and clearly documented.
- 4-panel training curve visualization (difficulty breakdown, epoch progression, loss/accuracy, agent comparison) is exactly what judges want to see.
- Per-step cumulative reward plots exist.
- The evaluation is on held-out seeds (separate from training) — good practice.

**Where it falls short:**
- **The SFT model barely beats random.** Looking at `run_real_training.py` and the evaluation logic, the SFT model trained on `facebook/opt-125m` is tiny and likely produces near-random scores. The training curve shows "lift over random" but if that lift is +0.02 on a 0.878 gap, that's a 2.3% improvement. Judges will see through this.
- **No GRPO improvement evidence.** The GRPO trainer exists in `train.py` but there's no evidence it was actually run and produced improvement. The spec (§53) says "start from a strong instruct or SFT checkpoint" — running GRPO on top of a near-random OPT-125M model won't produce useful gradients.
- **The "showing improvement" is mostly theoretical.** You've shown the *environment can differentiate* between random and oracle behavior (good). You haven't convincingly shown that *training on this environment makes a model better* (the actual criterion). The gap between "the environment has signal" and "the model learned from that signal" is what separates 7/10 from 9/10.
- **No frontier model benchmarks.** Running GPT-4o or Claude against your environment and showing their scores vs Random/Oracle would be powerful evidence that the environment tests real capability. You have the LLM agent infrastructure (`inference.py`, `web/agents.py`) but no published results.

**What you need to do to get this to 9+:**
1. Run the LLM agent (Claude Haiku, GPT-4o-mini) against all 3 tasks and publish the scores. Even if they score 0.3-0.5, that's interesting data: "frontier models partially solve this but don't match oracle, proving the env tests real reasoning."
2. If possible, run GRPO on a real instruct model (Llama 3.2 1B) even for a few steps and show any movement. Even +0.05 improvement with genuine RL would be the single most valuable artifact in your submission.
3. At minimum, make the SFT lift statistically significant. Run more eval seeds or use a larger base model.

---

## 4. Reward and Training Script/Pipeline Setup — 10% weight

### Score: **8.5 / 10**

**What's excellent:**
- The full pipeline exists end-to-end: `collect → sft → grpo → evaluate → plot`. This is exactly what the spec asks for.
- The GRPO reward function is environment-grounded: it literally resets the env, executes the completion as an action, and returns `obs.reward`. This is textbook RLVR. Judges will love this.
- Multiple reward components (info-gain, query-cost, step-cost, hypothesis penalty, terminal grading) with clear decomposition and breakdown logging.
- Deterministic grading with Jaccard node/edge similarity — no stochastic LLM-as-judge. This directly addresses §9 and §33 concerns.
- LoRA config is sensible (r=16, α=32 for SFT; r=8, α=16 for GRPO).
- The reward clamping to (0.01, 0.99) is defensive and correct.

**Minor issues:**
- The GRPO `reward_fn` only executes a **single action** per completion, not a full episode. This means the agent gets rewarded for one good query, not for a full investigation. A multi-turn GRPO loop would be stronger but is acknowledged as a gap in the Unsloth recipes (§59.6).
- The SFT data format uses Llama2-style `<s>[INST]` markers, which won't match OPT-125M's expectations. This is technically a bug in `run_real_training.py` but the main `train.py` is fine.
- No Unsloth integration despite the spec explicitly mentioning it. You use vanilla TRL + PEFT. This won't lose points but it won't earn bonus points either.

---

## Overall Score

| Criterion | Weight | Score | Weighted |
|---|---|---|---|
| Environment Innovation | 40% | 8.2 | 3.28 |
| Storytelling | 30% | 6.5 | 1.95 |
| Showing Improvement | 20% | 7.0 | 1.40 |
| Pipeline Setup | 10% | 8.5 | 0.85 |
| **Total** | **100%** | | **7.48 / 10** |

---

## Honest Placement Estimate

**If there are ~20 finalist teams:** You're in the **top 5-8 range**. Solidly above median. Probably not first place.

**Why not first place:**
1. Your storytelling/demo is your Achilles heel. Teams that build simpler environments but present them brilliantly will outscore you on the 30% storytelling weight.
2. You haven't shown convincing *model improvement* from training — only that the environment *could* train a model. The 20% "showing improvement" criterion wants *evidence*, not *potential*.
3. The environment is strong but not jaw-dropping. It's well-engineered, not paradigm-shifting.

**What would make this a first-place contender (in order of impact):**
1. 🎬 **A killer 90-second demo video** showing the live UI, investigation replay, and reward curves. This alone could move storytelling from 6.5 to 8.5, adding +0.6 to the total.
2. 📈 **One real frontier-model benchmark** (GPT-4o-mini scoring ~0.4 on your env) proving the env tests real intelligence. This adds credibility to the 40% innovation criterion.
3. 🏋️ **Any genuine RL improvement** — even a tiny GRPO lift on a real instruct model. This is the difference between "we built an env" and "we trained a model."
4. 🖼️ **One stunning visual** — animated SVG dependency graph, investigation replay GIF — at the top of the README.

**Bottom line:** You've built a genuinely good environment with strong engineering. The gap between where you are and first place is mostly *presentation and evidence*, not *technical quality*. Your codebase is probably the most mature in the competition. Your demo storytelling is probably average. Fix that asymmetry.
