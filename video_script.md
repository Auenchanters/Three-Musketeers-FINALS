# PostmortemEnv — 2-Minute Demo Video Script

**Total runtime target: ≤ 1:55 (5-second buffer)**
**Speaker: Tanush (demo driver)**

---

## Beat 1: Cold Open (0:00 – 0:15)

**[SCREEN: Black background, white text fading in]**

> *"Production burned for 12 minutes. Four services cascaded. Somebody has to write the RCA."*

**[Cut to: terminal with the PostmortemEnv console UI loading]**

> *"Every other RL environment teaches agents to control systems. This one teaches them to investigate what went wrong."*

---

## Beat 2: The Problem (0:15 – 0:40)

**[SCREEN: Side-by-side — "Control RL" vs "Epistemic RL" comparison table, animated]**

> *"Traditional RL envs give agents a lever. Pull it, the world changes, optimize the outcome. But real SRE work is forensics. The outage already happened. The telemetry is frozen. You don't control anything — you investigate."*

**[SCREEN: Service dependency graph diagram animates in]**

> *"PostmortemEnv gives the agent a frozen snapshot of a 4-service microservice architecture — logs, traces, commits, configs — and a bounded budget of queries to figure out the root cause and causal chain."*

---

## Beat 3: Live Demo (0:40 – 1:20)

**[SCREEN: PostmortemEnv Live Console UI — pick task1_recent_deploy, Random selected]**

> *"Here's our live investigation console. Let's start with a random agent."*

**[Click START — random agent runs. Watch the SVG service graph: nodes stay grey, score pops up ~0.09]**

> *"It queries random services, guesses wrong hypotheses… and scores 0.09. Watch the service graph — nothing lights up because it's not finding anything useful."*

**[Switch to Oracle, click START. Nodes light up white as queried, then at the end the causal chain animates: root cause pulses red, cascade edges flow with yellow dashes]**

> *"Now the oracle. Watch the graph come alive. Each service lights up as the agent investigates it. And when it submits — boom — the root cause pulses red, the cascade chain flows through the graph. Score: 0.96."*

**[SCREEN: Zoom in on score bar — 0.960]**

> *"That 0.88 gap is real. Deterministically graded — no LLM-as-judge. Pure function: 50% cause match, 30% chain similarity, 20% efficiency."*

---

## Beat 4: The Reward Curve (1:20 – 1:45)

**[SCREEN: `training_data/reward_curves.png` — the real Random vs SFT vs Oracle bar chart]**

> *"We SFT'd Llama 3.2 1B with LoRA on 60 oracle demonstrations — 20 seeds across 3 difficulty tiers. The green bars are the trained model. It beats random on every difficulty. That's real training signal from a real environment."*

**[SCREEN: The reward gap table from README]**

> *"A procedural generator built on 5 failure templates creates unlimited unique scenarios. Connection pool exhaustion, OOM cascades, AZ failover bugs, correlated multi-cause failures. Deploy it on HuggingFace Spaces and any LLM can train against it."*

---

## Beat 5: Close (1:45 – 1:55)

**[SCREEN: PostmortemEnv logo / title card with HF Space URL]**

> *"And it doesn't stop there — our ELO-based curriculum adapts difficulty as the agent improves. 10,000 validated unique scenarios, zero failures. Self-improving training at scale."*

> *"The first epistemic RL environment. PostmortemEnv."*

**[Fade to: Team name + HF Space link + hackathon theme badge]**

---

## Production Notes

- **Recording tool**: QuickTime screen recording + voiceover (or OBS)
- **Terminal font**: 16pt monospace, dark background, high contrast
- **Upload**: YouTube unlisted, link in README
- **Length check**: Rehearse 3× with a stopwatch before recording. Aim for 1:50 to leave margin.
- **Demo prep**: Run `uvicorn app:app --port 7860` and `python test_oracle_e2e.py` beforehand to make sure the env is warm and the demo won't stall.
