"""
Baseline inference script for PostmortemEnv.

Uses an OpenAI-compatible client. Emits [START], [STEP], [END] structured
logs to stdout. All other output goes to stderr.

Env vars: API_BASE_URL, HF_TOKEN, MODEL_NAME, ENV_URL
"""

import os
import sys
import re
import json
import time
import traceback
import httpx
from typing import Any
from openai import OpenAI
from client import PostmortemClient
from models.action import Action

INFERENCE_DEBUG = os.environ.get("INFERENCE_DEBUG", "0") == "1"

def _debug(msg: str):
    """Print debug info to stderr (never pollutes structured stdout)."""
    if INFERENCE_DEBUG:
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

def _info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)

def _warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)

def _error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")


def _require_hf_token() -> str:
    """Enforce ``HF_TOKEN`` only at run time so this module stays importable.

    Other scripts (e.g. ``scripts/run_frontier_benchmark.py``) reuse the
    prompt and parser helpers without needing the HF Inference Router.
    """
    if not HF_TOKEN:
        _error(
            "HF_TOKEN is not set. Export HF_TOKEN=<your-hf-token> to run the LLM "
            "agent against the HF Inference Router. To verify the environment "
            "without any key, run `python test_oracle_e2e.py`, or open the live UI "
            "(`uvicorn app:app --port 7860`) and pick Oracle or Random."
        )
        sys.exit(2)
    return HF_TOKEN


MAX_STEPS = 50
TASKS = ["task1_recent_deploy", "task2_cascade_chain", "task3_correlated_cause"]
TEMPERATURE = 0.2
MAX_TOKENS = 512
RETRY_DELAYS = [3, 6, 12]  # seconds between retries

SYSTEM_PROMPT = """You are an expert SRE investigator conducting a postmortem analysis of a cloud outage.

IMPORTANT: You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no text before or after. Just the JSON.

You are investigating a FROZEN outage — it already happened. Your job is to find the root cause and the causal chain of how the failure propagated.

Available actions:
- query_logs(service, keyword, time_window?): Search logs for a service by keyword. Optional time_window like "last_5m" or "during_incident" filters by time.
- fetch_trace(trace_id): Get a specific distributed trace
- diff_commit(commit_hash): See what changed in a specific commit
- inspect_config(config_id): See a specific config change
- inspect_infra(event_id): See details of an infrastructure event (DNS update, AZ failover, etc.)
- hypothesize(cause_entity_id): Test if a candidate is the root cause (non-terminal, gives feedback)
- explain_chain(chain): Test if your causal chain is correct (non-terminal, gives feedback)
- submit(final_cause, final_chain): Submit your final answer and end the investigation

INVESTIGATION STRATEGY:
1. Start by querying logs of the service with the HIGHEST error rate during the incident
2. Look for ERROR and CRITICAL log entries — they often point to the root cause
3. Check recent deployments (commits) near the incident start time — deployments are the #1 cause of outages
4. Use diff_commit to see what changed in suspicious commits
5. Trace the cascade: which service failed FIRST? Follow the dependency chain.
6. Use hypothesize to test your theory before submitting
7. Use explain_chain to verify your causal chain before submitting
8. When confident, use submit with your final_cause and final_chain

RULES:
- Each query costs you budget. Be efficient — don't repeat queries.
- Wrong hypotheses incur a penalty. Don't guess blindly.
- The causal chain should be ordered from root cause → downstream effects.
- For the chain, each entry needs {service, effect} describing what happened at that service.

RESPOND WITH ONLY JSON. Examples:
{"action_type": "query_logs", "service": "data", "keyword": "error"}
{"action_type": "query_logs", "service": "data", "keyword": "timeout", "time_window": "last_5m"}
{"action_type": "diff_commit", "commit_hash": "commit-abc123"}
{"action_type": "fetch_trace", "trace_id": "trace-001"}
{"action_type": "inspect_config", "config_id": "cfg-001"}
{"action_type": "inspect_infra", "event_id": "infra-001"}
{"action_type": "hypothesize", "cause_entity_id": "commit-abc123"}
{"action_type": "explain_chain", "chain": [{"service": "data", "effect": "connection_pool_exhaustion"}, {"service": "auth", "effect": "upstream_timeout"}]}
{"action_type": "submit", "final_cause": "commit-abc123", "final_chain": [{"service": "data", "effect": "pool_exhaustion"}, {"service": "auth", "effect": "timeout"}, {"service": "frontend", "effect": "5xx_errors"}]}
"""


def _safe_reward(r: Any) -> float:
    """Clamp a value to strictly inside (0.01, 0.99) for validator compliance."""
    try:
        val = float(r)
        if val != val:  # NaN check
            return 0.01
        return round(min(max(val, 0.01), 0.99), 4)
    except (ValueError, TypeError):
        return 0.01


def log_start(task, env, model):
    """Emit [START] line per spec."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    """Emit [STEP] line per spec. Reward is clamped to (0, 1)."""
    safe_r = _safe_reward(reward)
    action_str = json.dumps(action, separators=(",", ":"))
    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action_str} reward={safe_r:.4f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success, steps, rewards, score):
    """Emit [END] line per spec."""
    success_str = "true" if success else "false"
    safe_rewards = [_safe_reward(r) for r in rewards]
    rewards_str = ",".join(f"{r:.4f}" for r in safe_rewards)
    safe_score = _safe_reward(score)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str} score={safe_score:.4f}",
        flush=True,
    )


def format_observation(obs: dict) -> str:
    """Convert observation dict to LLM-friendly text."""
    remaining = obs.get("remaining_budget", 0)
    lines = [
        f"=== INCIDENT INVESTIGATION ===",
        f"Task: {obs.get('task_description', '')}",
        f"Step: {obs.get('step_number', 0)}/{obs.get('max_steps', 40)} ({remaining} steps remaining)",
        f"Incident Window: {obs.get('incident_window', {})}",
    ]

    # Urgency warnings
    if remaining <= 2:
        lines.append("*** URGENT: You must SUBMIT NOW or lose all progress! ***")
    elif remaining <= 5:
        lines.append("*** WARNING: Running low on budget. Submit soon. ***")

    # Service graph
    lines.append("\n--- SERVICE DEPENDENCY GRAPH ---")
    graph = obs.get("service_graph", {})
    for svc, deps in graph.items():
        dep_str = " → ".join(deps) if deps else "(no deps)"
        lines.append(f"  {svc} depends on: {dep_str}")

    # Service status
    lines.append("\n--- SERVICE STATUS DURING INCIDENT ---")
    for s in obs.get("services", []):
        if isinstance(s, dict):
            lines.append(
                f"  {s.get('name', '?')}: status={s.get('status', '?')} | "
                f"error_rate={s.get('error_rate_during_incident', 0):.1f}% | "
                f"deploys_24h={s.get('recent_deploy_count', 0)}"
            )

    # Recent commits
    commits = obs.get("available_commits", [])
    if commits:
        lines.append(f"\n--- RECENT COMMITS ({len(commits)} in last 24h) ---")
        for c in commits:
            lines.append(f"  {c.get('hash', '?')} | {c.get('service', '?')} | {c.get('timestamp', '?')} | {c.get('message', '')[:80]}")

    # Config changes
    configs = obs.get("available_config_changes", [])
    if configs:
        lines.append(f"\n--- CONFIG CHANGES ({len(configs)}) ---")
        for c in configs:
            lines.append(f"  {c.get('config_id', '?')} | {c.get('service', '?')} | {c.get('timestamp', '?')} | {c.get('description', '')[:80]}")

    # Available traces
    traces = obs.get("available_trace_ids", [])
    if traces:
        lines.append(f"\n--- AVAILABLE TRACES ({len(traces)}) ---")
        lines.append(f"  {', '.join(traces[:15])}")
        if len(traces) > 15:
            lines.append(f"  ... and {len(traces) - 15} more")

    # Infra events
    infra = obs.get("available_infra_events", [])
    if infra:
        lines.append(f"\n--- INFRASTRUCTURE EVENTS ({len(infra)}) ---")
        for e in infra:
            lines.append(f"  {e.get('event_id', '?')} | {e.get('timestamp', '?')} | {e.get('description', '')[:100]}")

    # Known facts
    facts = obs.get("known_facts", [])
    if facts:
        lines.append(f"\n--- CONFIRMED FACTS ({len(facts)}) ---")
        for f in facts:
            lines.append(f"  ✓ {f}")

    # Last query result
    qr = obs.get("query_result", "")
    if qr:
        lines.append(f"\n--- LAST QUERY RESULT ---")
        lines.append(qr)

    # Message
    msg = obs.get("message", "")
    if msg:
        lines.append(f"\n--- ENV MESSAGE ---")
        lines.append(msg)

    # Investigation stats
    lines.append(f"\n--- STATS ---")
    lines.append(f"  Hypotheses submitted: {obs.get('hypotheses_submitted', 0)}")
    lines.append(f"  Wrong hypotheses: {obs.get('wrong_hypotheses', 0)}")

    return "\n".join(lines)


def parse_action(text: str) -> dict:
    """Parse LLM response text into action dict."""
    text = text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: query_logs on the first service so we don't waste a step
    _warn(f"Could not parse LLM response: {text[:200]}")
    return {"action_type": "query_logs", "service": "data", "keyword": "error", "reason": "Parse failure fallback"}


def get_agent_action(client: OpenAI, observation_text: str, history: list, parse_failures: int = 0) -> str:
    """Ask the LLM to decide the next action, with retry on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_text},
    ]
    # Include recent history for context (max 5)
    deduped_history = []
    seen_actions = set()
    for h in reversed(history):
        if h["action"] not in seen_actions:
            seen_actions.add(h["action"])
            deduped_history.insert(0, h)
        if len(deduped_history) >= 5:
            break

    for h in deduped_history:
        messages.append({"role": "assistant", "content": h["action"]})
        messages.append({"role": "user", "content": h["result"]})

    if parse_failures > 0:
        messages.append({"role": "user", "content": (
            "REMINDER: Respond with ONLY a JSON object, no other text. "
            'Example: {"action_type": "query_logs", "service": "data", "keyword": "error"}'
        )})

    last_err = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay > 0:
            _warn(f"Retry {attempt}/{len(RETRY_DELAYS)} after {delay}s...")
            time.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            _debug(f"Raw LLM response: {text[:500]}")
            # Strip <think>...</think> tags from thinking models
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                _debug(f"After think-strip: {text[:300]}")
            return text
        except Exception as e:
            last_err = e
            _error(f"LLM call failed (attempt {attempt+1}): {e}")

    _error(f"All retries exhausted. Last error: {last_err}")
    return '{"action_type": "query_logs", "service": "data", "keyword": "error", "reason": "LLM error fallback"}'


def obs_to_dict(obs) -> dict:
    """Convert an Observation object (or dict) to a plain dict."""
    if isinstance(obs, dict):
        return obs
    d = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
    if "services" in d:
        d["services"] = [
            s.model_dump() if hasattr(s, "model_dump") else s
            for s in d["services"]
        ]
    return d


def run_task(llm_client: OpenAI, env_url: str, task_name: str) -> float:
    """Run a single task using the OpenEnv WebSocket client and return the score."""
    history = []
    rewards = []
    parse_failures = 0
    step_num = 0
    score = 0.50
    success = False

    log_start(task=task_name, env="PostmortemEnv", model=MODEL_NAME)

    try:
        sync_client = PostmortemClient(base_url=env_url)
        obs_raw = sync_client.reset(task_id=task_name)
        obs = obs_to_dict(obs_raw)

        for step_num in range(1, MAX_STEPS + 1):
            obs_text = format_observation(obs)

            # Safety net: force submit on last step
            task_max_steps = obs.get("max_steps", MAX_STEPS)
            remaining = obs.get("remaining_budget", task_max_steps - obs.get("step_number", 0))

            if remaining <= 1:
                _info(f"Auto-submitting on step {step_num} (last step safety net)")
                action_text = '{"action_type": "submit", "final_cause": "unknown", "final_chain": [], "reason": "Budget exhausted"}'
                action_dict = parse_action(action_text)
            else:
                action_text = get_agent_action(llm_client, obs_text, history, parse_failures)
                action_dict = parse_action(action_text)

            # Track parse failures
            if action_dict.get("reason", "").startswith("Parse failure"):
                parse_failures += 1
            else:
                parse_failures = 0

            if parse_failures >= 3:
                _warn("3 consecutive parse failures — submitting to save progress.")
                action_dict = {"action_type": "submit", "final_cause": "unknown", "final_chain": [], "reason": "Parse failure bailout"}
                parse_failures = 0

            # Build typed Action
            action = Action(
                action_type=action_dict.get("action_type", "query_logs"),
                service=action_dict.get("service"),
                keyword=action_dict.get("keyword"),
                time_window=action_dict.get("time_window"),
                trace_id=action_dict.get("trace_id"),
                commit_hash=action_dict.get("commit_hash"),
                config_id=action_dict.get("config_id"),
                cause_entity_id=action_dict.get("cause_entity_id"),
                chain=action_dict.get("chain"),
                final_cause=action_dict.get("final_cause"),
                final_chain=action_dict.get("final_chain"),
                reason=action_dict.get("reason"),
            )

            result = sync_client.step(action)
            obs = obs_to_dict(result.observation)
            reward = result.reward if isinstance(result.reward, (int, float)) else 0.0
            done = result.done
            rewards.append(reward)

            obs_msg = obs.get("message", "")
            step_error = obs_msg if ("INCORRECT" in obs_msg or "not found" in obs_msg.lower()) else None

            log_step(step=step_num, action=action_dict, reward=reward, done=done, error=step_error)

            history.append({
                "action": action_text,
                "result": obs.get("query_result", obs_msg),
            })

            if done:
                break

        # Get final state for scoring
        try:
            final_state = sync_client.get_state()
            state_dict = final_state.model_dump() if hasattr(final_state, "model_dump") else vars(final_state)
            _debug(f"Final state keys: {list(state_dict.keys())}")
        except Exception as e:
            _error(f"Failed to get final state: {e}")
            state_dict = {}

        # Use server-computed score if available
        final_score = state_dict.get("final_score")
        if final_score is not None:
            score = _safe_reward(final_score)
        else:
            score = _safe_reward(sum(rewards) / len(rewards) if rewards else 0.5)

        success = score >= 0.5

    except Exception as exc:
        _error(f"run_task crashed: {exc}")
        traceback.print_exc(file=sys.stderr)
        score = 0.01
        success = False
        step_num = max(step_num, 1)

    finally:
        log_end(
            success=success,
            steps=max(step_num, 1),
            rewards=rewards if rewards else [0.50],
            score=score,
        )

    return score


def main():
    """Run all tasks and report scores."""
    token = _require_hf_token()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=token)

    # Health check
    try:
        health = httpx.get(f"{ENV_URL}/health", timeout=10.0)
        _info(f"Environment healthy: {health.json()}")
    except Exception as e:
        _error(f"Cannot connect to environment at {ENV_URL}: {e}")
        sys.exit(1)

    scores = {}
    for task in TASKS:
        score = run_task(llm_client, ENV_URL, task)
        score = _safe_reward(score)
        scores[task] = score
        _info(f"Task {task}: {score:.4f}")

    # Emit structured per-task RESULT lines
    for task, sc in scores.items():
        print(f"[RESULT] task={task} score={sc:.4f}", flush=True)

    # Summary
    clamped_scores = [sc for sc in scores.values()]
    avg = sum(clamped_scores) / len(clamped_scores) if clamped_scores else 0.50
    avg = _safe_reward(avg)
    print(f"\n[SUMMARY] Average score: {avg:.4f}", flush=True)
    for task, sc in scores.items():
        print(f"  {task}: {sc:.4f}", flush=True)

    # Human-readable summary on stderr
    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL SCORES", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for task, sc in scores.items():
        status = "PASS" if sc >= 0.5 else "FAIL"
        print(f"  {task}: {sc:.4f} {status}", file=sys.stderr)
    print(f"\n  Average: {avg:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
