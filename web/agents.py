"""
Agent runners for the live UI.

Three agents:
  * OracleAgent: replays the hand-crafted optimal action sequence. Deterministic,
                 no LLM. Useful for showing "this is what perfect looks like".
  * RandomAgent: picks random valid actions. Useful as a baseline.
  * LLMAgent:    token-efficient investigation agent. Uses Anthropic prompt
                 caching on the system prompt and sends observation DELTAS
                 (not the full state) on each step. Default model is Claude
                 Haiku 4.5 for max speed / min cost.

Each agent is an async generator that yields event dicts the run manager
forwards over SSE:

    {"type": "step", "step": int, "action": dict, "observation": {...}}
    {"type": "thought", "text": str}            # LLM only
    {"type": "usage", "input_tokens": int, "output_tokens": int,
                      "cache_read_tokens": int, "cache_creation_tokens": int}
    {"type": "done", "score": float, "steps": int, "cause": str, "chain": [...]}
    {"type": "error", "message": str}
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any, AsyncGenerator

from data.generator import load_solution
from engine.environment import PostmortemEnvironment
from models.action import Action, ActionType


def _obs_summary(obs: Any) -> dict[str, Any]:
    """Compact representation of an observation for UI streaming."""
    return {
        "step_number": obs.step_number,
        "remaining_budget": obs.remaining_budget,
        "reward": obs.reward,
        "done": obs.done,
        "message": obs.message,
        "query_result": obs.query_result[:4000] if obs.query_result else "",
        "known_facts": list(obs.known_facts)[-12:],
        "hypotheses_submitted": obs.hypotheses_submitted,
        "wrong_hypotheses": obs.wrong_hypotheses,
    }


def _obs_seed(obs: Any) -> dict[str, Any]:
    """One-shot scenario summary sent once at episode start."""
    return {
        "task_description": obs.task_description,
        "max_steps": obs.max_steps,
        "incident_window": obs.incident_window,
        "service_graph": obs.service_graph,
        "services": [s.model_dump() for s in obs.services],
        "available_commits": obs.available_commits,
        "available_config_changes": obs.available_config_changes,
        "available_trace_ids": obs.available_trace_ids,
        "available_infra_events": obs.available_infra_events,
    }


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


class OracleAgent:
    agent_id = "oracle"

    async def run(
        self, env: PostmortemEnvironment, task_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        solution = load_solution(task_id)
        sequence = solution["optimal_action_sequence"]

        obs = env.reset(task_id=task_id)
        yield {"type": "reset", "task_id": task_id, "scenario": _obs_seed(obs)}

        for i, step in enumerate(sequence, start=1):
            step_copy = dict(step) if isinstance(step, dict) else {}
            reason = step_copy.pop("reason", "")
            action = Action(**step_copy)
            obs = env.step(action)
            yield {
                "type": "step",
                "step": i,
                "action": step_copy,
                "reason": reason,
                "observation": _obs_summary(obs),
            }
            if obs.done:
                break
            await asyncio.sleep(0.22)

        yield {
            "type": "done",
            "score": env.get_final_score(),
            "steps": env.state.steps_taken,
            "cause": env.state.ground_truth_cause,
            "chain": env.state.ground_truth_chain,
        }


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class RandomAgent:
    agent_id = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def run(
        self, env: PostmortemEnvironment, task_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        obs = env.reset(task_id=task_id)
        yield {"type": "reset", "task_id": task_id, "scenario": _obs_seed(obs)}

        services = list(obs.service_graph.keys())
        keywords = ["error", "deploy", "connection", "timeout", "OOM", "restart"]
        commits = [c["hash"] for c in obs.available_commits]
        configs = [c["config_id"] for c in obs.available_config_changes]
        traces = list(obs.available_trace_ids)
        infra_events = [e["event_id"] for e in obs.available_infra_events]

        i = 0
        while not obs.done and i < obs.max_steps:
            i += 1
            action = self._sample(
                services, keywords, commits, configs, traces, i, obs.max_steps,
                infra_events=infra_events,
            )
            step_dict = action.model_dump(exclude_none=True)
            obs = env.step(action)
            yield {
                "type": "step",
                "step": i,
                "action": step_dict,
                "observation": _obs_summary(obs),
            }
            await asyncio.sleep(0.05)

        yield {
            "type": "done",
            "score": env.get_final_score(),
            "steps": env.state.steps_taken,
            "cause": env.state.ground_truth_cause,
            "chain": env.state.ground_truth_chain,
        }

    def _sample(
        self,
        services: list[str],
        keywords: list[str],
        commits: list[str],
        configs: list[str],
        traces: list[str],
        step: int,
        max_steps: int,
        infra_events: list[str] | None = None,
    ) -> Action:
        if step >= max_steps - 1:
            return Action(action_type=ActionType.SUBMIT, final_cause="", final_chain=[])
        infra_events = infra_events or []
        roll = self._rng.random()
        if roll < 0.50:
            return Action(
                action_type=ActionType.QUERY_LOGS,
                service=self._rng.choice(services) if services else "data",
                keyword=self._rng.choice(keywords),
            )
        if roll < 0.65 and traces:
            return Action(action_type=ActionType.FETCH_TRACE, trace_id=self._rng.choice(traces))
        if roll < 0.77 and commits:
            return Action(action_type=ActionType.DIFF_COMMIT, commit_hash=self._rng.choice(commits))
        if roll < 0.85 and configs:
            return Action(action_type=ActionType.INSPECT_CONFIG, config_id=self._rng.choice(configs))
        if roll < 0.91 and infra_events:
            return Action(
                action_type=ActionType.INSPECT_INFRA,
                event_id=self._rng.choice(infra_events),
            )
        if roll < 0.96 and commits:
            return Action(
                action_type=ActionType.HYPOTHESIZE,
                cause_entity_id=self._rng.choice(commits),
            )
        return Action(
            action_type=ActionType.SUBMIT,
            final_cause=self._rng.choice(commits) if commits else "",
            final_chain=[],
        )


# ---------------------------------------------------------------------------
# LLM agent (token-efficient, prompt-cached)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are an SRE investigating a cloud outage. You see a frozen telemetry snapshot and must identify the ROOT CAUSE and CAUSAL CHAIN through queries.

Output ONLY a single compact JSON object, no prose, no markdown.

Actions:
  {"action_type":"query_logs","service":"<name>","keyword":"<word>","time_window":"last_5m"}
  {"action_type":"fetch_trace","trace_id":"<id>"}
  {"action_type":"diff_commit","commit_hash":"<hash>"}
  {"action_type":"inspect_config","config_id":"<id>"}
  {"action_type":"inspect_infra","event_id":"<id>"}
  {"action_type":"hypothesize","cause_entity_id":"<id>"}
  {"action_type":"explain_chain","chain":[{"service":"<s>","effect":"<e>"}, ...]}
  {"action_type":"submit","final_cause":"<id>","final_chain":[{"service":"<s>","effect":"<e>"}, ...]}

Strategy: start at the service with errors in the incident window → diff recent commits there → inspect configs / traces / infra events → hypothesize, then explain_chain, then submit. Spend steps like money. No repeats.""".strip()


def _first_user_turn(obs_seed: dict[str, Any]) -> str:
    """Minimal one-shot scenario brief. Marked for prompt caching upstream."""
    return json.dumps(
        {
            "brief": obs_seed["task_description"],
            "budget": obs_seed["max_steps"],
            "window": obs_seed["incident_window"],
            "graph": obs_seed["service_graph"],
            "services": [s.get("name") for s in obs_seed["services"]],
            "commits": [
                {"hash": c["hash"], "service": c["service"], "msg": c["message"][:80]}
                for c in obs_seed["available_commits"]
            ],
            "configs": [
                {"id": c["config_id"], "service": c["service"], "desc": c.get("description", "")[:60]}
                for c in obs_seed["available_config_changes"]
            ],
            "traces": obs_seed["available_trace_ids"],
            "infra": [
                {"id": e["event_id"], "desc": e["description"][:80]}
                for e in obs_seed["available_infra_events"]
            ],
        },
        separators=(",", ":"),
    )


def _delta_turn(obs_summary: dict[str, Any]) -> str:
    """Per-step observation delta. Omits static fields — keeps tokens tiny."""
    return json.dumps(
        {
            "step": obs_summary["step_number"],
            "budget_left": obs_summary["remaining_budget"],
            "facts_known": obs_summary["known_facts"],
            "last": obs_summary["query_result"][:1200],
            "wrong_hyp": obs_summary["wrong_hypotheses"],
        },
        separators=(",", ":"),
    )


class LLMAgent:
    """Token-efficient LLM agent. Anthropic by default; OpenAI fallback."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 180,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        if self.provider == "anthropic":
            default_model = "claude-haiku-4-5-20251001"
            default_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "moonshot":
            default_model = "moonshot-v1-8k" # fallback or kimi-k2.6 based on env
            default_key = os.getenv("MOONSHOT_API_KEY")
        else:
            default_model = "gpt-4o-mini"
            default_key = os.getenv("OPENAI_API_KEY")
            
        self.model = model or os.getenv("LLM_MODEL", default_model)
        self.api_key = api_key or default_key
        self.max_tokens = max_tokens
        self.agent_id = f"llm:{self.model}"

    async def run(
        self, env: PostmortemEnvironment, task_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        if not self.api_key:
            yield {
                "type": "error",
                "message": (
                    f"No API key for provider='{self.provider}'. Paste a key in the "
                    "LLM panel above, or set ANTHROPIC_API_KEY / OPENAI_API_KEY on "
                    "the server. Tip: Oracle and Random investigators run locally "
                    "with no key — try one of those to see the environment."
                ),
            }
            return

        obs = env.reset(task_id=task_id)
        seed = _obs_seed(obs)
        yield {"type": "reset", "task_id": task_id, "scenario": seed}

        # Transcript: one cached system + one cached seed + per-step deltas.
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _first_user_turn(seed)}
        ]

        total_in = 0
        total_out = 0
        total_cache_read = 0
        total_cache_create = 0
        step = 0

        while not obs.done and step < obs.max_steps:
            step += 1
            try:
                raw, usage = await self._complete(messages)
            except Exception as exc:  # noqa: BLE001
                yield {"type": "error", "message": f"LLM call failed: {exc}"}
                break

            total_in += usage.get("input_tokens", 0)
            total_out += usage.get("output_tokens", 0)
            total_cache_read += usage.get("cache_read_tokens", 0)
            total_cache_create += usage.get("cache_creation_tokens", 0)

            try:
                action_dict = _parse_action_json(raw)
            except ValueError:
                yield {
                    "type": "thought",
                    "text": f"Unparseable action, submitting best guess: {raw[:160]}",
                }
                action_dict = {"action_type": "submit", "final_cause": "", "final_chain": []}

            yield {
                "type": "thought",
                "text": f"(step {step}) {raw[:220]}",
            }
            action = Action(**action_dict)
            obs = env.step(action)
            summary = _obs_summary(obs)

            yield {
                "type": "step",
                "step": step,
                "action": action_dict,
                "observation": summary,
            }
            yield {
                "type": "usage",
                "input_tokens": total_in,
                "output_tokens": total_out,
                "cache_read_tokens": total_cache_read,
                "cache_creation_tokens": total_cache_create,
            }

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": _delta_turn(summary)})

            if obs.done:
                break

        yield {
            "type": "done",
            "score": env.get_final_score(),
            "steps": env.state.steps_taken,
            "cause": env.state.ground_truth_cause,
            "chain": env.state.ground_truth_chain,
            "usage": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "cache_read_tokens": total_cache_read,
                "cache_creation_tokens": total_cache_create,
            },
        }

    async def _complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
        if self.provider == "anthropic":
            return await self._complete_anthropic(messages)
        elif self.provider == "moonshot":
            return await self._complete_moonshot(messages)
        return await self._complete_openai(messages)

    async def _complete_anthropic(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, dict[str, int]]:
        import httpx

        # Cache the system prompt AND the one-shot scenario seed (first user turn).
        # Subsequent per-step deltas are small and not cached.
        rendered_messages: list[dict[str, Any]] = []
        for idx, m in enumerate(messages):
            if idx == 0 and m["role"] == "user":
                rendered_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": m["content"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                rendered_messages.append(m)

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": rendered_messages,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return text.strip(), {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "cache_read_tokens": int(usage.get("cache_read_input_tokens", 0)),
            "cache_creation_tokens": int(usage.get("cache_creation_input_tokens", 0)),
        }

    async def _complete_openai(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, dict[str, int]]:
        import httpx

        full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        payload = {
            "model": self.model,
            "messages": full,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text.strip(), {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "cache_read_tokens": int(
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            ),
            "cache_creation_tokens": 0,
        }


    async def _complete_moonshot(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, dict[str, int]]:
        import httpx

        full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        payload = {
            "model": self.model,
            "messages": full,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.moonshot.cn/v1/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text.strip(), {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }

def _parse_action_json(raw: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response."""
    s = raw.strip()
    # Strip fences like ```json ... ```
    if s.startswith("```"):
        first_nl = s.find("\n")
        last_fence = s.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            s = s[first_nl + 1 : last_fence].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    candidate = s[start : end + 1]
    return json.loads(candidate)
