"""Shared prompt, parsing, and environment helpers for training/evaluation.

Keeping these utilities in one place prevents the SFT script, GRPO reward
function, Colab notebook, and frontier benchmark from silently drifting apart.
"""

from __future__ import annotations

import json
from typing import Any

from models.action import Action


SYSTEM_PROMPT = (
    "You are an expert SRE investigator. Respond with ONLY a JSON object "
    "containing your next investigation action. Available actions: "
    "query_logs, fetch_trace, diff_commit, inspect_config, inspect_infra, "
    "discover_topology, hypothesize, explain_chain, submit. "
    "Use 'discover_topology' (with optional 'service') to reveal "
    "dependencies in narrative mode where the graph is initially hidden."
)


def reset_with_scenario(env: Any, scenario: dict[str, Any], *, as_dict: bool = True) -> Any:
    """Reset an env from an in-memory scenario using the public env method."""
    obs = env.reset_from_scenario(scenario)
    if as_dict:
        return obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    return obs


def format_observation_compact(obs_dict: dict[str, Any]) -> str:
    """Compact observation text used by SFT, GRPO, and evaluation prompts."""
    d = obs_dict
    lines = [
        "Task: " + str(d.get("task_description", ""))[:240],
        "Step: %d/%d" % (d.get("step_number", 0), d.get("max_steps", 40)),
        "Budget: %d steps remaining" % d.get("remaining_budget", 0),
    ]

    for s in d.get("services", []):
        svc = s if isinstance(s, dict) else (s.model_dump() if hasattr(s, "model_dump") else vars(s))
        lines.append(
            "  %s: status=%s err=%.1f%% deploys=%d"
            % (
                svc.get("name", "?"),
                svc.get("status", "?"),
                float(svc.get("error_rate_during_incident") or 0),
                int(svc.get("recent_deploy_count") or 0),
            )
        )

    commits = d.get("available_commits", [])
    if commits:
        lines.append("Recent commits:")
        for c in commits[:8]:
            lines.append(
                "  commit %s [%s] %s"
                % (c.get("hash", "?"), c.get("service", "?"), c.get("message", "")[:80])
            )

    configs = d.get("available_config_changes", [])
    if configs:
        lines.append("Config changes:")
        for c in configs[:6]:
            lines.append(
                "  config %s [%s] %s"
                % (c.get("config_id", "?"), c.get("service", "?"), c.get("description", "")[:80])
            )

    infra = d.get("available_infra_events", [])
    if infra:
        lines.append("Infra events:")
        for e in infra[:6]:
            lines.append("  infra %s %s" % (e.get("event_id", "?"), e.get("description", "")[:80]))

    facts = d.get("known_facts", [])
    if facts:
        lines.append("Known facts (%d):" % len(facts))
        for fact in facts[-6:]:
            lines.append("  > " + str(fact)[:180])

    qr = d.get("query_result", "")
    if qr:
        lines.append("Last result: " + str(qr)[:600])

    return "\n".join(lines)


def parse_action_json(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract the first JSON action object from a model completion."""
    default = fallback or {"action_type": "query_logs", "service": "data", "keyword": "error"}
    s = (text or "").strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        last_fence = s.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            s = s[first_nl + 1:last_fence].strip()

    start = s.find("{")
    end = s.rfind("}") + 1
    if start < 0 or end <= start:
        return dict(default)

    try:
        parsed = json.loads(s[start:end])
    except json.JSONDecodeError:
        return dict(default)

    if not isinstance(parsed, dict) or not parsed.get("action_type"):
        return dict(default)
    return parsed


def parse_action_plan(text: str) -> list[dict[str, Any]]:
    """Parse a completion as a *plan* — a sequence of action dicts.

    Recognised formats (tried in order):

    1. ``[ {action_type: ...}, {action_type: ...}, ... ]`` — a JSON list.
    2. Multiple ``{...}`` JSON objects separated by whitespace, commas,
       newlines, or markdown fences. Each object is parsed independently;
       any object missing ``action_type`` is skipped.
    3. A single object — degenerate plan of length 1.

    Used by the GRPO ``reward_fn`` to roll the environment forward
    through the model's planned trajectory and return the *cumulative*
    episode reward, not just the reward for one action. This addresses
    the brutal-rating §4 "GRPO only sees per-action reward" concern by
    aligning the optimisation signal with the deliverable: a complete
    investigation, not a single move.

    Returns an empty list when nothing can be parsed.
    """
    s = (text or "").strip()
    if not s:
        return []

    # Strip a single fenced ``` block if present.
    if s.startswith("```"):
        first_nl = s.find("\n")
        last_fence = s.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            s = s[first_nl + 1:last_fence].strip()

    # Try a JSON array first.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [
                a for a in parsed
                if isinstance(a, dict) and a.get("action_type")
            ]
        if isinstance(parsed, dict) and parsed.get("action_type"):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fall back: scan for top-level ``{...}`` blocks. We track brace depth
    # so nested objects (e.g. ``"chain": [{...}]``) don't terminate parsing
    # prematurely. Strings (with escaped chars) are tracked too so a stray
    # ``}`` inside a string doesn't fool the depth counter.
    actions: list[dict[str, Any]] = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = s[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and obj.get("action_type"):
                        actions.append(obj)
                    start = -1

    return actions


def action_from_dict(action_dict: dict[str, Any]) -> Action:
    """Build an Action while ignoring unknown keys from model output."""
    return Action(
        action_type=action_dict.get("action_type", "query_logs"),
        service=action_dict.get("service"),
        keyword=action_dict.get("keyword"),
        time_window=action_dict.get("time_window"),
        trace_id=action_dict.get("trace_id"),
        commit_hash=action_dict.get("commit_hash"),
        config_id=action_dict.get("config_id"),
        event_id=action_dict.get("event_id"),
        cause_entity_id=action_dict.get("cause_entity_id"),
        chain=action_dict.get("chain"),
        final_cause=action_dict.get("final_cause"),
        final_chain=action_dict.get("final_chain"),
        reason=action_dict.get("reason"),
    )


def build_chat_text(tokenizer: Any, conversation: list[dict[str, str]]) -> str:
    """Render a (system + alternating user/assistant) trace using the model's
    own chat template.

    Llama-3, Qwen, Mistral, Gemma, Phi-3 all register a chat template via
    :py:meth:`PreTrainedTokenizer.apply_chat_template`. Using it instead of a
    hard-coded Llama-2 ``[INST]<<SYS>>`` string is essential when training a
    Llama-3 model on Llama-2 markers will leak into completions and hurt
    quality.

    ``conversation`` is a list of ``{"role", "content"}`` dicts as produced by
    :func:`train.collect_demonstrations`. A leading ``"system"`` message with
    :data:`SYSTEM_PROMPT` is inserted automatically.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation)

    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None and getattr(tokenizer, "chat_template", None):
        try:
            return apply(messages, tokenize=False, add_generation_prompt=False)
        except (ValueError, TypeError):
            pass

    parts = [f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"]
    for j in range(0, len(conversation) - 1, 2):
        user_msg = conversation[j]["content"]
        asst_msg = conversation[j + 1]["content"] if j + 1 < len(conversation) else ""
        if j == 0:
            parts.append(f"{user_msg} [/INST] {asst_msg} </s>")
        else:
            parts.append(f"<s>[INST] {user_msg} [/INST] {asst_msg} </s>")
    return "".join(parts)


def build_chat_prompt(tokenizer: Any, observation_text: str) -> str:
    """Build an inference prompt for one observation using the chat template.

    Mirrors :func:`build_chat_text` but adds the assistant generation prompt so
    the model continues with its JSON action.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_text},
    ]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None and getattr(tokenizer, "chat_template", None):
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, TypeError):
            pass

    return f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{observation_text} [/INST]"
