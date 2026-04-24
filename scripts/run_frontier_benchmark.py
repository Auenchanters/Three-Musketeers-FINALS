"""Frontier-model benchmark for PostmortemEnv.

Runs one or more LLMs in-process against the 3 hand-crafted hard tasks
(plus optional procedural seeds) and saves a comparable score table.

Backends (auto-selected from env vars):
  * ``anthropic:<model>`` — needs ``ANTHROPIC_API_KEY``
  * ``openai:<model>``    — needs ``OPENAI_API_KEY``
  * ``hf:<repo/id>``      — needs ``HF_TOKEN`` (uses HF Inference Router,
                            an OpenAI-compatible endpoint)

Usage::

    python scripts/run_frontier_benchmark.py \
        --models anthropic:claude-haiku-4-5 openai:gpt-4o-mini \
        --runs-per-task 3 \
        --output training_data/frontier_results.json

If no ``--models`` flag is passed, we pick whichever models the available
API keys allow (Claude Haiku, GPT-4o-mini, then HF Llama-3.2-1B as fallback).
The script never crashes on a missing key — it simply skips that model and
records the reason in the output JSON.

The point of this benchmark: prove the env tests *real reasoning*. A
frontier model scoring ~0.3-0.5 (vs Random 0.10 / Oracle 0.98) is the
single most credible piece of evidence we can ship to judges.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# --- repo-local imports ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.environment import PostmortemEnvironment  # noqa: E402
from models.action import Action  # noqa: E402
from inference import (  # noqa: E402  (reuse existing prompt / parser)
    SYSTEM_PROMPT,
    format_observation,
    parse_action,
    obs_to_dict,
)

HAND_CRAFTED_TASKS = [
    "task1_recent_deploy",
    "task2_cascade_chain",
    "task3_correlated_cause",
]

DEFAULT_MAX_STEPS = 25  # cap per episode to keep cost bounded


# --- backend abstraction -----------------------------------------------------


@dataclass
class ChatBackend:
    """Pluggable chat backend. ``call(messages) -> str``."""

    label: str        # short tag used in output JSON, e.g. "claude-haiku-4-5"
    provider: str     # "anthropic" | "openai" | "hf"
    model: str        # provider-specific model id
    call: Callable[[List[Dict[str, str]]], str]
    available: bool = True
    skip_reason: Optional[str] = None


def _make_anthropic_backend(model: str) -> ChatBackend:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    label = model
    if not api_key:
        return ChatBackend(
            label=label, provider="anthropic", model=model,
            call=lambda _msgs: "",
            available=False,
            skip_reason="ANTHROPIC_API_KEY not set",
        )
    try:
        import anthropic  # type: ignore
    except ImportError:
        return ChatBackend(
            label=label, provider="anthropic", model=model,
            call=lambda _msgs: "",
            available=False,
            skip_reason="`anthropic` package not installed",
        )

    client = anthropic.Anthropic(api_key=api_key)

    def _call(messages: List[Dict[str, str]]) -> str:
        # Anthropic API splits system from user/assistant messages
        system = ""
        chat: List[Dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat.append({"role": m["role"], "content": m["content"]})
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.2,
            system=system,
            messages=chat,
        )
        return "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )

    return ChatBackend(label=label, provider="anthropic", model=model, call=_call)


def _make_openai_backend(model: str) -> ChatBackend:
    api_key = os.environ.get("OPENAI_API_KEY")
    label = model
    if not api_key:
        return ChatBackend(
            label=label, provider="openai", model=model,
            call=lambda _msgs: "",
            available=False,
            skip_reason="OPENAI_API_KEY not set",
        )
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return ChatBackend(
            label=label, provider="openai", model=model,
            call=lambda _msgs: "",
            available=False,
            skip_reason="`openai` package not installed",
        )

    client = OpenAI(api_key=api_key)

    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip()

    return ChatBackend(label=label, provider="openai", model=model, call=_call)


def _make_hf_backend(model: str) -> ChatBackend:
    """HuggingFace Inference Router (OpenAI-compatible)."""
    hf_token = os.environ.get("HF_TOKEN")
    label = model.split("/")[-1] if "/" in model else model
    if not hf_token:
        return ChatBackend(
            label=label, provider="hf", model=model,
            call=lambda _msgs: "",
            available=False,
            skip_reason="HF_TOKEN not set",
        )
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return ChatBackend(
            label=label, provider="hf", model=model,
            call=lambda _msgs: "",
            available=False,
            skip_reason="`openai` package not installed",
        )

    base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    client = OpenAI(api_key=hf_token, base_url=base_url)

    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Some thinking models wrap reasoning in <think>...</think>
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    return ChatBackend(label=label, provider="hf", model=model, call=_call)


def build_backend(spec: str) -> ChatBackend:
    """Parse a ``provider:model`` spec into a ``ChatBackend``."""
    if ":" not in spec:
        raise ValueError(
            f"Bad model spec '{spec}'. Use 'anthropic:<model>', "
            "'openai:<model>', or 'hf:<repo/id>'."
        )
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider == "anthropic":
        return _make_anthropic_backend(model)
    if provider == "openai":
        return _make_openai_backend(model)
    if provider in ("hf", "huggingface"):
        return _make_hf_backend(model)
    raise ValueError(f"Unknown provider '{provider}' in spec '{spec}'")


def auto_select_backends() -> List[ChatBackend]:
    """Pick the strongest available backend per provider."""
    candidates = [
        "anthropic:claude-haiku-4-5",
        "openai:gpt-4o-mini",
        "hf:meta-llama/Llama-3.2-1B-Instruct",
    ]
    backends = [build_backend(spec) for spec in candidates]
    available = [b for b in backends if b.available]
    if not available:
        # Still return them so the output JSON records the skip reasons.
        print(
            "[WARN] No API keys detected (ANTHROPIC_API_KEY / OPENAI_API_KEY / "
            "HF_TOKEN). Will write a results file documenting which backends "
            "were skipped.",
            file=sys.stderr,
        )
    return backends


# --- episode runner ----------------------------------------------------------


def run_episode(
    backend: ChatBackend,
    env: PostmortemEnvironment,
    task_id: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    history_window: int = 4,
) -> Dict[str, Any]:
    """Run one episode of ``task_id`` with ``backend`` and return the result row."""
    obs = env.reset(task_id=task_id)
    obs_dict = obs_to_dict(obs)

    rewards: List[float] = []
    actions_log: List[str] = []
    history: List[Dict[str, str]] = []
    last_action_dict: Dict[str, Any] = {}

    err_msg: Optional[str] = None
    step_no = 0

    for step_no in range(1, max_steps + 1):
        obs_text = format_observation(obs_dict)
        remaining = obs_dict.get("remaining_budget", max_steps - step_no)

        # Auto-submit on last step to lock in any progress
        if remaining <= 1:
            action_dict = {
                "action_type": "submit",
                "final_cause": last_action_dict.get("final_cause", "unknown"),
                "final_chain": last_action_dict.get("final_chain", []),
                "reason": "Auto-submit on last step",
            }
            response_text = json.dumps(action_dict)
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-history_window:]:
                messages.append({"role": "assistant", "content": h["action"]})
                messages.append({"role": "user", "content": h["result"]})
            messages.append({"role": "user", "content": obs_text})
            try:
                response_text = backend.call(messages)
            except Exception as e:  # noqa: BLE001
                err_msg = f"backend error: {e}"
                response_text = (
                    '{"action_type":"submit","final_cause":"unknown",'
                    '"final_chain":[],"reason":"Backend exception"}'
                )
            action_dict = parse_action(response_text)

        last_action_dict = action_dict

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
        try:
            obs = env.step(action)
        except Exception as e:  # noqa: BLE001
            err_msg = f"env.step error: {e}"
            break
        obs_dict = obs_to_dict(obs)
        rewards.append(float(getattr(obs, "reward", 0.0) or 0.0))
        actions_log.append(action_dict.get("action_type", "?"))
        history.append({
            "action": json.dumps(action_dict, separators=(",", ":")),
            "result": obs_dict.get("query_result", obs_dict.get("message", ""))[:600],
        })

        if obs_dict.get("done") or err_msg:
            break

    state = env.state
    final_score = float(state.final_score) if state.final_score is not None else (
        rewards[-1] if rewards else 0.01
    )

    return {
        "model": backend.label,
        "provider": backend.provider,
        "model_id": backend.model,
        "task_id": task_id,
        "n_steps": step_no,
        "final_score": round(final_score, 4),
        "reward_trace": [round(r, 4) for r in rewards],
        "action_sequence": actions_log,
        "error": err_msg,
    }


# --- driver ------------------------------------------------------------------


def run_benchmark(
    backends: List[ChatBackend],
    tasks: List[str],
    runs_per_task: int,
    max_steps: int,
    output_path: Path,
) -> Dict[str, Any]:
    started = time.time()
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for backend in backends:
        if not backend.available:
            skipped.append({
                "model": backend.label,
                "provider": backend.provider,
                "reason": backend.skip_reason or "unknown",
            })
            print(f"[SKIP] {backend.label} ({backend.provider}) — {backend.skip_reason}",
                  file=sys.stderr)
            continue

        env = PostmortemEnvironment()
        for task_id in tasks:
            for run_idx in range(runs_per_task):
                t0 = time.time()
                try:
                    row = run_episode(backend, env, task_id, max_steps=max_steps)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    row = {
                        "model": backend.label,
                        "provider": backend.provider,
                        "model_id": backend.model,
                        "task_id": task_id,
                        "n_steps": 0,
                        "final_score": 0.01,
                        "reward_trace": [],
                        "action_sequence": [],
                        "error": f"runner crash: {e}",
                    }
                row["run_idx"] = run_idx
                row["elapsed_s"] = round(time.time() - t0, 2)
                rows.append(row)
                print(
                    f"[RUN] {backend.label} | {task_id} | run {run_idx+1}/{runs_per_task} "
                    f"-> score={row['final_score']:.3f} steps={row['n_steps']} "
                    f"({row['elapsed_s']:.1f}s)",
                    file=sys.stderr,
                )

    # --- aggregation ---------------------------------------------------------
    summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        bucket = summary.setdefault(model, {
            "provider": row["provider"],
            "model_id": row["model_id"],
            "scores": [],
            "by_task": {},
        })
        bucket["scores"].append(row["final_score"])
        per_task = bucket["by_task"].setdefault(row["task_id"], [])
        per_task.append(row["final_score"])

    for model, bucket in summary.items():
        scores = bucket["scores"]
        bucket["mean_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0
        bucket["n_runs"] = len(scores)
        bucket["min_score"] = round(min(scores), 4) if scores else 0.0
        bucket["max_score"] = round(max(scores), 4) if scores else 0.0
        bucket["by_task_mean"] = {
            t: round(sum(s) / len(s), 4) for t, s in bucket["by_task"].items()
        }

    payload = {
        "tasks": tasks,
        "runs_per_task": runs_per_task,
        "max_steps": max_steps,
        "elapsed_s": round(time.time() - started, 2),
        "skipped": skipped,
        "summary": summary,
        "runs": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[DONE] wrote {output_path}", file=sys.stderr)
    return payload


def _print_table(payload: Dict[str, Any]) -> None:
    print("\n" + "=" * 64, file=sys.stderr)
    print("Frontier-model benchmark", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"{'Model':<32} {'Mean':>7} {'Min':>7} {'Max':>7} {'N':>4}", file=sys.stderr)
    print("-" * 64, file=sys.stderr)
    for model, bucket in payload["summary"].items():
        print(
            f"{model:<32} {bucket['mean_score']:>7.3f} "
            f"{bucket['min_score']:>7.3f} {bucket['max_score']:>7.3f} "
            f"{bucket['n_runs']:>4}",
            file=sys.stderr,
        )
    if payload["skipped"]:
        print("\nSkipped:", file=sys.stderr)
        for s in payload["skipped"]:
            print(f"  - {s['model']} ({s['provider']}): {s['reason']}", file=sys.stderr)
    print("=" * 64, file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "List of backends, e.g. anthropic:claude-haiku-4-5 openai:gpt-4o-mini "
            "hf:meta-llama/Llama-3.2-1B-Instruct. Omit to auto-select based on "
            "available API keys."
        ),
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=HAND_CRAFTED_TASKS,
        help=f"Task IDs to evaluate (default: {HAND_CRAFTED_TASKS}).",
    )
    p.add_argument(
        "--runs-per-task",
        type=int,
        default=3,
        help="How many independent runs per (model, task) pair (default: 3).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "training_data" / "frontier_results.json",
        help="Where to write the JSON results.",
    )
    args = p.parse_args()

    if args.models:
        backends = [build_backend(spec) for spec in args.models]
    else:
        backends = auto_select_backends()

    payload = run_benchmark(
        backends,
        tasks=args.tasks,
        runs_per_task=args.runs_per_task,
        max_steps=args.max_steps,
        output_path=args.output,
    )
    _print_table(payload)


if __name__ == "__main__":
    main()
