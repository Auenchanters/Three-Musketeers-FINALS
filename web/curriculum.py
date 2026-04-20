"""
Adaptive curriculum: tracks per-task ELO and picks difficulty for the next run.

Simple ELO-style update after each run:
    expected = 1 / (1 + 10 ** ((task_elo - agent_elo) / 400))
    delta    = K * (score - expected)

Difficulty multiplier rises when the agent beats its expected score on a task
and falls otherwise. Procedural seeds read this multiplier to scale noise /
chain depth / red-herring count.

State is persisted to training_data/curriculum_state.json so the HF Space
retains progress across restarts (ephemeral; resets on container rebuild).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

STATE_PATH = Path(__file__).resolve().parent.parent / "training_data" / "curriculum_state.json"
K_FACTOR = 24.0
DEFAULT_AGENT_ELO = 1200.0
DEFAULT_TASK_ELO = {
    "task1_recent_deploy": 1000.0,
    "task2_cascade_chain": 1300.0,
    "task3_correlated_cause": 1600.0,
}

_lock = threading.RLock()


def _default_state() -> dict[str, Any]:
    return {
        "agents": {},
        "tasks": {
            tid: {
                "elo": elo,
                "attempts": 0,
                "solves": 0,
                "difficulty_multiplier": 1.0,
                "last_score": None,
            }
            for tid, elo in DEFAULT_TASK_ELO.items()
        },
        "recent_runs": [],
    }


def load_state() -> dict[str, Any]:
    with _lock:
        if not STATE_PATH.exists():
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = _default_state()
            STATE_PATH.write_text(json.dumps(state, indent=2))
            return state
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return _default_state()


def save_state(state: dict[str, Any]) -> None:
    with _lock:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))



def get_agent_elo(agent_id: str) -> float:
    state = load_state()
    return float(state["agents"].get(agent_id, {}).get("elo", DEFAULT_AGENT_ELO))


def get_task_state(task_id: str) -> dict[str, Any]:
    state = load_state()
    tasks = state.setdefault("tasks", {})
    if task_id not in tasks:
        tasks[task_id] = {
            "elo": DEFAULT_TASK_ELO.get(task_id, 1400.0),
            "attempts": 0,
            "solves": 0,
            "difficulty_multiplier": 1.0,
            "last_score": None,
        }
        save_state(state)
    return tasks[task_id]


def difficulty_multiplier(task_id: str) -> float:
    """Procedural seed generator reads this to scale noise & chain depth."""
    return float(get_task_state(task_id).get("difficulty_multiplier", 1.0))


def record_run(
    agent_id: str,
    task_id: str,
    score: float,
    steps_used: int,
    max_steps: int,
) -> dict[str, Any]:
    """Update ELO + difficulty after a run. Returns the delta summary."""
    with _lock:
        state = load_state()
        agents = state.setdefault("agents", {})
        tasks = state.setdefault("tasks", {})

        agent = agents.setdefault(
            agent_id, {"elo": DEFAULT_AGENT_ELO, "runs": 0, "wins": 0}
        )
        task = tasks.setdefault(
            task_id,
            {
                "elo": DEFAULT_TASK_ELO.get(task_id, 1400.0),
                "attempts": 0,
                "solves": 0,
                "difficulty_multiplier": 1.0,
                "last_score": None,
            },
        )

        expected = 1.0 / (1.0 + 10.0 ** ((task["elo"] - agent["elo"]) / 400.0))
        solved = 1.0 if score >= 0.70 else 0.0
        agent_delta = K_FACTOR * (solved - expected)
        task_delta = -agent_delta

        agent["elo"] = round(agent["elo"] + agent_delta, 2)
        agent["runs"] = int(agent.get("runs", 0)) + 1
        agent["wins"] = int(agent.get("wins", 0)) + int(solved)

        task["elo"] = round(task["elo"] + task_delta, 2)
        task["attempts"] = int(task.get("attempts", 0)) + 1
        task["solves"] = int(task.get("solves", 0)) + int(solved)
        task["last_score"] = round(float(score), 4)

        # Difficulty tuning: clamp in [0.6, 2.5]
        mult = float(task.get("difficulty_multiplier", 1.0))
        if solved and steps_used <= max_steps * 0.6:
            mult = min(2.5, mult * 1.08)
        elif solved:
            mult = min(2.5, mult * 1.02)
        else:
            mult = max(0.6, mult * 0.94)
        task["difficulty_multiplier"] = round(mult, 3)

        # History tail (last 20)
        recent = state.setdefault("recent_runs", [])
        recent.append(
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "score": round(float(score), 4),
                "steps_used": steps_used,
                "max_steps": max_steps,
                "solved": bool(solved),
            }
        )
        state["recent_runs"] = recent[-20:]

        save_state(state)

        return {
            "agent_elo": agent["elo"],
            "agent_elo_delta": round(agent_delta, 2),
            "task_elo": task["elo"],
            "task_elo_delta": round(task_delta, 2),
            "difficulty_multiplier": task["difficulty_multiplier"],
            "solved": bool(solved),
        }
