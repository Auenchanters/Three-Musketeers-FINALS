"""
PostmortemEnv — Data Loader

Loads curated scenario JSON files and procedurally generated scenarios.
Supports both hand-crafted fixtures AND seed-based generation.
"""

import functools
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# Base paths
DATA_DIR = Path(__file__).parent
SCENARIOS_DIR = DATA_DIR / "scenarios"
SOLUTIONS_DIR = DATA_DIR / "solutions"
GENERATED_DIR = DATA_DIR / "generated"

# Task ID → filename mapping (hand-crafted scenarios)
SCENARIO_FILES = {
    "task1_recent_deploy": "task1_recent_deploy.json",
    "task2_cascade_chain": "task2_cascade_chain.json",
    "task3_correlated_cause": "task3_correlated_cause.json",
}

SOLUTION_FILES = {
    "task1_recent_deploy": "task1_solution.json",
    "task2_cascade_chain": "task2_solution.json",
    "task3_correlated_cause": "task3_solution.json",
}


@functools.lru_cache(maxsize=8)
def _read_json_cached(path_str: str) -> str:
    """Cache the *raw* JSON text for the 3 hand-crafted scenarios + 3
    solutions. Returning text instead of the parsed dict is intentional —
    callers ``json.loads`` it to get a fresh, mutation-safe dict every
    time, while still skipping the disk read after the first call.
    """
    with open(path_str, "r", encoding="utf-8") as f:
        return f.read()


def load_scenario(task_id: str) -> Dict[str, Any]:
    """
    Load a scenario by task_id.

    First checks hand-crafted scenarios, then generated ones,
    then tries procedural generation on-the-fly for seed_* IDs.
    """
    # 1) Hand-crafted (cached on disk-read since the 3 fixture files never change)
    if task_id in SCENARIO_FILES:
        scenario_path = SCENARIOS_DIR / SCENARIO_FILES[task_id]
        return json.loads(_read_json_cached(str(scenario_path)))

    # 2) Pre-generated
    gen_path = GENERATED_DIR / "scenarios" / f"{task_id}.json"
    if gen_path.exists():
        with open(gen_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 3) On-the-fly procedural generation for seed_* IDs
    if task_id.startswith("seed_"):
        from data.seed_generator import generate_scenario
        parts = task_id.split("_")
        # Format: seed_{num}_{difficulty}
        if len(parts) >= 3:
            try:
                seed = int(parts[1])
                difficulty = parts[2]
                return generate_scenario(seed, difficulty)
            except (ValueError, IndexError):
                pass

    available = ", ".join(list(SCENARIO_FILES.keys()) + _list_generated_tasks())
    raise ValueError(f"Unknown task_id: '{task_id}'. Available: {available}")


def load_solution(task_id: str) -> Dict[str, Any]:
    """Load the oracle solution for a task."""
    # 1) Hand-crafted (cached on disk-read since the 3 fixture files never change)
    if task_id in SOLUTION_FILES:
        solution_path = SOLUTIONS_DIR / SOLUTION_FILES[task_id]
        return json.loads(_read_json_cached(str(solution_path)))

    # 2) Pre-generated
    gen_path = GENERATED_DIR / "solutions" / f"{task_id}_solution.json"
    if gen_path.exists():
        with open(gen_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 3) On-the-fly for seed_* IDs
    if task_id.startswith("seed_"):
        from data.seed_generator import generate_scenario, generate_oracle_solution
        parts = task_id.split("_")
        if len(parts) >= 3:
            try:
                seed = int(parts[1])
                difficulty = parts[2]
                scenario = generate_scenario(seed, difficulty)
                return generate_oracle_solution(scenario, seed)
            except (ValueError, IndexError):
                pass

    available = ", ".join(list(SOLUTION_FILES.keys()) + _list_generated_tasks())
    raise ValueError(f"Unknown task_id: '{task_id}'. Available: {available}")


def get_available_tasks() -> List[str]:
    """Return list of available task IDs (hand-crafted only for core tasks)."""
    return list(SCENARIO_FILES.keys())


def get_all_tasks(include_generated: bool = True) -> List[str]:
    """Return all available task IDs including generated ones."""
    tasks = list(SCENARIO_FILES.keys())
    if include_generated:
        tasks += _list_generated_tasks()
    return tasks


def get_ground_truth(task_id: str) -> Dict[str, Any]:
    """Get the oracle ground truth for a task."""
    solution = load_solution(task_id)
    return solution["ground_truth"]


def _list_generated_tasks() -> List[str]:
    """List task IDs from generated scenarios directory."""
    gen_scenarios = GENERATED_DIR / "scenarios"
    if not gen_scenarios.exists():
        return []
    return [
        p.stem for p in gen_scenarios.glob("*.json")
    ]
