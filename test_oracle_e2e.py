"""
PostmortemEnv — Oracle End-to-End Validation

Runs the pre-computed optimal solutions against the environment to prove
that the full pipeline works correctly: reset -> step -> grading -> scoring.

NO LLM or API credits required. This uses deterministic oracle solutions.

Usage:
    python test_oracle_e2e.py                     # Test against local server
    python test_oracle_e2e.py --url https://...   # Test against deployed HF Space
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import PostmortemClient
from models.action import Action
from data.generator import load_solution, get_available_tasks


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={json.dumps(action)} reward={reward} done={done} error={error}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={success} steps={steps} score={score} rewards={json.dumps(rewards)}",
        flush=True,
    )


def run_oracle_task(env_url, task_id):
    """Run the oracle solution for a single task and return results."""
    solution = load_solution(task_id)
    actions = solution["optimal_action_sequence"]
    gt = solution["ground_truth"]

    log_start(task=task_id, env="PostmortemEnv", model="oracle-solution")

    sync_client = PostmortemClient(base_url=env_url).sync()
    rewards = []
    steps_taken = 0

    with sync_client:
        result = sync_client.reset(task_id=task_id)
        obs = result.observation
        print(f"  Task: {task_id} | Remaining budget: {obs.remaining_budget}", flush=True)

        for i, step_data in enumerate(actions, 1):
            action = Action(
                action_type=step_data["action_type"],
                service=step_data.get("service"),
                keyword=step_data.get("keyword"),
                time_window=step_data.get("time_window"),
                trace_id=step_data.get("trace_id"),
                commit_hash=step_data.get("commit_hash"),
                config_id=step_data.get("config_id"),
                cause_entity_id=step_data.get("cause_entity_id"),
                chain=step_data.get("chain"),
                final_cause=step_data.get("final_cause"),
                final_chain=step_data.get("final_chain"),
                reason=step_data.get("reason"),
            )

            result = sync_client.step(action)
            obs = result.observation
            reward = result.reward if isinstance(result.reward, (int, float)) else 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = i

            action_dict = {
                "action_type": step_data["action_type"],
            }
            if step_data.get("service"):
                action_dict["service"] = step_data["service"]
            if step_data.get("keyword"):
                action_dict["keyword"] = step_data["keyword"]
            if step_data.get("commit_hash"):
                action_dict["commit_hash"] = step_data["commit_hash"]
            if step_data.get("trace_id"):
                action_dict["trace_id"] = step_data["trace_id"]
            if step_data.get("config_id"):
                action_dict["config_id"] = step_data["config_id"]
            if step_data.get("cause_entity_id"):
                action_dict["cause_entity_id"] = step_data["cause_entity_id"]

            log_step(step=i, action=action_dict, reward=reward, done=done)

            if done:
                break

        # Get final score from state
        try:
            final_state = sync_client.state()
            state_dict = final_state.model_dump() if hasattr(final_state, "model_dump") else vars(final_state)
            score = state_dict.get("final_score", 0.01)
        except Exception:
            score = 0.01

    score = max(0.01, min(0.99, float(score))) if score is not None else 0.01
    success = score >= 0.5

    log_end(success=success, steps=steps_taken, score=round(score, 4), rewards=rewards)

    return {
        "task_id": task_id,
        "score": round(score, 3),
        "success": success,
        "steps": steps_taken,
        "ground_truth_cause": gt.get("cause", "?"),
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle E2E Validation for PostmortemEnv")
    parser.add_argument("--url", default="http://localhost:7860", help="Environment server URL")
    args = parser.parse_args()

    env_url = os.environ.get("ENV_URL", args.url)

    print("=" * 60)
    print("PostmortemEnv - Oracle End-to-End Validation")
    print(f"Environment: {env_url}")
    print("Model: oracle-solution (deterministic, no LLM needed)")
    print("=" * 60)
    print()

    # Health check
    try:
        import httpx
        health = httpx.get(f"{env_url}/health", timeout=10.0)
        print(f"Health check: {health.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to {env_url}: {e}")
        print("Start the server first: uvicorn app:app --host 0.0.0.0 --port 7860")
        print("Or run via Docker:      docker run -p 7860:7860 postmortemenv")
        sys.exit(1)

    print()
    results = {}
    tasks = get_available_tasks()

    for task_id in tasks:
        print(f"\n{'='*60}")
        try:
            result = run_oracle_task(env_url, task_id)
            results[task_id] = result
        except Exception as e:
            print(f"ERROR running {task_id}: {e}")
            import traceback
            traceback.print_exc()
            results[task_id] = {"score": 0.01, "success": False, "error": str(e)}

    # Final report
    print(f"\n\n{'='*60}")
    print("ORACLE VALIDATION RESULTS")
    print("=" * 60)

    all_passed = True
    for task_id, result in results.items():
        score = result.get("score", 0)
        status = "PASS" if result.get("success") else "FAIL"
        gt_cause = result.get("ground_truth_cause", "?")

        print(f"\n  {task_id}:")
        print(f"    Score:       {score:.3f} ({status})")
        print(f"    Steps:       {result.get('steps', 0)}")
        print(f"    Root Cause:  {gt_cause}")

        if not result.get("success"):
            all_passed = False

    avg = sum(r.get("score", 0) for r in results.values()) / len(results) if results else 0
    print(f"\n  Average Score: {avg:.3f}")
    print("=" * 60)

    if all_passed:
        print("\n  ALL TASKS PASSED - Environment is working correctly!")
        print("  Ready for competition submission.")
    else:
        print("\n  Some tasks did not reach 0.5 threshold.")
        print("  Check scores and oracle solutions above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
