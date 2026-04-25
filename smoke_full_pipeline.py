"""End-to-end smoke for the live training panel.

Verifies the in-app REINFORCE loop produces a meaningful lift over the random
baseline on every default task within ~30 seconds total. Exits non-zero if
any task fails the lift threshold.

Run with: ``py -3.10 smoke_full_pipeline.py``
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from web.training_loop import LearningSession

# Tasks chosen to span all five difficulty/topology shapes.
TASKS = [
    "task1_recent_deploy",
    "task2_cascade_chain",
    "task3_correlated_cause",
    "task4_multi_region_failover",
    "task5_data_corruption_cascade",
]

EPISODES = 500          # matches the UI default ("visibly upward in ~500 eps")
MIN_LIFT = 0.05         # at least +5% absolute over the random baseline
TIMEOUT_S = 60          # per-task safety net


async def _train_one(task_id: str, seed: int = 42) -> dict:
    sess = LearningSession(
        session_id=uuid.uuid4().hex[:12],
        task_id=task_id,
        n_episodes=EPISODES,
        seed=seed,
        runtime="test",
    )
    t0 = time.time()
    last_metric: dict | None = None
    last_baseline: float | None = None
    done_event: dict | None = None
    await sess.run()
    for ev in sess.metrics:
        t = ev.get("type")
        if t == "baselines":
            last_baseline = ev.get("random")
        elif t in ("metric", "done"):
            last_metric = ev
        if t == "done":
            done_event = ev
    dt = time.time() - t0
    assert last_metric is not None, f"no metrics from {task_id}"
    rolling = (
        last_metric.get("rolling_mean")
        or last_metric.get("final_rolling_mean")
    )
    out = {
        "task_id": task_id,
        "seed": seed,
        "elapsed_s": round(dt, 1),
        "episode": last_metric.get("episode") or last_metric.get("n_episodes"),
        "rolling_mean": rolling,
        "baseline": last_baseline,
        "lift": last_metric.get("lift_over_random"),
    }
    # Attach the per-rubric breakdown if the done event had it (always
    # true for the neural policy now, optional for backward-compat).
    if done_event and "rubric_breakdown" in done_event:
        out["rubric_breakdown"] = done_event["rubric_breakdown"]
    return out


async def main() -> int:
    results = []
    for tid in TASKS:
        print(f"--- training {tid}")
        r = await _train_one(tid)
        results.append(r)
        ok = (r["lift"] is not None) and (r["lift"] >= MIN_LIFT)
        flag = "OK " if ok else "WARN"
        print(
            f"  {flag} ep={r['episode']:>4} "
            f"trained={r['rolling_mean']:.3f} "
            f"baseline={r['baseline']:.3f} "
            f"lift=+{r['lift']:.3f} "
            f"in {r['elapsed_s']:.1f}s"
        )
    out = Path("training_data") / "live_training_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults written to {out}")
    failures = [r for r in results if (r["lift"] or 0) < MIN_LIFT]
    if failures:
        print(f"\nFAILED ({len(failures)}/{len(results)} below threshold)")
        return 1
    print(f"\nALL PASS ({len(results)}/{len(results)} above +{MIN_LIFT:.2f} lift)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
