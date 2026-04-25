"""
Tests for ``web/training_loop.py`` — the live REINFORCE learner that powers
the frontend's "Live training" panel.

These tests are deliberately fast (≤300 episodes per task, ``runtime="test"``
to skip the per-episode UI sleep) and only assert structural properties of
the output, not exact numerical scores. The point is to catch regressions
in:

1. Event shape (the SSE stream contract on the backend depends on this).
2. The action menu being non-empty for every shipped task.
3. The chart being able to draw something — i.e. ``random_baseline`` is set
   and at least one metric event is emitted.

We *don't* assert "agent learns +X" because REINFORCE's seed sensitivity
on small budgets makes such a test flaky and a spurious failure here would
block frontend dev. Visible-learning convergence is verified manually by
running the chart in the browser, where the cadence + 500-episode default
make the curve unmistakably climb above the random baseline.
"""

from __future__ import annotations

import asyncio

from web.training_loop import (
    LearningSession,
    _candidate_actions,
    _estimate_random_baseline,
    train_blocking,
)
from data.generator import load_scenario
from engine.environment import PostmortemEnvironment


def test_action_menu_nonempty_for_all_shipped_tasks() -> None:
    for tid in (
        "task1_recent_deploy",
        "task2_cascade_chain",
        "task3_correlated_cause",
        "task4_multi_region_failover",
        "task5_data_corruption_cascade",
    ):
        scen = load_scenario(tid)
        menu = _candidate_actions(scen)
        assert menu, f"empty menu for {tid}"
        # We trim pair candidates to the 2 most recent commits so the menu
        # stays under ~150 actions — much larger and 500-episode REINFORCE
        # can't get enough samples per action for the chart to climb.
        assert len(menu) <= 200, f"menu too large for {tid}: {len(menu)}"


def test_train_blocking_returns_well_formed_summary() -> None:
    summary = train_blocking(task_id="task1_recent_deploy", n_episodes=40, seed=42)
    assert summary["status"] == "done"
    assert summary["n_metrics"] == 40
    assert summary["random_baseline"] is not None
    assert 0.0 <= summary["random_baseline"] <= 1.0
    assert summary["final_mean"] is not None
    assert 0.0 <= summary["final_mean"] <= 1.0


def test_session_emits_full_sse_event_sequence() -> None:
    """The frontend relies on `baselines` → `metric*` → `done` ordering."""
    sess = LearningSession(
        session_id="t",
        task_id="task1_recent_deploy",
        n_episodes=20,
        seed=1,
        runtime="test",
    )
    asyncio.new_event_loop().run_until_complete(sess.run())

    types = [m.get("type") for m in sess.metrics]
    assert types[0] == "baselines"
    assert types[-1] == "done"
    # Every middle event is a per-episode metric.
    middle = types[1:-1]
    assert all(t == "metric" for t in middle), f"unexpected middle types: {middle}"
    assert len(middle) == 20

    # Episode counter is monotonic and dense.
    eps = [m["episode"] for m in sess.metrics if m.get("type") == "metric"]
    assert eps == list(range(1, 21))


def test_random_baseline_estimator_is_in_unit_interval() -> None:
    env = PostmortemEnvironment()
    for tid in ("task1_recent_deploy", "task5_data_corruption_cascade"):
        b = _estimate_random_baseline(env, tid, n=4)
        assert 0.0 <= b <= 1.0, f"{tid}: random baseline {b} outside [0,1]"
