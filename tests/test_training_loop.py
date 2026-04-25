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

import random as _random
import numpy as _np

from web.training_loop import (
    FEATURE_DIM,
    LearningSession,
    _NeuralPolicy,
    _candidate_actions,
    _estimate_random_baseline,
    _featurize,
    _service_index_table,
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


# --- Neural policy ----------------------------------------------------------


def test_featurize_yields_unit_vector_of_expected_dim() -> None:
    """Featurizer must produce a fixed-dim vector with values in [0, 1]."""
    env = PostmortemEnvironment()
    env.reset(task_id="task1_recent_deploy")
    scen = load_scenario("task1_recent_deploy")
    menu = _candidate_actions(scen)
    svc_index = _service_index_table(menu)

    f = _featurize(env, action_history={}, services_queried=set(),
                   service_index=svc_index, max_episode_steps=24)
    assert f.shape == (FEATURE_DIM,), f"unexpected shape {f.shape}"
    assert _np.all(f >= 0.0) and _np.all(f <= 1.0), \
        f"feature values out of [0,1]: min={f.min()} max={f.max()}"
    assert f[0] == 1.0, "bias feature must be 1.0"


def test_neural_policy_initial_distribution_is_uniform() -> None:
    """Zero-init weights must yield a uniform softmax (matches random baseline)."""
    rng = _random.Random(0)
    policy = _NeuralPolicy(n_actions=10, rng=rng)
    f = _np.ones(FEATURE_DIM, dtype=_np.float64)
    p = policy.probs(f)
    assert p.shape == (10,)
    expected = 1.0 / 10.0
    assert _np.allclose(p, expected, atol=1e-9), \
        f"initial policy non-uniform: {p}"


def test_neural_policy_update_changes_logits_in_advantage_direction() -> None:
    """Positive advantage on action a should increase p(a) at the same state."""
    rng = _random.Random(1)
    policy = _NeuralPolicy(n_actions=4, rng=rng)
    f = _np.zeros(FEATURE_DIM, dtype=_np.float64)
    f[0] = 1.0  # bias-only state

    p_before = policy.probs(f).copy()
    trajectory = [(f, 2)]  # picked action 2
    policy.update(trajectory, terminal_return=1.0,
                  lr_policy=0.5, lr_value=0.0, entropy_coef=0.0)
    p_after = policy.probs(f)

    assert p_after[2] > p_before[2], \
        f"p(a=2) did not increase: before={p_before} after={p_after}"
    # Other actions should have decreased (probabilities sum to 1).
    for j in (0, 1, 3):
        assert p_after[j] < p_before[j], \
            f"p(a={j}) did not decrease: before={p_before} after={p_after}"


def test_neural_session_emits_neural_policy_kind_in_baselines() -> None:
    """The frontend reads policy_kind from the baselines event for the badge."""
    sess = LearningSession(
        session_id="nt",
        task_id="task1_recent_deploy",
        n_episodes=10,
        seed=7,
        runtime="test",
        policy_kind="neural",
    )
    asyncio.new_event_loop().run_until_complete(sess.run())
    baselines = next(m for m in sess.metrics if m.get("type") == "baselines")
    assert baselines.get("policy_kind") == "neural"


def test_neural_train_blocking_lifts_above_random_on_easy_task() -> None:
    """End-to-end smoke: 500 episodes of neural REINFORCE must beat random.

    Uses task1 (the easiest single-cause scenario) with a fixed seed so this
    test is deterministic. Threshold is set well below the typical lift
    (~+0.38) to leave headroom for env tweaks without making the test flaky.
    """
    summary = train_blocking(
        task_id="task1_recent_deploy", n_episodes=500, seed=42, policy_kind="neural"
    )
    assert summary["status"] == "done"
    assert summary["n_metrics"] == 500
    assert summary["lift_over_random"] is not None
    assert summary["lift_over_random"] > 0.20, (
        f"neural policy failed to lift on task1: lift={summary['lift_over_random']:.3f}, "
        f"final={summary['final_mean']:.3f}, random={summary['random_baseline']:.3f}"
    )


def test_tabular_path_still_works_via_kwarg() -> None:
    """Backward-compat: the original tabular policy is one keyword away."""
    summary = train_blocking(
        task_id="task1_recent_deploy", n_episodes=80, seed=42, policy_kind="tabular"
    )
    assert summary["status"] == "done"
    assert summary["n_metrics"] == 80
