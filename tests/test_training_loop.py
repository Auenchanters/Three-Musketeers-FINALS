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
    _build_chain_candidates,
    _candidate_actions,
    _estimate_random_baseline,
    _featurize,
    _observed_effects,
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


def test_done_event_carries_per_rubric_breakdown() -> None:
    """The frontend reads rubric_breakdown to show *why* the score is what it is.

    The breakdown is the average over the final 20-ep window of each of the
    five rubrics (cause/chain/efficiency/investigation/anti_gaming). With
    chain mining enabled in ``_candidate_actions``, the chain_accuracy raw
    score can now be > 0 once the policy learns to prefer chain-bearing
    SUBMITs — the assertion below only requires it to be a valid number in
    [0, 1].
    """
    sess = LearningSession(
        session_id="rb",
        task_id="task1_recent_deploy",
        n_episodes=60,
        seed=42,
        runtime="test",
        policy_kind="neural",
    )
    asyncio.new_event_loop().run_until_complete(sess.run())
    done = next(m for m in sess.metrics if m.get("type") == "done")
    rb = done.get("rubric_breakdown")
    assert isinstance(rb, list) and len(rb) == 5, f"expected 5 rubrics, got {rb}"
    names = {r["rubric"] for r in rb}
    assert names == {
        "cause_correctness", "chain_accuracy", "efficiency",
        "investigation_quality", "anti_gaming",
    }, f"unexpected rubric names: {names}"
    weights = sum(r["weight"] for r in rb)
    assert abs(weights - 1.0) < 1e-6, f"weights don't sum to 1: {weights}"
    # The chain rubric must have a raw score (never None / missing).
    chain = next(r for r in rb if r["rubric"] == "chain_accuracy")
    assert "mean_raw_score" in chain
    assert 0.0 <= chain["mean_raw_score"] <= 1.0


def test_env_exposes_final_rubric_breakdown_after_submit() -> None:
    """The env getter populates after a submit, stays None before."""
    env = PostmortemEnvironment()
    env.reset(task_id="task1_recent_deploy")
    assert env.get_final_rubric_breakdown() is None, \
        "breakdown should be None before any submit"
    from models.action import Action, ActionType
    env.step(Action(action_type=ActionType.SUBMIT, final_cause="commit-a1b2c3", final_chain=[]))
    rb = env.get_final_rubric_breakdown()
    assert rb is not None and isinstance(rb, list) and len(rb) == 5


# --- Chain-mining regression suite ---------------------------------------
#
# These tests guard the fix that lets the live policy actually score
# ``chain_accuracy`` (25% of the total rubric weight). Before the fix the
# action menu hard-coded ``final_chain=[]`` for every SUBMIT candidate, so
# the maximum reachable score was 0.75 by construction and the live chart
# plateaued around 0.66. The fix mines (service, effect) pairs from the
# public log haystack — the same logs the agent reads via QUERY_LOGS — and
# uses them to populate chain-bearing SUBMITs in the action menu.


def test_observed_effects_extracts_known_chain_strings() -> None:
    """The hand-crafted task1 scenario's logs name the GT chain effects.

    Specifically: 'data' logs say "Connection pool ... exhausted", 'auth'
    logs say "504 Gateway Timeout" / "upstream", 'frontend' logs say "5xx"
    or "503 ... returned to client". The mined dict must contain those
    canonical effect strings on the matching service.
    """
    scenario = load_scenario("task1_recent_deploy")
    effects = _observed_effects(scenario)
    assert "connection_pool_exhaustion" in effects.get("data", []) or \
           "connection_pool_exhaustion" in effects.get("auth", []), \
        f"connection_pool_exhaustion not surfaced anywhere: {effects}"
    assert "upstream_timeout" in effects.get("auth", []) or \
           "upstream_timeout" in effects.get("frontend", []), \
        f"upstream_timeout not surfaced anywhere: {effects}"
    # frontend should pick up 5xx/503 messaging
    fe = effects.get("frontend", [])
    assert "5xx_errors_to_users" in fe, f"expected 5xx in frontend, got {fe}"


def test_observed_effects_does_not_read_ground_truth_chain() -> None:
    """Helper must work on a scenario dict with no ``chain`` field at all.

    The ``chain`` field is GT and must never leak into the policy's input.
    Stripping it from a clone of the scenario should not affect the mined
    effect set in any way.
    """
    scenario = load_scenario("task1_recent_deploy")
    stripped = {k: v for k, v in scenario.items() if k != "chain"}
    assert "chain" not in stripped
    full_effects = _observed_effects(scenario)
    stripped_effects = _observed_effects(stripped)
    assert full_effects == stripped_effects, (
        "observed_effects depends on scenario['chain'] — that's the GT. "
        f"full={full_effects} stripped={stripped_effects}"
    )


def test_train_blocking_clears_0_70_with_chain_candidates() -> None:
    """Headline regression: with chain mining the live policy clears 0.85.

    Before the fix, ``train_blocking(task1, 500)`` plateaued around 0.66
    because the policy submitted ``final_chain=[]`` (chain rubric = 0.0
    → mathematically capped at 0.75 total). The chain-bearing SUBMIT
    candidates introduced by ``_build_chain_candidates`` let the policy
    score chain_accuracy partially via the value baseline. With adaptive
    reward boosting (kicks in after 50 episodes of stagnation) and 1000
    episodes, the policy now reliably clears 0.85.
    """
    summary = train_blocking(
        task_id="task1_recent_deploy", n_episodes=1000, seed=42, policy_kind="neural"
    )
    assert summary["status"] == "done"
    assert summary["final_mean"] is not None
    assert summary["final_mean"] > 0.85, (
        f"chain mining didn't lift past 0.85: final_mean={summary['final_mean']:.3f}, "
        f"random={summary['random_baseline']:.3f}, lift={summary['lift_over_random']:.3f}. "
        "Either _observed_effects regressed or _build_chain_candidates produced "
        "no chain-bearing SUBMITs for task1."
    )


def test_action_menu_stays_under_200_on_largest_task() -> None:
    """task4 is the 12-node failover scenario; chain candidates × cause
    candidates can multiply quickly. Cap the menu so 500-episode REINFORCE
    still has every action sampled enough times to learn.
    """
    scenario = load_scenario("task4_multi_region_failover")
    menu = _candidate_actions(scenario)
    assert 0 < len(menu) < 200, (
        f"action menu blew up to {len(menu)} on task4 — chain candidates × cause "
        "candidates is too aggressive. Tighten CHAIN_K or per-service effect cap."
    )


def test_build_chain_candidates_returns_2_to_3_node_chains() -> None:
    """Chain shapes should match seed_generator's typical 3-node templates.

    Empty input → empty output (graceful on scenarios with no log signal).
    Otherwise every chain has 1-3 nodes (we lead with a single failing
    service, then append up to 2 dependents).
    """
    assert _build_chain_candidates({}, {}) == []
    scenario = load_scenario("task1_recent_deploy")
    chains = _build_chain_candidates(scenario, _observed_effects(scenario))
    assert chains, "expected at least one chain candidate for task1"
    for c in chains:
        assert 1 <= len(c) <= 3, f"chain length out of range: {c}"
        for node in c:
            assert "service" in node and "effect" in node
