"""Tests for the PostmortemEnvironment core logic."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.environment import PostmortemEnvironment
from models.action import Action, ActionType


@pytest.fixture
def env():
    """Create a fresh environment instance."""
    return PostmortemEnvironment()


@pytest.fixture
def easy_env(env):
    """Environment reset with easy task."""
    env.reset(task_id="task1_recent_deploy")
    return env


@pytest.fixture
def medium_env(env):
    """Environment reset with medium task."""
    env.reset(task_id="task2_cascade_chain")
    return env


@pytest.fixture
def hard_env(env):
    """Environment reset with hard task."""
    env.reset(task_id="task3_correlated_cause")
    return env


class TestReset:
    def test_reset_easy(self, env):
        obs = env.reset(task_id="task1_recent_deploy")
        assert obs.max_steps == 40
        assert obs.step_number == 0
        assert obs.remaining_budget == 40
        assert "data" in obs.service_graph
        assert len(obs.services) == 4
        assert len(obs.available_commits) > 0

    def test_reset_medium(self, env):
        obs = env.reset(task_id="task2_cascade_chain")
        assert obs.max_steps == 75
        assert obs.remaining_budget == 75

    def test_reset_hard(self, env):
        obs = env.reset(task_id="task3_correlated_cause")
        assert obs.max_steps == 120

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError):
            env.reset(task_id="nonexistent_task")

    def test_reset_clears_state(self, env):
        obs = env.reset(task_id="task1_recent_deploy")
        assert obs.known_facts == []
        assert obs.hypotheses_submitted == 0
        assert obs.wrong_hypotheses == 0

    def test_reward_in_valid_range(self, env):
        obs = env.reset(task_id="task1_recent_deploy")
        assert 0.0 < obs.reward < 1.0


class TestQueryLogs:
    def test_query_logs_with_results(self, easy_env):
        action = Action(action_type=ActionType.QUERY_LOGS, service="data", keyword="error")
        obs = easy_env.step(action)
        assert "error" in obs.query_result.lower() or "Error" in obs.query_result
        assert obs.step_number == 1
        assert obs.remaining_budget == 39

    def test_query_logs_invalid_service(self, easy_env):
        action = Action(action_type=ActionType.QUERY_LOGS, service="nonexistent", keyword="error")
        obs = easy_env.step(action)
        assert "Unknown service" in obs.message

    def test_query_logs_no_service(self, easy_env):
        action = Action(action_type=ActionType.QUERY_LOGS, keyword="error")
        obs = easy_env.step(action)
        assert "requires" in obs.message.lower()

    def test_query_logs_discovers_facts(self, easy_env):
        action = Action(action_type=ActionType.QUERY_LOGS, service="data", keyword="ConnectionPool")
        obs = easy_env.step(action)
        # Should discover relevant facts about pool exhaustion
        assert obs.step_number == 1


class TestFetchTrace:
    def test_fetch_valid_trace(self, easy_env):
        action = Action(action_type=ActionType.FETCH_TRACE, trace_id="trace-001")
        obs = easy_env.step(action)
        assert "trace-001" in obs.query_result.lower() or "Trace" in obs.query_result
        assert obs.step_number == 1

    def test_fetch_invalid_trace(self, easy_env):
        action = Action(action_type=ActionType.FETCH_TRACE, trace_id="trace-999")
        obs = easy_env.step(action)
        assert "not found" in obs.message.lower()

    def test_fetch_trace_no_id(self, easy_env):
        action = Action(action_type=ActionType.FETCH_TRACE)
        obs = easy_env.step(action)
        assert "requires" in obs.message.lower()


class TestDiffCommit:
    def test_diff_valid_commit(self, easy_env):
        action = Action(action_type=ActionType.DIFF_COMMIT, commit_hash="commit-a1b2c3")
        obs = easy_env.step(action)
        assert "commit-a1b2c3" in obs.query_result
        assert "diff" in obs.query_result.lower() or "Diff" in obs.query_result

    def test_diff_invalid_commit(self, easy_env):
        action = Action(action_type=ActionType.DIFF_COMMIT, commit_hash="commit-nonexistent")
        obs = easy_env.step(action)
        assert "not found" in obs.message.lower()

    def test_diff_discovers_relevant_commit(self, easy_env):
        # commit-a1b2c3 is the root cause and should be marked relevant
        action = Action(action_type=ActionType.DIFF_COMMIT, commit_hash="commit-a1b2c3")
        obs = easy_env.step(action)
        state = easy_env.state
        assert "commit-a1b2c3" in state.facts_discovered


class TestInspectConfig:
    def test_inspect_valid_config(self, easy_env):
        action = Action(action_type=ActionType.INSPECT_CONFIG, config_id="cfg-004")
        obs = easy_env.step(action)
        assert "cfg-004" in obs.query_result

    def test_inspect_invalid_config(self, easy_env):
        action = Action(action_type=ActionType.INSPECT_CONFIG, config_id="cfg-999")
        obs = easy_env.step(action)
        assert "not found" in obs.message.lower()


class TestHypothesize:
    def test_correct_hypothesis(self, easy_env):
        action = Action(action_type=ActionType.HYPOTHESIZE, cause_entity_id="commit-a1b2c3")
        obs = easy_env.step(action)
        assert "CORRECT" in obs.query_result
        assert obs.wrong_hypotheses == 0

    def test_wrong_hypothesis(self, easy_env):
        action = Action(action_type=ActionType.HYPOTHESIZE, cause_entity_id="commit-d4e5f6")
        obs = easy_env.step(action)
        assert "INCORRECT" in obs.query_result
        assert obs.wrong_hypotheses == 1

    def test_multiple_wrong_hypotheses(self, easy_env):
        for _ in range(3):
            action = Action(action_type=ActionType.HYPOTHESIZE, cause_entity_id="commit-wrong")
            obs = easy_env.step(action)
        assert obs.wrong_hypotheses == 3


class TestExplainChain:
    def test_perfect_chain(self, easy_env):
        chain = [
            {"service": "data", "effect": "connection_pool_exhaustion"},
            {"service": "auth", "effect": "upstream_timeout"},
            {"service": "frontend", "effect": "5xx_errors_to_users"},
        ]
        action = Action(action_type=ActionType.EXPLAIN_CHAIN, chain=chain)
        obs = easy_env.step(action)
        assert "similarity: 1.00" in obs.query_result

    def test_partial_chain(self, easy_env):
        chain = [
            {"service": "data", "effect": "connection_pool_exhaustion"},
            {"service": "frontend", "effect": "errors"},
        ]
        action = Action(action_type=ActionType.EXPLAIN_CHAIN, chain=chain)
        obs = easy_env.step(action)
        assert "similarity" in obs.query_result.lower()

    def test_wrong_chain(self, easy_env):
        chain = [
            {"service": "batch", "effect": "something_wrong"},
        ]
        action = Action(action_type=ActionType.EXPLAIN_CHAIN, chain=chain)
        obs = easy_env.step(action)
        # Similarity should be low
        assert "similarity" in obs.query_result.lower()


class TestSubmit:
    def test_submit_correct_answer(self, easy_env):
        chain = [
            {"service": "data", "effect": "connection_pool_exhaustion"},
            {"service": "auth", "effect": "upstream_timeout"},
            {"service": "frontend", "effect": "5xx_errors_to_users"},
        ]
        action = Action(
            action_type=ActionType.SUBMIT,
            final_cause="commit-a1b2c3",
            final_chain=chain,
        )
        obs = easy_env.step(action)
        assert obs.done is True
        state = easy_env.state
        assert state.final_score is not None
        assert state.final_score > 0.5  # correct answer should score well

    def test_submit_wrong_answer(self, easy_env):
        action = Action(
            action_type=ActionType.SUBMIT,
            final_cause="commit-wrong",
            final_chain=[{"service": "batch", "effect": "nothing"}],
        )
        obs = easy_env.step(action)
        assert obs.done is True
        state = easy_env.state
        assert state.final_score is not None
        assert state.final_score < 0.5  # wrong answer

    def test_submit_ends_episode(self, easy_env):
        action = Action(
            action_type=ActionType.SUBMIT,
            final_cause="commit-a1b2c3",
            final_chain=[],
        )
        obs = easy_env.step(action)
        assert obs.done is True

        # Should raise on second step
        with pytest.raises(RuntimeError):
            easy_env.step(action)


class TestBudgetExhaustion:
    def test_budget_ends_episode(self, env):
        obs = env.reset(task_id="task1_recent_deploy")
        max_steps = obs.max_steps

        # Use up all steps with queries
        for i in range(max_steps):
            action = Action(action_type=ActionType.QUERY_LOGS, service="data", keyword="info")
            obs = env.step(action)
            if obs.done:
                break

        assert obs.done is True


class TestState:
    def test_state_before_init(self, env):
        with pytest.raises(RuntimeError):
            _ = env.state

    def test_state_after_reset(self, easy_env):
        state = easy_env.state
        assert state.task_id == "task1_recent_deploy"
        assert state.task_difficulty == "easy"
        assert state.ground_truth_cause == "commit-a1b2c3"
        assert len(state.ground_truth_chain) == 3
        assert state.done is False

    def test_state_after_steps(self, easy_env):
        action = Action(action_type=ActionType.QUERY_LOGS, service="data", keyword="error")
        easy_env.step(action)
        state = easy_env.state
        assert state.steps_taken == 1


class TestRewardClamping:
    def test_rewards_always_in_range(self, easy_env):
        """Every observation reward must be strictly inside (0, 1)."""
        actions = [
            Action(action_type=ActionType.QUERY_LOGS, service="data", keyword="error"),
            Action(action_type=ActionType.FETCH_TRACE, trace_id="trace-001"),
            Action(action_type=ActionType.DIFF_COMMIT, commit_hash="commit-a1b2c3"),
            Action(action_type=ActionType.HYPOTHESIZE, cause_entity_id="commit-wrong"),
            Action(action_type=ActionType.HYPOTHESIZE, cause_entity_id="commit-a1b2c3"),
            Action(action_type=ActionType.SUBMIT, final_cause="commit-a1b2c3", final_chain=[]),
        ]
        for action in actions:
            obs = easy_env.step(action)
            assert 0.0 < obs.reward < 1.0, f"Reward {obs.reward} out of (0,1) range for {action.action_type}"
            if obs.done:
                break


class TestMediumTask:
    def test_cascade_root_cause(self, medium_env):
        state = medium_env.state
        assert state.ground_truth_cause == "commit-bat100"
        assert len(state.ground_truth_chain) == 4

    def test_correct_submission(self, medium_env):
        chain = [
            {"service": "batch", "effect": "oom_crash_loop"},
            {"service": "data", "effect": "batch_dependency_timeout"},
            {"service": "auth", "effect": "permissions_lookup_failure"},
            {"service": "frontend", "effect": "complete_service_unavailability"},
        ]
        action = Action(
            action_type=ActionType.SUBMIT,
            final_cause="commit-bat100",
            final_chain=chain,
        )
        obs = medium_env.step(action)
        assert obs.done is True
        state = medium_env.state
        assert state.final_score > 0.5


class TestHardTask:
    def test_correlated_root_cause(self, hard_env):
        state = hard_env.state
        assert state.ground_truth_cause_type == "correlated"
        assert "commit-auth200" in state.ground_truth_cause

    def test_correct_submission(self, hard_env):
        chain = [
            {"service": "data", "effect": "az_a_network_degradation"},
            {"service": "data", "effect": "az_b_endpoint_not_registered"},
            {"service": "auth", "effect": "failover_closes_all_connections"},
            {"service": "auth", "effect": "zero_connections_no_recovery"},
            {"service": "frontend", "effect": "complete_auth_failure"},
        ]
        action = Action(
            action_type=ActionType.SUBMIT,
            final_cause="commit-auth200+infra-202",
            final_chain=chain,
        )
        obs = hard_env.step(action)
        assert obs.done is True
        state = hard_env.state
        assert state.final_score > 0.5
