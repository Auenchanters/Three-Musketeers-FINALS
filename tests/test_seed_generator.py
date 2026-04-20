"""Tests for the procedural scenario seed generator."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.seed_generator import generate_scenario, generate_oracle_solution


class TestGenerateScenario:
    def test_basic_generation(self):
        scenario = generate_scenario(42, "easy")
        assert scenario["task_difficulty"] == "easy"
        assert scenario["max_steps"] == 40
        assert "service_graph" in scenario
        assert "logs" in scenario
        assert "ground_truth" in scenario

    def test_deterministic(self):
        """Same seed + difficulty = same scenario."""
        s1 = generate_scenario(42, "easy")
        s2 = generate_scenario(42, "easy")
        assert s1["task_id"] == s2["task_id"]
        assert s1["ground_truth"]["cause"] == s2["ground_truth"]["cause"]
        assert s1["ground_truth"]["chain"] == s2["ground_truth"]["chain"]

    def test_different_seeds_different_scenarios(self):
        s1 = generate_scenario(42, "easy")
        s2 = generate_scenario(99, "easy")
        # Should have different root causes (very likely)
        assert s1["task_id"] != s2["task_id"]

    def test_easy_generation(self):
        scenario = generate_scenario(100, "easy")
        assert scenario["max_steps"] == 40
        assert len(scenario["services"]) == 4
        assert len(scenario["ground_truth"]["chain"]) >= 2

    def test_medium_generation(self):
        scenario = generate_scenario(100, "medium")
        assert scenario["max_steps"] == 75

    def test_hard_generation(self):
        scenario = generate_scenario(100, "hard")
        assert scenario["max_steps"] == 120

    def test_has_four_services(self):
        for seed in [1, 42, 99, 200]:
            scenario = generate_scenario(seed, "easy")
            assert len(scenario["service_graph"]) == 4
            assert "frontend" in scenario["service_graph"]
            assert "auth" in scenario["service_graph"]
            assert "data" in scenario["service_graph"]
            assert "batch" in scenario["service_graph"]

    def test_has_logs_for_all_services(self):
        scenario = generate_scenario(42, "medium")
        for svc in ["frontend", "auth", "data", "batch"]:
            assert svc in scenario["logs"]
            assert len(scenario["logs"][svc]) > 0

    def test_has_relevant_facts(self):
        scenario = generate_scenario(42, "easy")
        assert len(scenario["relevant_fact_ids"]) > 0

    def test_ground_truth_cause_exists(self):
        for seed in range(10):
            scenario = generate_scenario(seed, "easy")
            cause = scenario["ground_truth"]["cause"]
            assert cause, "Ground truth cause should not be empty"

    def test_many_seeds_stable(self):
        """Generate 50 scenarios and ensure none crash."""
        for seed in range(50):
            for diff in ["easy", "medium", "hard"]:
                scenario = generate_scenario(seed, diff)
                assert scenario["task_id"]
                assert scenario["ground_truth"]["cause"]
                assert len(scenario["ground_truth"]["chain"]) > 0


class TestGenerateOracleSolution:
    def test_oracle_solution_valid(self):
        scenario = generate_scenario(42, "easy")
        solution = generate_oracle_solution(scenario, 42)
        assert solution["task_id"] == scenario["task_id"]
        assert "optimal_action_sequence" in solution
        assert len(solution["optimal_action_sequence"]) > 0

    def test_oracle_ends_with_submit(self):
        for seed in range(10):
            scenario = generate_scenario(seed, "easy")
            solution = generate_oracle_solution(scenario, seed)
            last = solution["optimal_action_sequence"][-1]
            assert last["action_type"] == "submit"

    def test_oracle_cause_matches_ground_truth(self):
        for seed in range(10):
            scenario = generate_scenario(seed, "medium")
            solution = generate_oracle_solution(scenario, seed)
            submit_action = solution["optimal_action_sequence"][-1]
            assert submit_action["final_cause"] == scenario["ground_truth"]["cause"]


class TestSeedGeneratorWithEnvironment:
    """Integration tests running generated scenarios through the real environment."""

    def test_env_accepts_generated_scenario(self):
        from engine.environment import PostmortemEnvironment
        from models.action import Action

        env = PostmortemEnvironment()
        scenario = generate_scenario(42, "easy")
        solution = generate_oracle_solution(scenario, 42)

        # Reset with on-the-fly task ID
        obs = env.reset(task_id=scenario["task_id"])
        assert obs.max_steps == scenario["max_steps"]

        # Run oracle solution
        for step_data in solution["optimal_action_sequence"]:
            action = Action(
                action_type=step_data["action_type"],
                service=step_data.get("service"),
                keyword=step_data.get("keyword"),
                trace_id=step_data.get("trace_id"),
                commit_hash=step_data.get("commit_hash"),
                config_id=step_data.get("config_id"),
                cause_entity_id=step_data.get("cause_entity_id"),
                chain=step_data.get("chain"),
                final_cause=step_data.get("final_cause"),
                final_chain=step_data.get("final_chain"),
                reason=step_data.get("reason"),
            )
            obs = env.step(action)
            if obs.done:
                break

        assert obs.done
        state = env.state
        assert state.final_score is not None
        assert state.final_score > 0.5

    def test_multiple_seeds_through_env(self):
        from engine.environment import PostmortemEnvironment
        from models.action import Action

        env = PostmortemEnvironment()
        scores = []

        for seed in range(5):
            for diff in ["easy", "medium"]:
                scenario = generate_scenario(seed, diff)
                solution = generate_oracle_solution(scenario, seed)

                obs = env.reset(task_id=scenario["task_id"])
                for step_data in solution["optimal_action_sequence"]:
                    action = Action(
                        action_type=step_data["action_type"],
                        service=step_data.get("service"),
                        keyword=step_data.get("keyword"),
                        trace_id=step_data.get("trace_id"),
                        commit_hash=step_data.get("commit_hash"),
                        config_id=step_data.get("config_id"),
                        cause_entity_id=step_data.get("cause_entity_id"),
                        chain=step_data.get("chain"),
                        final_cause=step_data.get("final_cause"),
                        final_chain=step_data.get("final_chain"),
                        reason=step_data.get("reason"),
                    )
                    obs = env.step(action)
                    if obs.done:
                        break

                score = env.state.final_score or 0.01
                scores.append(score)
                assert score > 0.5, f"Seed {seed} {diff} scored {score}"

        # All oracle runs should score well
        avg = sum(scores) / len(scores)
        assert avg > 0.8, f"Average oracle score {avg} too low"
