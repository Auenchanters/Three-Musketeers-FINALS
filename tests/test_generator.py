"""Tests for the data loader (generator.py)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import load_scenario, load_solution, get_available_tasks, get_ground_truth


class TestGetAvailableTasks:
    def test_returns_three_tasks(self):
        tasks = get_available_tasks()
        assert len(tasks) == 3
        assert "task1_recent_deploy" in tasks
        assert "task2_cascade_chain" in tasks
        assert "task3_correlated_cause" in tasks


class TestLoadScenario:
    def test_load_easy(self):
        scenario = load_scenario("task1_recent_deploy")
        assert scenario["task_id"] == "task1_recent_deploy"
        assert scenario["task_difficulty"] == "easy"
        assert scenario["max_steps"] == 40
        assert "service_graph" in scenario
        assert "logs" in scenario
        assert "traces" in scenario
        assert "commits" in scenario
        assert "ground_truth" in scenario

    def test_load_medium(self):
        scenario = load_scenario("task2_cascade_chain")
        assert scenario["task_difficulty"] == "medium"
        assert scenario["max_steps"] == 75

    def test_load_hard(self):
        scenario = load_scenario("task3_correlated_cause")
        assert scenario["task_difficulty"] == "hard"
        assert scenario["max_steps"] == 120

    def test_load_invalid_task(self):
        with pytest.raises(ValueError):
            load_scenario("nonexistent_task")

    def test_service_graph_has_four_services(self):
        for task_id in get_available_tasks():
            scenario = load_scenario(task_id)
            graph = scenario["service_graph"]
            assert len(graph) == 4
            assert "frontend" in graph
            assert "auth" in graph
            assert "data" in graph
            assert "batch" in graph

    def test_has_ground_truth(self):
        for task_id in get_available_tasks():
            scenario = load_scenario(task_id)
            gt = scenario["ground_truth"]
            assert "cause" in gt
            assert "chain" in gt
            assert len(gt["chain"]) > 0

    def test_has_relevant_facts(self):
        for task_id in get_available_tasks():
            scenario = load_scenario(task_id)
            facts = scenario.get("relevant_fact_ids", [])
            assert len(facts) > 0

    def test_logs_contain_relevant_entries(self):
        """At least some log entries should be marked relevant."""
        for task_id in get_available_tasks():
            scenario = load_scenario(task_id)
            relevant_count = 0
            for service, logs in scenario["logs"].items():
                for log in logs:
                    if log.get("relevant", False):
                        relevant_count += 1
            assert relevant_count > 0, f"No relevant logs in {task_id}"


class TestLoadSolution:
    def test_load_easy_solution(self):
        solution = load_solution("task1_recent_deploy")
        assert solution["task_id"] == "task1_recent_deploy"
        assert "ground_truth" in solution
        assert "optimal_action_sequence" in solution
        assert len(solution["optimal_action_sequence"]) > 0

    def test_load_medium_solution(self):
        solution = load_solution("task2_cascade_chain")
        assert len(solution["optimal_action_sequence"]) > 0

    def test_load_hard_solution(self):
        solution = load_solution("task3_correlated_cause")
        assert len(solution["optimal_action_sequence"]) > 0

    def test_load_invalid_solution(self):
        with pytest.raises(ValueError):
            load_solution("nonexistent_task")

    def test_solution_ends_with_submit(self):
        """Every oracle solution should end with a submit action."""
        for task_id in get_available_tasks():
            solution = load_solution(task_id)
            last_action = solution["optimal_action_sequence"][-1]
            assert last_action["action_type"] == "submit"

    def test_solution_cause_matches_scenario(self):
        """The solution's ground truth should match the scenario's ground truth."""
        for task_id in get_available_tasks():
            scenario = load_scenario(task_id)
            solution = load_solution(task_id)
            assert scenario["ground_truth"]["cause"] == solution["ground_truth"]["cause"]


class TestGetGroundTruth:
    def test_returns_ground_truth(self):
        gt = get_ground_truth("task1_recent_deploy")
        assert gt["cause"] == "commit-a1b2c3"
        assert gt["cause_type"] == "commit"
        assert len(gt["chain"]) == 3

    def test_hard_task_correlated(self):
        gt = get_ground_truth("task3_correlated_cause")
        assert gt["cause_type"] == "correlated"
        assert "contributing_causes" in gt
