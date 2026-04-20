"""Tests for the deterministic oracle grader."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.grader import Grader, _normalized_edit_distance


class TestNormalizedEditDistance:
    def test_identical_chains(self):
        chain = [
            {"service": "data", "effect": "pool_exhaustion"},
            {"service": "auth", "effect": "timeout"},
        ]
        assert _normalized_edit_distance(chain, chain) == 0.0

    def test_completely_different(self):
        pred = [{"service": "data", "effect": "crash"}]
        actual = [{"service": "auth", "effect": "timeout"}]
        assert _normalized_edit_distance(pred, actual) == 1.0

    def test_empty_predicted(self):
        actual = [{"service": "data", "effect": "crash"}]
        assert _normalized_edit_distance([], actual) == 1.0

    def test_empty_actual(self):
        pred = [{"service": "data", "effect": "crash"}]
        assert _normalized_edit_distance(pred, []) == 1.0

    def test_both_empty(self):
        assert _normalized_edit_distance([], []) == 0.0

    def test_partial_match(self):
        pred = [
            {"service": "data", "effect": "pool_exhaustion"},
            {"service": "frontend", "effect": "5xx"},
        ]
        actual = [
            {"service": "data", "effect": "pool_exhaustion"},
            {"service": "auth", "effect": "timeout"},
            {"service": "frontend", "effect": "5xx"},
        ]
        dist = _normalized_edit_distance(pred, actual)
        assert 0.0 < dist < 1.0

    def test_length_mismatch(self):
        pred = [{"service": "a", "effect": "1"}]
        actual = [
            {"service": "a", "effect": "1"},
            {"service": "b", "effect": "2"},
            {"service": "c", "effect": "3"},
        ]
        dist = _normalized_edit_distance(pred, actual)
        # Edit distance = 2 insertions, max_len = 3
        assert abs(dist - 2/3) < 0.01


class TestCauseMatch:
    def test_exact_match(self):
        assert Grader.check_cause_match("commit-abc", "commit-abc") is True

    def test_case_insensitive(self):
        assert Grader.check_cause_match("COMMIT-ABC", "commit-abc") is True

    def test_no_match(self):
        assert Grader.check_cause_match("commit-abc", "commit-xyz") is False

    def test_empty_submitted(self):
        assert Grader.check_cause_match("", "commit-abc") is False

    def test_correlated_exact_match(self):
        assert Grader.check_cause_match(
            "commit-abc+infra-xyz",
            "commit-abc+infra-xyz",
            cause_type="correlated",
        ) is True

    def test_correlated_contributing_cause(self):
        assert Grader.check_cause_match(
            "commit-abc",
            "commit-abc+infra-xyz",
            cause_type="correlated",
            contributing_causes=["commit-abc", "infra-xyz"],
        ) is True

    def test_correlated_wrong_cause(self):
        assert Grader.check_cause_match(
            "commit-wrong",
            "commit-abc+infra-xyz",
            cause_type="correlated",
            contributing_causes=["commit-abc", "infra-xyz"],
        ) is False


class TestChainSimilarity:
    def test_perfect_match(self):
        chain = [
            {"service": "data", "effect": "pool"},
            {"service": "auth", "effect": "timeout"},
        ]
        assert Grader.compute_chain_similarity(chain, chain) == 1.0

    def test_none_predicted(self):
        actual = [{"service": "data", "effect": "pool"}]
        assert Grader.compute_chain_similarity(None, actual) == 0.0

    def test_empty_predicted(self):
        actual = [{"service": "data", "effect": "pool"}]
        sim = Grader.compute_chain_similarity([], actual)
        assert sim == 0.0


class TestFinalScore:
    def test_perfect_score(self):
        """Correct cause + perfect chain + immediate submit = high score."""
        chain = [{"service": "data", "effect": "crash"}]
        score = Grader.compute_final_score(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=1,
            max_steps=40,
            n_wrong_hypotheses=0,
        )
        assert score > 0.9

    def test_wrong_cause_low_score(self):
        """Wrong cause = at most 0.5 from chain + efficiency."""
        chain = [{"service": "data", "effect": "crash"}]
        score = Grader.compute_final_score(
            submitted_cause="commit-wrong",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=1,
            max_steps=40,
            n_wrong_hypotheses=0,
        )
        assert score < 0.5

    def test_hypothesis_penalties(self):
        """Wrong hypotheses should reduce score."""
        chain = [{"service": "data", "effect": "crash"}]
        score_clean = Grader.compute_final_score(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=5,
            max_steps=40,
            n_wrong_hypotheses=0,
        )
        score_dirty = Grader.compute_final_score(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=5,
            max_steps=40,
            n_wrong_hypotheses=3,
        )
        assert score_clean > score_dirty

    def test_efficiency_matters(self):
        """Using fewer steps should give a higher score."""
        chain = [{"service": "data", "effect": "crash"}]
        score_fast = Grader.compute_final_score(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=5,
            max_steps=40,
            n_wrong_hypotheses=0,
        )
        score_slow = Grader.compute_final_score(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=35,
            max_steps=40,
            n_wrong_hypotheses=0,
        )
        assert score_fast > score_slow

    def test_score_always_clamped(self):
        """Score must always be in (0.01, 0.99)."""
        # Very bad submission
        score = Grader.compute_final_score(
            submitted_cause="",
            submitted_chain=None,
            ground_truth_cause="commit-abc",
            ground_truth_chain=[{"service": "data", "effect": "crash"}],
            cause_type="commit",
            steps_used=40,
            max_steps=40,
            n_wrong_hypotheses=10,
        )
        assert 0.01 <= score <= 0.99

    def test_correlated_partial_credit(self):
        """For correlated causes, identifying one contributing cause gets partial credit."""
        chain = [{"service": "data", "effect": "crash"}]
        score = Grader.compute_final_score(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc+infra-xyz",
            ground_truth_chain=chain,
            cause_type="correlated",
            steps_used=5,
            max_steps=120,
            n_wrong_hypotheses=0,
            contributing_causes=["commit-abc", "infra-xyz"],
        )
        # Should get partial credit (0.5 * 0.5 = 0.25 from cause)
        assert score > 0.3
