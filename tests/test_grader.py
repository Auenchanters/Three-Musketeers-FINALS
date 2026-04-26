"""Tests for the deterministic oracle grader."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.grader import Grader, _compute_graph_similarity


class TestGraphSimilarity:
    def test_identical_chains(self):
        chain = [
            {"service": "data", "effect": "pool_exhaustion"},
            {"service": "auth", "effect": "timeout"},
        ]
        assert _compute_graph_similarity(chain, chain) == 1.0

    def test_completely_different(self):
        pred = [{"service": "data", "effect": "crash"}]
        actual = [{"service": "auth", "effect": "timeout"}]
        # Single-element chains have no edges, so edge_sim = 1.0 (empty sets).
        # Node overlap is 0/2 = 0.0. Total = 0.4*0.0 + 0.6*1.0 = 0.6
        assert _compute_graph_similarity(pred, actual) == pytest.approx(0.6)

    def test_empty_predicted(self):
        actual = [{"service": "data", "effect": "crash"}]
        assert _compute_graph_similarity([], actual) == 0.0

    def test_empty_actual(self):
        pred = [{"service": "data", "effect": "crash"}]
        assert _compute_graph_similarity(pred, []) == 0.0

    def test_both_empty(self):
        assert _compute_graph_similarity([], []) == 1.0

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
        sim = _compute_graph_similarity(pred, actual)
        assert 0.0 < sim < 1.0

    def test_length_mismatch(self):
        pred = [{"service": "a", "effect": "1"}]
        actual = [
            {"service": "a", "effect": "1"},
            {"service": "b", "effect": "2"},
            {"service": "c", "effect": "3"},
        ]
        sim = _compute_graph_similarity(pred, actual)
        # Partial node overlap + no edge overlap → low but nonzero
        assert 0.0 < sim < 0.5


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

    def test_correlated_contributing_cause_is_not_full_match(self):
        # Naming a single contributing cause is NOT a full match —
        # it routes to the partial-credit branch in compute_final_score.
        assert Grader.check_cause_match(
            "commit-abc",
            "commit-abc+infra-xyz",
            cause_type="correlated",
            contributing_causes=["commit-abc", "infra-xyz"],
        ) is False

    def test_correlated_wrong_cause(self):
        assert Grader.check_cause_match(
            "commit-wrong",
            "commit-abc+infra-xyz",
            cause_type="correlated",
            contributing_causes=["commit-abc", "infra-xyz"],
        ) is False


class TestContributingCauseMatch:
    def test_identifies_single_contributing_cause(self):
        assert Grader.is_contributing_cause_match(
            "commit-abc",
            cause_type="correlated",
            contributing_causes=["commit-abc", "infra-xyz"],
        ) is True

    def test_full_combined_id_is_not_contributing_only(self):
        # is_contributing_cause_match ignores exact-match cases — those
        # are handled by check_cause_match.
        assert Grader.is_contributing_cause_match(
            "commit-abc+infra-xyz",
            cause_type="correlated",
            contributing_causes=["commit-abc", "infra-xyz"],
        ) is False

    def test_non_correlated_returns_false(self):
        assert Grader.is_contributing_cause_match(
            "commit-abc",
            cause_type="commit",
            contributing_causes=["commit-abc"],
        ) is False

    def test_unknown_contributing_returns_false(self):
        assert Grader.is_contributing_cause_match(
            "commit-other",
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
        assert score >= 0.85

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


class TestAntiGamingHypothesisAttempts:
    """Brutal-rating R5 fix: ``hypothesis_attempts > 3`` must penalise.

    The env now passes ``_hypothesis_call_count`` (which keeps incrementing
    past the 3-attempt cap) into the grader, so spamming hypothesize()
    actually hurts the terminal score.
    """

    def _baseline_kwargs(self):
        chain = [{"service": "data", "effect": "crash"}]
        return dict(
            submitted_cause="commit-abc",
            submitted_chain=chain,
            ground_truth_cause="commit-abc",
            ground_truth_chain=chain,
            cause_type="commit",
            steps_used=5,
            max_steps=40,
            n_wrong_hypotheses=0,
        )

    def test_hypothesis_spam_penalises_score(self):
        """Score must strictly decrease as hypothesis_attempts goes 3 -> 6."""
        kw = self._baseline_kwargs()
        score_clean = Grader.compute_final_score(**kw, hypothesis_attempts=3)
        score_spam = Grader.compute_final_score(**kw, hypothesis_attempts=6)
        assert score_spam < score_clean, (
            "hypothesis_attempts > 3 should reduce the anti_gaming rubric "
            "and therefore the total score"
        )

    def test_hypothesis_attempts_3_or_less_is_neutral(self):
        """The penalty branch only fires when attempts strictly exceed 3."""
        kw = self._baseline_kwargs()
        s_2 = Grader.compute_final_score(**kw, hypothesis_attempts=2)
        s_3 = Grader.compute_final_score(**kw, hypothesis_attempts=3)
        assert s_2 == s_3
