"""Deterministic oracle grading formula for PostmortemEnv.

Terminal scoring is now driven by the formal :class:`RubricSet` defined
in :mod:`engine.rubrics`.  The five composable rubrics are:

1. **Cause correctness** (40 %) — did the agent name the right root cause?
2. **Chain accuracy** (25 %) — Jaccard node+edge similarity of the chain.
3. **Efficiency** (15 %) — how quickly the agent solved relative to budget.
4. **Investigation quality** (10 %) — fact coverage × action diversity.
5. **Anti-gaming** (10 %) — penalises brute-force guess patterns.

All scores are clamped to (0.01, 0.99) for OpenEnv validator compliance.
"""

from typing import Any, Dict, List, Optional

from engine.rubrics import POSTMORTEM_RUBRICS


def _compute_graph_similarity(predicted: List[Dict[str, str]], actual: List[Dict[str, str]]) -> float:
    """Compute structural graph similarity between two causal chains.

    Each chain is interpreted as a sequence of directed nodes:
    ``[A → B → C]``.  Similarity combines:
    • **Node Jaccard** (weight 0.4) — overlap of ``(service, effect)`` tuples.
    • **Edge Jaccard** (weight 0.6) — overlap of directed bigrams.

    Returns a value in ``[0.0, 1.0]``; ``1.0`` = identical.
    """
    if not actual:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0

    # Extract nodes
    pred_nodes = set((s.get("service", ""), s.get("effect", "")) for s in predicted)
    act_nodes = set((s.get("service", ""), s.get("effect", "")) for s in actual)

    # Extract edges (bigrams)
    pred_edges = set()
    for i in range(len(predicted) - 1):
        n1 = (predicted[i].get("service", ""), predicted[i].get("effect", ""))
        n2 = (predicted[i+1].get("service", ""), predicted[i+1].get("effect", ""))
        pred_edges.add((n1, n2))

    act_edges = set()
    for i in range(len(actual) - 1):
        n1 = (actual[i].get("service", ""), actual[i].get("effect", ""))
        n2 = (actual[i+1].get("service", ""), actual[i+1].get("effect", ""))
        act_edges.add((n1, n2))

    # Compute node Jaccard similarity (weight: 0.4)
    nodes_intersection = len(pred_nodes.intersection(act_nodes))
    nodes_union = len(pred_nodes.union(act_nodes))
    node_sim = nodes_intersection / nodes_union if nodes_union > 0 else 0.0

    # Compute edge Jaccard similarity (weight: 0.6)
    edges_intersection = len(pred_edges.intersection(act_edges))
    edges_union = len(pred_edges.union(act_edges))
    edge_sim = edges_intersection / edges_union if edges_union > 0 else 1.0 if not act_edges and not pred_edges else 0.0

    return 0.4 * node_sim + 0.6 * edge_sim


class Grader:
    """Deterministic oracle grader for PostmortemEnv.

    Uses the composable rubric system defined in ``engine.rubrics``.
    The five rubrics are independently scored and then aggregated:

    ``r_terminal = Σ (rubric_weight × rubric_score)``

    All scores clamped to ``(0.01, 0.99)`` for validator compliance.
    """

    @staticmethod
    def check_cause_match(
        submitted_cause: str,
        ground_truth_cause: str,
        cause_type: str = "commit",
        contributing_causes: Optional[List[str]] = None,
    ) -> bool:
        """Check if the submitted cause is an *exact* match for the root cause.

        For ``cause_type == 'correlated'`` scenarios, the only response that
        counts as fully correct is the combined cause ID (e.g.
        ``'commit-abc+infra-xyz'``). Naming a single contributing cause is
        not considered "correct" here — that case is awarded 0.5 partial
        credit by :func:`compute_final_score` via its ``partial_credit``
        branch. Keeping ``check_cause_match`` strict makes that branch
        reachable; previously it returned ``True`` for any contributing
        cause, which silently turned the partial-credit logic into dead
        code (see analysis_results (2).md C5).
        """
        if not submitted_cause or not ground_truth_cause:
            return False

        submitted = submitted_cause.strip().lower()
        truth = ground_truth_cause.strip().lower()

        return submitted == truth

    @staticmethod
    def is_contributing_cause_match(
        submitted_cause: str,
        cause_type: str,
        contributing_causes: Optional[List[str]] = None,
    ) -> bool:
        """Return True iff the submission names *one* of the contributing
        causes for a correlated incident (and is not the full combined ID).

        Used by terminal scoring to award 0.5 partial cause credit. Does
        not consider exact-match cases — those are handled by
        :func:`check_cause_match`.
        """
        if cause_type != "correlated" or not contributing_causes:
            return False
        if not submitted_cause:
            return False
        submitted = submitted_cause.strip().lower()
        for cc in contributing_causes:
            if submitted == cc.strip().lower():
                return True
        return False

    @staticmethod
    def compute_chain_similarity(
        predicted_chain: Optional[List[Dict[str, str]]],
        actual_chain: List[Dict[str, str]],
    ) -> float:
        """Compute similarity between predicted and actual causal chains.

        Returns value in ``[0.0, 1.0]`` where ``1.0`` = perfect match.
        """
        if predicted_chain is None:
            return 0.0

        similarity = _compute_graph_similarity(predicted_chain, actual_chain)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def compute_final_score(
        submitted_cause: str,
        submitted_chain: Optional[List[Dict[str, str]]],
        ground_truth_cause: str,
        ground_truth_chain: List[Dict[str, str]],
        cause_type: str,
        steps_used: int,
        max_steps: int,
        n_wrong_hypotheses: int,
        contributing_causes: Optional[List[str]] = None,
        facts_found: int = 0,
        total_facts: int = 1,
        action_types_used: int = 0,
        repeated_queries: int = 0,
        hypothesis_attempts: int = 0,
    ) -> float:
        """Compute the final episode score using the composable rubric system.

        Parameters
        ----------
        submitted_cause, submitted_chain
            The agent's final answer.
        ground_truth_cause, ground_truth_chain
            The oracle answer.
        cause_type
            ``'commit'``, ``'config'``, ``'infra'``, or ``'correlated'``.
        steps_used, max_steps
            Efficiency inputs.
        n_wrong_hypotheses
            Count of incorrect ``hypothesize`` calls.
        contributing_causes
            For correlated causes, list of constituent cause IDs.
        facts_found, total_facts
            Investigation quality inputs.
        action_types_used
            Number of distinct action types the agent used.
        repeated_queries
            Number of exact-duplicate queries the agent issued.
        hypothesis_attempts
            Total number of ``hypothesize`` calls (correct + incorrect).

        Returns
        -------
        float
            Score clamped to ``(0.01, 0.99)``.
        """
        # --- Cause correctness ---
        is_correct = Grader.check_cause_match(
            submitted_cause, ground_truth_cause, cause_type, contributing_causes
        )
        partial_credit = 0.0
        if not is_correct and Grader.is_contributing_cause_match(
            submitted_cause, cause_type, contributing_causes
        ):
            partial_credit = 0.5

        # --- Chain similarity ---
        chain_sim = Grader.compute_chain_similarity(submitted_chain, ground_truth_chain)

        # --- Evaluate via rubric system ---
        result = POSTMORTEM_RUBRICS.evaluate(
            cause_match=is_correct,
            partial_credit=partial_credit,
            chain_similarity=chain_sim,
            steps_used=steps_used,
            max_steps=max_steps,
            facts_found=facts_found,
            total_facts=total_facts,
            action_types_used=action_types_used,
            wrong_hypotheses=n_wrong_hypotheses,
            repeated_queries=repeated_queries,
            hypothesis_attempts=hypothesis_attempts,
        )

        raw_score = result["total"]

        # Clamp to (0.01, 0.99) for validator compliance
        try:
            val = float(raw_score)
            if val != val:  # NaN check
                return 0.01
            return round(min(max(val, 0.01), 0.99), 4)
        except (ValueError, TypeError):
            return 0.01

    @staticmethod
    def compute_final_score_with_breakdown(
        submitted_cause: str,
        submitted_chain: Optional[List[Dict[str, str]]],
        ground_truth_cause: str,
        ground_truth_chain: List[Dict[str, str]],
        cause_type: str,
        steps_used: int,
        max_steps: int,
        n_wrong_hypotheses: int,
        contributing_causes: Optional[List[str]] = None,
        facts_found: int = 0,
        total_facts: int = 1,
        action_types_used: int = 0,
        repeated_queries: int = 0,
        hypothesis_attempts: int = 0,
    ) -> Dict[str, Any]:
        """Same as ``compute_final_score`` but returns the full rubric breakdown.

        Returns
        -------
        dict
            ``{"score": float, "rubrics": [...]}``
        """
        # --- Cause correctness ---
        is_correct = Grader.check_cause_match(
            submitted_cause, ground_truth_cause, cause_type, contributing_causes
        )
        partial_credit = 0.0
        if not is_correct and Grader.is_contributing_cause_match(
            submitted_cause, cause_type, contributing_causes
        ):
            partial_credit = 0.5

        chain_sim = Grader.compute_chain_similarity(submitted_chain, ground_truth_chain)

        result = POSTMORTEM_RUBRICS.evaluate(
            cause_match=is_correct,
            partial_credit=partial_credit,
            chain_similarity=chain_sim,
            steps_used=steps_used,
            max_steps=max_steps,
            facts_found=facts_found,
            total_facts=total_facts,
            action_types_used=action_types_used,
            wrong_hypotheses=n_wrong_hypotheses,
            repeated_queries=repeated_queries,
            hypothesis_attempts=hypothesis_attempts,
        )

        raw_score = result["total"]
        try:
            val = float(raw_score)
            if val != val:
                val = 0.01
            clamped = round(min(max(val, 0.01), 0.99), 4)
        except (ValueError, TypeError):
            clamped = 0.01

        return {
            "score": clamped,
            "rubrics": result["breakdown"],
        }
