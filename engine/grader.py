"""Deterministic oracle grading formula for PostmortemEnv."""

from typing import List, Dict, Optional


def _compute_graph_similarity(predicted: List[Dict[str, str]], actual: List[Dict[str, str]]) -> float:
    """
    Compute structural graph similarity between two causal chains.

    Each chain is interpreted as a sequence of nodes: [A -> B -> C].
    We compute the Jaccard similarity of both the individual nodes (service, effect)
    and the directed edges between them.

    Returns a value in [0.0, 1.0] where 1.0 = identical, 0.0 = completely different.
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
    """
    Deterministic oracle grader for PostmortemEnv.

    Scoring formula (from Foundation Doc §5.4):
        r_terminal = 0.5 * is_correct_cause
                   + 0.3 * chain_similarity
                   + 0.2 * efficiency_bonus
                   - 0.1 * n_wrong_hypotheses

    Where:
        - is_correct_cause: 1 if submitted cause matches ground truth, else 0
        - chain_similarity: compute_graph_similarity(predicted, actual)
        - efficiency_bonus: max(0, 1 - steps_used / max_steps)
        - n_wrong_hypotheses: count of incorrect hypothesize calls

    All scores clamped to (0.01, 0.99) for validator compliance.
    """

    @staticmethod
    def check_cause_match(
        submitted_cause: str,
        ground_truth_cause: str,
        cause_type: str = "commit",
        contributing_causes: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if the submitted cause matches the ground truth.

        For correlated causes (cause_type == 'correlated'), accepts:
        - Exact match on the combined cause ID (e.g. 'commit-abc+infra-xyz')
        - Any of the contributing causes individually (partial credit handled in scoring)
        """
        if not submitted_cause or not ground_truth_cause:
            return False

        # Normalize for comparison
        submitted = submitted_cause.strip().lower()
        truth = ground_truth_cause.strip().lower()

        if submitted == truth:
            return True

        # For correlated causes, check if submitted matches any contributing cause
        if cause_type == "correlated" and contributing_causes:
            for cc in contributing_causes:
                if submitted == cc.strip().lower():
                    return True

        return False

    @staticmethod
    def compute_chain_similarity(
        predicted_chain: Optional[List[Dict[str, str]]],
        actual_chain: List[Dict[str, str]],
    ) -> float:
        """
        Compute similarity between predicted and actual causal chains.

        Returns value in [0.0, 1.0] where 1.0 = perfect match.
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
    ) -> float:
        """
        Compute the final episode score.

        Args:
            submitted_cause: The agent's submitted root cause entity ID.
            submitted_chain: The agent's submitted causal chain.
            ground_truth_cause: The true root cause entity ID.
            ground_truth_chain: The true causal chain.
            cause_type: Type of cause ('commit', 'config', 'infra', 'correlated').
            steps_used: Number of steps the agent took.
            max_steps: Maximum steps allowed for this task.
            n_wrong_hypotheses: Count of incorrect hypothesize calls.
            contributing_causes: For correlated causes, list of contributing cause IDs.

        Returns:
            Score clamped to (0.01, 0.99).
        """
        # Component 1: Cause correctness (0.5 weight)
        is_correct = Grader.check_cause_match(
            submitted_cause, ground_truth_cause, cause_type, contributing_causes
        )
        cause_score = 1.0 if is_correct else 0.0

        # For correlated causes, partial credit if only one contributing cause identified
        if cause_type == "correlated" and not is_correct and contributing_causes:
            submitted_lower = (submitted_cause or "").strip().lower()
            for cc in contributing_causes:
                if submitted_lower == cc.strip().lower():
                    cause_score = 0.5  # partial credit
                    break

        # Component 2: Chain similarity (0.3 weight)
        chain_sim = Grader.compute_chain_similarity(submitted_chain, ground_truth_chain)

        # Component 3: Efficiency bonus (0.2 weight)
        if max_steps > 0:
            efficiency = max(0.0, 1.0 - steps_used / max_steps)
        else:
            efficiency = 0.0

        # Combine
        raw_score = (
            0.5 * cause_score
            + 0.3 * chain_sim
            + 0.2 * efficiency
            - 0.1 * n_wrong_hypotheses
        )

        # Clamp to (0.01, 0.99) for validator compliance
        try:
            val = float(raw_score)
            if val != val:  # NaN check
                return 0.01
            return round(min(max(val, 0.01), 0.99), 4)
        except (ValueError, TypeError):
            return 0.01
