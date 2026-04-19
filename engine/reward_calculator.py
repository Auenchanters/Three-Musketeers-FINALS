"""Per-step reward computation for PostmortemEnv. Deterministic and stateless."""

from models.reward import Reward

# --- Reward constants (from Foundation Doc §5.4) ---

# Information gain: awarded when a query uncovers oracle-labeled relevant facts
ALPHA = 0.05    # reward per relevant fact discovered
BETA = 0.01     # cost per query (encourages efficiency)

# Hypothesis rewards/penalties
HYPOTHESIS_CORRECT_SIGNAL = 0.02    # small positive for correct hypothesis
HYPOTHESIS_WRONG_PENALTY = -0.10    # penalty for incorrect hypothesis

# Step cost
STEP_COST = -0.005  # small cost per step to push toward efficiency

# Submit (terminal)
# Terminal reward is computed by the Grader, not here


class RewardCalculator:
    """Computes per-step rewards for PostmortemEnv."""

    @staticmethod
    def query_reward(n_relevant_facts_found: int) -> Reward:
        """
        Reward for a query action (query_logs, fetch_trace, diff_commit, inspect_config).

        r_step = +alpha * information_gain - beta * query_cost + step_cost
        """
        info_gain = ALPHA * n_relevant_facts_found
        query_cost = BETA
        total = info_gain - query_cost + STEP_COST

        if n_relevant_facts_found > 0:
            msg = f"Query found {n_relevant_facts_found} relevant fact(s). +{info_gain:.3f} info gain."
        else:
            msg = "Query returned results but no new relevant facts discovered."

        return Reward(
            value=total,
            breakdown={
                "information_gain": round(info_gain, 4),
                "query_cost": round(-query_cost, 4),
                "step_cost": STEP_COST,
            },
            message=msg,
        )

    @staticmethod
    def hypothesis_correct_reward() -> Reward:
        """Reward for a correct hypothesis (confirmation signal)."""
        total = HYPOTHESIS_CORRECT_SIGNAL + STEP_COST
        return Reward(
            value=total,
            breakdown={
                "hypothesis_correct": HYPOTHESIS_CORRECT_SIGNAL,
                "step_cost": STEP_COST,
            },
            message="Hypothesis CORRECT! You identified the right root cause.",
        )

    @staticmethod
    def hypothesis_wrong_reward() -> Reward:
        """Penalty for an incorrect hypothesis."""
        total = HYPOTHESIS_WRONG_PENALTY + STEP_COST
        return Reward(
            value=total,
            breakdown={
                "hypothesis_penalty": HYPOTHESIS_WRONG_PENALTY,
                "step_cost": STEP_COST,
            },
            message="Hypothesis INCORRECT. The submitted cause does not match the ground truth.",
        )

    @staticmethod
    def chain_feedback_reward(chain_similarity: float) -> Reward:
        """Reward for an explain_chain action — returns similarity feedback."""
        # Small info gain proportional to how close the chain is
        info_gain = 0.02 * chain_similarity
        total = info_gain + STEP_COST

        if chain_similarity >= 0.8:
            msg = f"Chain is very close to the ground truth (similarity: {chain_similarity:.2f})."
        elif chain_similarity >= 0.5:
            msg = f"Chain partially matches the ground truth (similarity: {chain_similarity:.2f})."
        else:
            msg = f"Chain does not match well (similarity: {chain_similarity:.2f}). Revise your hypothesis."

        return Reward(
            value=total,
            breakdown={
                "chain_feedback": round(info_gain, 4),
                "step_cost": STEP_COST,
            },
            message=msg,
        )

    @staticmethod
    def submit_reward(terminal_score: float) -> Reward:
        """Reward for submitting the final answer. Wraps the grader output."""
        return Reward(
            value=terminal_score,
            breakdown={
                "terminal_score": round(terminal_score, 4),
            },
            message=f"Episode complete. Final score: {terminal_score:.4f}",
        )

    @staticmethod
    def invalid_action_reward(message: str) -> Reward:
        """Penalty for an invalid action."""
        return Reward(
            value=STEP_COST,
            breakdown={"step_cost": STEP_COST},
            message=message,
        )
