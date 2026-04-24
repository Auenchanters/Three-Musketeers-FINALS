"""Per-step reward computation for PostmortemEnv.

Deterministic and stateless — every method is a pure function of its
arguments.  The reward constants are tuned so that per-step signals are
meaningful during GRPO/PPO (≈35% of total episode return) rather than
being dwarfed by the terminal score (≈65%).

Anti-gaming measures
────────────────────
• Escalating penalty for repeated (service, keyword) queries.
• Coherence bonus for following the dependency graph.
• Action diversity bonus for using ≥3 distinct action types.
• These signals are *independent* of the terminal rubric, giving the
  optimizer multiple orthogonal objectives — harder to hack.
"""

from typing import Dict, List, Optional, Set, Tuple

from models.reward import Reward

# ── Reward constants (rebalanced: stronger per-step signal) ──────────

# Information gain: awarded when a query uncovers oracle-labeled relevant facts
ALPHA = 0.08          # reward per relevant fact discovered  (was 0.05)
BETA = 0.01           # cost per query (encourages efficiency)

# Hypothesis rewards/penalties
HYPOTHESIS_CORRECT_SIGNAL = 0.02
HYPOTHESIS_WRONG_PENALTY = -0.12   # slightly harsher  (was -0.10)

# Step cost
STEP_COST = -0.005

# Anti-gaming constants
REPEAT_QUERY_BASE_PENALTY = -0.02  # first repeat
REPEAT_QUERY_ESCALATION = -0.02    # each additional repeat
COHERENCE_BONUS = 0.02             # following the dependency graph
DIVERSITY_BONUS_PER_TYPE = 0.01    # per distinct action type (above 2)


class RewardCalculator:
    """Computes per-step rewards for PostmortemEnv.

    All methods are static and pure-functional.  Side-effect tracking
    (repeat counts, visited services, etc.) lives in the environment.
    """

    @staticmethod
    def query_reward(
        n_relevant_facts_found: int,
        repeat_count: int = 0,
        coherence_hit: bool = False,
    ) -> Reward:
        """Reward for a query action (query_logs, fetch_trace, diff_commit, etc.).

        Parameters
        ----------
        n_relevant_facts_found : int
            How many *new* oracle-labeled facts this query surfaced.
        repeat_count : int
            How many times the exact same (service, keyword) pair has been
            issued before in this episode.  0 = first time.
        coherence_hit : bool
            Whether this query's target service is a graph-neighbour of the
            previously queried service (follows the dependency DAG).
        """
        # --- Base info-gain ---
        info_gain = ALPHA * n_relevant_facts_found
        query_cost = BETA
        total = info_gain - query_cost + STEP_COST

        # --- Repeat penalty (escalating) ---
        repeat_penalty = 0.0
        if repeat_count > 0:
            repeat_penalty = REPEAT_QUERY_BASE_PENALTY + REPEAT_QUERY_ESCALATION * (repeat_count - 1)
            total += repeat_penalty

        # --- Coherence bonus ---
        coh = 0.0
        if coherence_hit:
            coh = COHERENCE_BONUS
            total += coh

        # --- Message ---
        parts = []
        if n_relevant_facts_found > 0:
            parts.append(f"Found {n_relevant_facts_found} relevant fact(s) (+{info_gain:.3f}).")
        else:
            parts.append("No new relevant facts discovered.")
        if repeat_count > 0:
            parts.append(f"Repeated query ({repeat_count}x before, penalty {repeat_penalty:+.3f}).")
        if coherence_hit:
            parts.append(f"Followed dependency graph (+{coh:.3f}).")

        return Reward(
            value=total,
            breakdown={
                "information_gain": round(info_gain, 4),
                "query_cost": round(-query_cost, 4),
                "step_cost": STEP_COST,
                "repeat_penalty": round(repeat_penalty, 4),
                "coherence_bonus": round(coh, 4),
            },
            message=" ".join(parts),
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
    def hypothesis_limit_reward() -> Reward:
        """Penalty when the agent exceeds the hypothesis attempt cap."""
        return Reward(
            value=STEP_COST,
            breakdown={"step_cost": STEP_COST},
            message=(
                "Hypothesis limit reached (max 3 per episode). "
                "Investigate more before guessing."
            ),
        )

    @staticmethod
    def chain_feedback_reward(chain_similarity: float) -> Reward:
        """Reward for an explain_chain action — returns qualitative feedback.

        The numeric similarity is *not* exposed to the agent (anti-gaming).
        Only a qualitative band is returned.
        """
        # Small info gain proportional to how close the chain is
        info_gain = 0.02 * chain_similarity
        total = info_gain + STEP_COST

        if chain_similarity >= 0.8:
            msg = "Your causal chain is very close to the actual chain. Strong reasoning."
        elif chain_similarity >= 0.5:
            msg = "Your chain partially matches. Some steps are correct, others need revision."
        elif chain_similarity >= 0.2:
            msg = "Your chain has some relevant elements but the overall structure is off."
        else:
            msg = "Your chain does not match well. Re-examine the evidence and revise."

        return Reward(
            value=total,
            breakdown={
                "chain_feedback": round(info_gain, 4),
                "step_cost": STEP_COST,
            },
            message=msg,
        )

    @staticmethod
    def diversity_reward(n_distinct_types: int) -> float:
        """Per-step bonus for using diverse action types.

        Returns the *increment* to add (not the total).  Only fires
        when the agent has used ≥3 distinct action types.
        """
        if n_distinct_types >= 3:
            return DIVERSITY_BONUS_PER_TYPE
        return 0.0

    @staticmethod
    def submit_reward(terminal_score: float) -> Reward:
        """Reward for submitting the final answer. Wraps the rubric output."""
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
