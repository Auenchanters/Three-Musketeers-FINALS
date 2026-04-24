"""Reward model. Feedback structure returned after each agent action."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Reward(BaseModel):
    """Feedback after each step.

    The reward is decomposed into components so agents (and humans)
    can understand what drove the score:

    - ``information_gain``: positive reward for finding relevant facts
    - ``query_cost``: small negative per query to encourage efficiency
    - ``hypothesis_penalty``: negative for wrong hypotheses
    - ``repeat_penalty``: escalating negative for duplicate queries
    - ``coherence_bonus``: positive for following the dependency graph
    - ``terminal_score``: final grading score (only on submit)

    When the episode ends (``submit`` action), ``rubric_scores`` contains
    the full breakdown from the composable rubric system.
    """
    value: float = Field(description="The numeric reward value")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Component breakdown, e.g. {'information_gain': 0.08, 'query_cost': -0.01}"
    )
    message: str = Field(default="", description="Human-readable explanation of the reward")
    rubric_scores: Optional[List[Dict]] = Field(
        default=None,
        description=(
            "Per-rubric breakdown from the composable rubric system "
            "(only populated on terminal submit action). Each entry has: "
            "rubric, raw_score, weight, weighted_score, description."
        ),
    )
