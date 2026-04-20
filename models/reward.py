"""Reward model. Feedback structure returned after each agent action."""

from pydantic import BaseModel, Field
from typing import Dict


class Reward(BaseModel):
    """
    Feedback after each step.

    The reward is decomposed into components so agents (and humans)
    can understand what drove the score:
    - information_gain: positive reward for finding relevant facts
    - query_cost: small negative per query to encourage efficiency
    - hypothesis_penalty: negative for wrong hypotheses
    - terminal_score: final grading score (only on submit)
    """
    value: float = Field(description="The numeric reward value")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Component breakdown, e.g. {'information_gain': 0.05, 'query_cost': -0.01}"
    )
    message: str = Field(default="", description="Human-readable explanation of the reward")
