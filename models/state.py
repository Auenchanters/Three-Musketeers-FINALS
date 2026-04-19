"""Environment state model. Internal oracle/god-mode view for grading."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from openenv.core.env_server.types import State as BaseState


class EnvironmentState(BaseState):
    """
    Full internal environment state (god-mode view).

    This is returned by the state() endpoint for debugging and grading.
    It includes data the agent never sees, such as the ground-truth
    root cause and causal chain.
    """
    task_id: str = Field(description="ID of the current task")
    task_difficulty: str = Field(description="'easy', 'medium', or 'hard'")
    ground_truth_cause: str = Field(description="The true root cause entity ID")
    ground_truth_cause_type: str = Field(description="Type of root cause: 'commit', 'config', 'infra', or 'correlated'")
    ground_truth_chain: List[Dict[str, str]] = Field(
        default_factory=list,
        description="True causal chain: [{service, effect}, ...]"
    )
    relevant_fact_ids: List[str] = Field(
        default_factory=list,
        description="Oracle-labeled IDs of relevant facts in the telemetry"
    )
    facts_discovered: List[str] = Field(
        default_factory=list,
        description="Relevant facts the agent has found so far"
    )
    total_relevant_facts: int = Field(default=0, description="Total number of relevant facts in the scenario")
    hypotheses_submitted: List[Dict[str, str]] = Field(
        default_factory=list,
        description="History of hypotheses: [{cause_entity_id, result: correct/incorrect}]"
    )
    steps_taken: int = Field(default=0, description="Number of steps taken")
    max_steps: int = Field(default=40, description="Maximum steps for this task")
    done: bool = Field(default=False, description="Whether the episode is complete")
    final_score: Optional[float] = Field(default=None, description="Final graded score (only set after submit)")
