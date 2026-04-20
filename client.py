"""
PostmortemEnv — OpenEnv Client

Implements the required abstract methods for WebSocket-based communication
with the PostmortemEnv server.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult
from models.action import Action
from models.observation import Observation
from models.state import EnvironmentState


class PostmortemClient(EnvClient[Action, Observation, EnvironmentState]):
    """
    Client for interacting with the PostmortemEnv environment.
    Use this class to communicate with the deployed HuggingFace Space.
    """

    def __init__(self, base_url: str = "http://localhost:7860", **kwargs):
        super().__init__(base_url, **kwargs)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        """Convert a JSON response from the env server to StepResult[Observation]."""
        obs_data = payload.get("observation", payload)
        observation = Observation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EnvironmentState:
        """Convert a JSON response from the state endpoint to EnvironmentState."""
        state_data = payload.get("state", payload)
        return EnvironmentState(**state_data)

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        return action.model_dump(exclude_none=True)
