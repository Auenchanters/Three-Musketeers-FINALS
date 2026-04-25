"""Small synchronous HTTP client for PostmortemEnv smoke tests and demos."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from models.action import Action
from models.observation import Observation
from models.state import EnvironmentState


@dataclass
class StepResult:
    """Result of one environment step."""

    observation: Observation
    reward: Optional[float] = None
    done: bool = False


class PostmortemClient:
    """Synchronous client for the stateful HTTP endpoints.

    The official OpenEnv client is WebSocket-first in recent releases. This
    lightweight client keeps the repo's CLI validators and inference harness
    simple while the server still exposes the canonical OpenEnv ``/ws`` path.
    """

    def __init__(self, base_url: str = "http://localhost:7860", request_timeout_s: float = 15.0):
        self._base = base_url.rstrip("/")
        self._timeout = float(request_timeout_s)
        self._http = requests.Session()

    def reset(self, task_id: str = "", seed: Optional[int] = None, **kwargs: Any) -> Observation:
        body: Dict[str, Any] = {}
        if task_id:
            body["task_id"] = task_id
        if seed is not None:
            body["seed"] = seed
        body.update(kwargs)
        resp = self._http.post(f"{self._base}/reset", json=body, timeout=self._timeout)
        resp.raise_for_status()
        return self._parse_result(resp.json()).observation

    def step(self, action: Action, **kwargs: Any) -> StepResult:
        payload = action.model_dump(exclude_none=True)
        if kwargs:
            payload = {"action": payload, **kwargs}
        resp = self._http.post(f"{self._base}/step", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def get_state(self) -> EnvironmentState:
        resp = self._http.get(f"{self._base}/state", timeout=self._timeout)
        resp.raise_for_status()
        return self._parse_state(resp.json())

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Convert a JSON response from the env server to StepResult."""
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
