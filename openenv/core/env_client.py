"""
Local shim for openenv.core.env_client — Python 3.9 compatible.

Provides EnvClient (alias for HTTPEnvClient) and StepResult.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar
from dataclasses import dataclass

import requests

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")


@dataclass
class StepResult(Generic[ObsT]):
    """Result of one environment step."""
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False


class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    """
    Base class for OpenEnv environment clients.

    Communicates with an env server over HTTP.
    """

    def __init__(self, base_url: str, request_timeout_s: float = 15.0, **kwargs):
        self._base = base_url.rstrip("/")
        self._timeout = float(request_timeout_s)
        self._http = requests.Session()

    # --- abstract methods subclasses must implement ---

    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> "StepResult[ObsT]":
        ...

    @abstractmethod
    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        ...

    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        ...

    # --- concrete HTTP methods ---

    def reset(self, task_id: str = "", seed: Optional[int] = None, **kwargs) -> ObsT:
        """POST /reset to start a new episode."""
        body: Dict[str, Any] = {}
        if task_id:
            body["task_id"] = task_id
        if seed is not None:
            body["seed"] = seed
        body.update(kwargs)
        resp = self._http.post(
            f"{self._base}/reset", json=body, timeout=self._timeout
        )
        resp.raise_for_status()
        result = self._parse_result(resp.json())
        return result.observation

    def step(self, action: ActT, **kwargs) -> "StepResult[ObsT]":
        """POST /step with the action payload."""
        payload = self._step_payload(action)
        resp = self._http.post(
            f"{self._base}/step", json=payload, timeout=self._timeout
        )
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def get_state(self) -> StateT:
        """GET /state to inspect internal env state."""
        resp = self._http.get(f"{self._base}/state", timeout=self._timeout)
        resp.raise_for_status()
        return self._parse_state(resp.json())
