"""
Local shim for openenv.core.env_server — Python 3.9 compatible.

Provides create_app() which wraps a PostmortemEnvironment into a FastAPI
app with /reset, /step, /state, /health endpoints matching the openenv-core
HTTP server contract.
"""

from typing import Any, Dict, Optional, Type

from fastapi import Body, FastAPI

from .types import Observation, Action, EnvironmentMetadata
from .interfaces import Environment


def create_app(
    env,
    action_cls: Type = None,
    observation_cls: Type = None,
    env_name: Optional[str] = None,
) -> FastAPI:
    """
    Create a FastAPI application with standard OpenEnv endpoints.

    Args:
        env: The Environment class (will be instantiated) or instance.
        action_cls: The Action model class.
        observation_cls: The Observation model class.
        env_name: Optional name for the environment.

    Returns:
        FastAPI application instance.
    """
    app = FastAPI(title="PostmortemEnv HTTP Server")

    # Instantiate if a class was passed
    if isinstance(env, type):
        env_instance = env()
    else:
        env_instance = env

    @app.post("/reset")
    async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        task_id = request.get("task_id", "task1_recent_deploy")
        seed = request.get("seed")
        obs = env_instance.reset(task_id=task_id, seed=seed)
        return _serialize_obs(obs)

    @app.post("/step")
    async def step(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Execute an action and return the resulting observation."""
        action_data = request if "action_type" in request else request.get("action", request)
        # Remove metadata if present
        action_data.pop("metadata", None)
        action = action_cls(**action_data)
        obs = env_instance.step(action)
        return _serialize_obs(obs)

    @app.get("/state")
    async def get_state() -> Dict[str, Any]:
        """Return the internal environment state."""
        state = env_instance.state
        if hasattr(state, "model_dump"):
            return state.model_dump()
        return vars(state)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check."""
        return {"status": "healthy"}

    @app.get("/schema")
    async def schema() -> Dict[str, Any]:
        """Return the action and observation JSON schemas."""
        schemas = {}
        if action_cls and hasattr(action_cls, "model_json_schema"):
            schemas["action"] = action_cls.model_json_schema()
        if observation_cls and hasattr(observation_cls, "model_json_schema"):
            schemas["observation"] = observation_cls.model_json_schema()
        return schemas

    @app.get("/metadata")
    async def metadata() -> Dict[str, Any]:
        """Return environment metadata."""
        if hasattr(env_instance, "get_metadata"):
            meta = env_instance.get_metadata()
            if hasattr(meta, "model_dump"):
                return meta.model_dump()
            return vars(meta)
        return {"name": env_name or "PostmortemEnv", "version": "1.0.0"}

    return app


def _serialize_obs(obs) -> Dict[str, Any]:
    """Convert an Observation to the wire format expected by EnvClient."""
    if hasattr(obs, "model_dump"):
        obs_dict = obs.model_dump()
    else:
        obs_dict = vars(obs)

    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    obs_dict.pop("metadata", None)

    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }
