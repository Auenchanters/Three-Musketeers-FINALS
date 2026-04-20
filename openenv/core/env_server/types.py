"""
Local shim for openenv-core types — Python 3.9 compatible.

Replicates the base Observation, Action, and State classes from
openenv-core 0.1.0 using Pydantic BaseModel instead of
@dataclass(kw_only=True) (which requires Python 3.10+).
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union


# Type aliases (same as upstream)
Scalar = Union[int, float, bool]


class Action(BaseModel):
    """Base class for all environment actions."""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Base class for all environment observations."""
    done: bool = False
    reward: Optional[Union[bool, int, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Base class for environment state."""
    episode_id: Optional[str] = None
    step_count: int = 0


class EnvironmentMetadata(BaseModel):
    """Metadata about an environment for documentation and UI purposes."""
    name: str = ""
    description: str = ""
    readme_content: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    documentation_url: Optional[str] = None
