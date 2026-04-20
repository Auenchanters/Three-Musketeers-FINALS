"""
Local shim for openenv-core interfaces — Python 3.9 compatible.

Replicates the Environment ABC from openenv-core without requiring
Python 3.10+ features.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from .types import Action, Observation, State

A = TypeVar("A", bound=Action)
O = TypeVar("O", bound=Observation)
S = TypeVar("S", bound=State)


class Environment(ABC, Generic[A, O, S]):
    """Base class for all OpenEnv environments."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> O:
        ...

    @abstractmethod
    def step(self, action: A, **kwargs: Any) -> O:
        ...
