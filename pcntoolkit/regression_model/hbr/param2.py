from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Param(ABC):
    def __init__(self, name: str, dims: Optional[tuple[str] | str] = None):
        self.name = name
        self.dims = dims

    @abstractmethod
    def sample(self, data) -> Any:
        pass

    @property
    @abstractmethod
    def has_random_effect(self) -> bool:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(self, dict) -> Param:
        pass


class FixedParam(Param):
    def sample(self, data):
        return 1

    @property
    def has_random_effect(self):
        return False

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dict):
        pass


def param(
    name,
    dims,
    dist_name,
    dist_params,
    linear,
    slope,
    intercept,
    mapping,
    mapping_params,
    random,
    centered,
    mu,
    sigma,
    offset,
) -> Param:
    return FixedParam(name)
