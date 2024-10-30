from __future__ import annotations

from abc import ABC, abstractmethod


class RegConf(ABC):
    """
    A class containig the configuration of a regression model. It should only contain configuration parameters, and not learned coefficients or such. Those are stored in the model itself.
    """

    def __post_init__(self) -> None:
        """
        Checks if the configuration is valid.
        """
        configuration_problems = self.detect_configuration_problems()
        if len(configuration_problems) > 0:
            problem_list = "\n".join(
                [f"{i+1}:\t{v}" for i, v in enumerate(configuration_problems)]
            )
            raise ValueError(
                f"The following problems have been detected in the regression model configuration:\n{problem_list}"
            )
        else:
            print("Configuration of regression model is valid.")

    @property
    @abstractmethod
    def has_random_effect(self) -> bool:
        return False

    @abstractmethod
    def detect_configuration_problems(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, dct: dict) -> RegConf:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, dct: dict) -> RegConf:
        pass

    @abstractmethod
    def to_dict(self, path: str | None = None) -> dict:
        pass
