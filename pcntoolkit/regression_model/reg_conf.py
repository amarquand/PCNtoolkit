import json
from abc import ABC, abstractmethod


class RegConf(ABC):
    """
    A class containig the configuration of a regression model. It should only contain configuration parameters, and not learned coefficients or such. Those are stored in the model itself.
    """

    def __post_init__(self):
        """
        Checks if the configuration is valid.
        """
        configuration_problems = self.detect_configuration_problems()
        if not configuration_problems == "":
            raise ValueError(
                f"The following problems have been detected in the regression model configuration:\n{configuration_problems}")
        else:
            print("Configuration of regression model is valid.")

    @abstractmethod
    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a string.
        """
        pass


    def save_as_json(self, path: str):
        """
        Saves the configuration as a json file.
        """
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)
            

    @classmethod
    def load_from_json(cls, path) -> 'RegConf':
        """
        Loads the configuration from a json file.
        """
        with open(path, "r") as f:
            conf_dict = json.load(f)
        return cls(**conf_dict)
