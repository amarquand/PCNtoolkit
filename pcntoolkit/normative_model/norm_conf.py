import os
import json
from dataclasses import dataclass


@dataclass(frozen=True)  # Creates an immutable datasclass
class NormConf:
    """
    Configuration for a normative model. Contains parameters for cross-validation, logging and saving output. 
    Performs checks on these configurations.
    This does not care about the underlying regression model. 
    """
    perform_cv: bool = False
    cv_folds: int = 0
    log_dir: str = "./logs"
    save_dir: str = "./saves"

    def __post_init__(self):
        """
        Checks if the configuration is valid.
        """
        configuration_problems = self.detect_configuration_problems()
        if not configuration_problems == "":
            raise ValueError(
                f"The following problems have been detected in the normative model configuration:\n{configuration_problems}")
        else:
            print("Configuration of normative model is valid.")

    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a string.
        """

        configuration_problems = ""
        problem_count: int = 0

        def add_problem(problem: str):
            nonlocal problem_count
            nonlocal configuration_problems
            problem_count += 1
            configuration_problems += f"{problem_count}:\t{problem}\n"

        performisbool = isinstance(self.perform_cv, bool)
        foldsisint = isinstance(self.cv_folds, int)
        logisstr = isinstance(self.log_dir, str)
        saveisstr = isinstance(self.save_dir, str)

        if not performisbool:
            add_problem(
                f"perform_cv is not a boolean, but {type(self.perform_cv).__name__}")
        if not foldsisint:
            add_problem(
                f"cv_folds is not an integer, but {type(self.cv_folds).__name__}")
        if not logisstr:
            add_problem(f"log_dir is not a string, but {type(self.log_dir).__name__}")
        if not saveisstr:
            add_problem(
                f"save_dir is not a string, but {type(self.save_dir).__name__}")
        if performisbool and foldsisint:
            if self.perform_cv and self.cv_folds < 2:
                add_problem(
                    f"cv_folds must be at least 2, but is {self.cv_folds}")
        if logisstr:
            if os.path.exists(self.log_dir):
                if not os.path.isdir(self.log_dir):
                    add_problem(
                        f"Provided log_dir is not a directory, but {self.get_type_of_object(self.log_dir)}")
            else:
                add_problem(f"Provided log_dir does not exist")

        if saveisstr:
            if os.path.exists(self.save_dir):
                if not os.path.isdir(self.save_dir):
                    add_problem(
                        f"Provided save_dir is not a directory, but {self.get_type_of_object(self.save_dir)}")
            else:
                add_problem(f"Provided save_dir does not exist")

        return configuration_problems

    def get_type_of_object(self, path: str) -> str:
        """
        Returns the type of the object that the path points to.
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                return "directory"
            elif os.path.isfile(path):
                return "file"
            else:
                return "other"
        else:
            return "nonexistent"

    def save_as_json(self, path: str):
        """
        Saves the configuration as a json file.
        """
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    @classmethod
    def load_from_json(cls, path) -> 'NormConf':
        """
        Loads the configuration from a json file.
        """
        with open(path, "r") as f:
            conf_dict = json.load(f)
        return cls(**conf_dict)
