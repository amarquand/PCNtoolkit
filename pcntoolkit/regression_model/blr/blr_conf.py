from dataclasses import dataclass

from pcntoolkit.regression_model.reg_conf import RegConf


@dataclass(frozen=True)
class BLRConf(RegConf):
    # some configuration parameters
    n_iter: int = 100
    tol: float = 1e-3

    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a list of strings.
        The super class will throw an exception if the configuration is invalid, and show the problems.
        """

        # DESIGN CHOICE (stijn):
        # This mutable field need to be local here, because the dataclass is defined as immutable.
        configuration_problems = []

        def add_problem(problem: str):
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        # some configuration checks
        # ...
        if self.example_parameter < 0:
            add_problem("Example parameter must be greater than 0.")

        return configuration_problems

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        return cls(
            n_iter=args["n_iter"],
            tol=args["tol"],
        )

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from a dictionary.
        """
        return cls(
            n_iter=dict["n_iter"],
            tol=dict["tol"],
        )

    def to_dict(self):
        """
        Converts the configuration to a dictionary.
        """
        return {
            "n_iter": self.n_iter,
            "tol": self.tol,
        }
