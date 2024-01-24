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
