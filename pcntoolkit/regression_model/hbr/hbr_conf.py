from dataclasses import dataclass
from pcntoolkit.regression_model.reg_conf import RegConf


@dataclass(frozen=True)
class HBRConf(RegConf):
    # some configuration parameters
    linear_mu: bool=True
    random_intercept_mu: bool=False
    random_slope_mu: bool=False

    linear_sigma: bool=False

    likelihood: str="Normal"


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

        return configuration_problems