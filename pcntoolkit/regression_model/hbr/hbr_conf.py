from dataclasses import dataclass
from pcntoolkit.regression_model.hbr.paramconf import ParamConf, default_mu_conf, default_sigma_conf
from pcntoolkit.regression_model.reg_conf import RegConf

@dataclass(frozen=True)
class HBRConf(RegConf):
    # sampling config
    n_samples: int=1000
    n_tune: int=1000
    n_cores: int=1

    # model config
    likelihood: str="Normal"
    mu:ParamConf = default_mu_conf()
    sigma:ParamConf = default_sigma_conf()


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
