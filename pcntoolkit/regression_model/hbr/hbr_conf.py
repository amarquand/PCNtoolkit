from dataclasses import dataclass

from pcntoolkit.regression_model.hbr.param import Param
from pcntoolkit.regression_model.reg_conf import RegConf


@dataclass(frozen=True)
class HBRConf(RegConf):
    # sampling config
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    cores: int = 1

    # model config
    likelihood: str = "Normal"

    # prior config
    mu: Param = None
    sigma: Param = None
    epsilon: Param = None
    delta: Param = None

    # mu = Param(name="mu", linear=True, slope=Param("slope_mu", random=True), intercept=Param("intercept_mu", dist_name="Cauchy", dist_params=(0, 1)))

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

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from command line arguments.
        """
        # Filter out the arguments that are not relevant for this configuration
        args_filt = {k: v for k, v in dict.items() if k in cls.__dataclass_fields__}
        self = cls(**args_filt)
        if self.likelihood == "Normal":
            object.__setattr__(self, "mu", Param.from_dict("mu", dict))
            object.__setattr__(self, "sigma", Param.from_dict("sigma", dict))
        elif self.likelihood.startswith("SHASH"):
            object.__setattr__(self, "mu", Param.from_dict("mu", dict))
            object.__setattr__(self, "sigma", Param.from_dict("sigma", dict))
            object.__setattr__(self, "epsilon", Param.from_dict("epsilon", dict))
            object.__setattr__(self, "delta", Param.from_dict("delta", dict))
        return self

    def to_dict(self):
        conf_dict = {
            "draws": self.draws,
            "tune": self.tune,
            "cores": self.cores,
            "likelihood": self.likelihood,
        }
        if self.mu:
            conf_dict["mu"] = self.mu.to_dict()
        if self.sigma:
            conf_dict["sigma"] = self.sigma.to_dict()
        if self.epsilon:
            conf_dict["epsilon"] = self.epsilon.to_dict()
        if self.delta:
            conf_dict["delta"] = self.delta.to_dict()
        return conf_dict
