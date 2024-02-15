from dataclasses import dataclass, field

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

    # prior config with defaults
    mu: Param = field(default_factory=Param.default_mu)
    sigma: Param = field(default_factory=Param.default_sigma)
    epsilon: Param = field(default_factory=Param.default_epsilon)
    delta: Param = field(default_factory=Param.default_delta)

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

        # Check positivity of priors
        if self.sigma:
            if self.sigma.linear:
                if self.sigma.mapping == "identity":
                    add_problem(
                        "Sigma has to be strictly positive. It is a linear regression, so it can be potentially negative, because no mapping to the positive domain has been specified. Use 'mapping=softplus' or 'mapping=exp'."
                    )
        # Same for delta
        if self.likelihood.startswith("SHASH"):
            if self.epsilon:
                if self.epsilon.linear:
                    if self.epsilon.mapping == "identity":
                        add_problem(
                            "Epsilon has to be strictly positive. It is a linear regression, so it can be potentially negative, because no mapping to the positive domain has been specified. Use 'mapping=softplus' or 'mapping=exp'."
                        )

        return configuration_problems

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        # Filter out the arguments that are not relevant for this configuration
        args_filt = {k: v for k, v in args.items() if k in cls.__dataclass_fields__}
        likelihood = args_filt.get("likelihood", "Normal")
        if likelihood == "Normal":
            args_filt["mu"] = Param.from_args("mu", args)
            args_filt["sigma"] = Param.from_args("sigma", args)
        elif likelihood.startswith("SHASH"):
            args_filt["mu"] = Param.from_args("mu", args)
            args_filt["sigma"] = Param.from_args("sigma", args)
            args_filt["epsilon"] = Param.from_args("epsilon", args)
            args_filt["delta"] = Param.from_args("delta", args)
        self = cls(**args_filt)
        return self

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from a dictionary.
        """
        # Filter out the arguments that are not relevant for this configuration
        args_filt = {k: v for k, v in dict.items() if k in cls.__dataclass_fields__}
        likelihood = args_filt.get("likelihood", "Normal")
        if likelihood == "Normal":
            args_filt["mu"] = Param.from_dict(dict["mu"])
            args_filt["sigma"] = Param.from_dict(dict["sigma"])
        elif likelihood.startswith("SHASH"):
            args_filt["mu"] = Param.from_dict(dict["mu"])
            args_filt["sigma"] = Param.from_dict(dict["sigma"])
            args_filt["epsilon"] = Param.from_dict(dict["epsilon"])
            args_filt["delta"] = Param.from_dict(dict["delta"])
        self = cls(**args_filt)
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
