from dataclasses import dataclass, field

from pcntoolkit.regression_model.hbr.param import Param
from pcntoolkit.regression_model.reg_conf import RegConf

# Default configuration values
DRAWS = 1000
TUNE = 1000
CHAINS = 2
CORES = 1
LIKELIHOOD = "Normal"


@dataclass(frozen=True)
class HBRConf(RegConf):
    # sampling config
    draws: int = DRAWS
    tune: int = TUNE
    chains: int = CHAINS
    cores: int = CORES

    # model config
    likelihood: str = LIKELIHOOD

    # prior config with defaults
    mu: Param = field(default_factory=Param.default_mu)
    sigma: Param = field(default_factory=Param.default_sigma)
    epsilon: Param = field(default_factory=Param.default_epsilon)
    delta: Param = field(default_factory=Param.default_delta)

    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a list of strings.
        """
        configuration_problems = []

        def add_problem(problem: str):
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        # Check if nuts_sampler is valid
        if self.nuts_sampler not in ["pymc", "nutpie"]:
            add_problem(
                f"Nuts sampler '{self.nuts_sampler}' is not supported. Please specify a valid nuts sampler. Available options are 'pymc' and 'nutpie'."
            )

        # Check if likelihood is valid
        if self.likelihood not in ["Normal", "SHASHb", "SHASHo", "SHASHo2"]:
            add_problem(
                f"Likelihood '{self.likelihood}' is not supported. Please specify a valid likelihood."
            )

        # Check positivity of sigma
        if self.sigma:
            if self.sigma.linear:
                if self.sigma.mapping == "identity":
                    add_problem(
                        "Sigma must be strictly positive. As it's derived from a linear regression, it could potentially be negative without a proper mapping to the positive domain. To ensure positivity, use 'mapping=softplus' or 'mapping=exp'."
                    )
        # Check positivity of delta
        if self.likelihood.startswith("SHASH"):
            if self.delta:
                if self.delta.linear:
                    if self.delta.mapping == "identity":
                        add_problem(
                            "Delta must be strictly positive. As it's derived from a linear regression, it could potentially be negative without a proper mapping to the positive domain. To ensure positivity, use 'mapping=softplus' or 'mapping=exp'."
                        )
        # Check if epsilon and delta are provided for SHASH likelihoods
        if self.likelihood.startswith("SHASH"):
            if not self.epsilon:
                add_problem(
                    "Epsilon must be provided for SHASH likelihoods. Please specify epsilon."
                )
            if not self.delta:
                add_problem(
                    "Delta must be provided for SHASH likelihoods. Please specify delta."
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

    @property
    def has_random_effect(self):
        for attr in ["mu", "sigma", "epsilon", "delta"]:
            if getattr(self, attr) and getattr(self, attr).has_random_effect:
                return True
        return False
