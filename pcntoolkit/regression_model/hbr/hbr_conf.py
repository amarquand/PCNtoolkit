from dataclasses import dataclass
from pcntoolkit.regression_model.hbr.param import Param
from pcntoolkit.regression_model.reg_conf import RegConf


@dataclass(frozen=True)
class HBRConf(RegConf):
    # sampling config
    n_samples: int = 1000
    n_tune: int = 1000
    n_cores: int = 1

    # model config
    likelihood: str = "Normal"

    # prior config
    mu: Param = None
    sigma: Param = None
    epsilon: Param = None
    delta: Param = None

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
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        # Filter out the arguments that are not relevant for this configuration
        args_filt = {k: v for k, v in args.items() if k in cls.__dataclass_fields__}
        self = cls(**args_filt)
        if self.likelihood == "Normal":
            object.__setattr__(self, "mu", Param.from_dict("mu", args))
            object.__setattr__(
                self, "sigma", Param.from_dict("sigma",  args))
        elif self.likelihood.startswith("SHASH"):
            object.__setattr__(self, "mu", Param.from_dict("mu", args))
            object.__setattr__(
                self, "sigma", Param.from_dict("sigma",  args))
            object.__setattr__(
                self, "epsilon", Param.from_dict("epsilon", args))
            object.__setattr__(
                self, "delta", Param.from_dict("delta", args))
        return self
