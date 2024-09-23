from dataclasses import dataclass

from pcntoolkit.regression_model.blr.warp import WarpBase, WarpAffine, WarpCompose, WarpLog, WarpSinArcsinh, WarpBoxCox
from pcntoolkit.regression_model.reg_conf import RegConf

# Default configuration values
N_ITER = 100
TOL = 1e-3
ARD = False
OPTIMIZER = "l-bfgs-b"
L_BFGS_B_L = 0.1
L_BFGS_B_EPSILON = 0.1
L_BFGS_B_NORM = "l2"

@dataclass(frozen=True)
class BLRConf(RegConf):
    # some configuration parameters
    n_iter: int = N_ITER
    tol: float = TOL

    # use ard
    ard: bool = ARD

    # optimization parameters
    optimizer:str = OPTIMIZER # options: "l-bfgs-b", "cg", "powell", " nelder-mead"
    l_bfgs_b_l: float = L_BFGS_B_L
    l_bfgs_b_epsilon: float = L_BFGS_B_EPSILON
    l_bfgs_b_norm: str = L_BFGS_B_NORM

    # TODO implement var groups, var_covariates, and warp
    # var_groups: list = None
    # heteroskedastic: bool = False
    # warp: WarpBase = None
    # warp_reparam: bool = False

    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a list of strings.
        The super class will throw an exception if the configuration is invalid, and show the problems.
        """

        # DESIGN CHOICE (stijn)
        # This mutable field need to be local here, because the dataclass is defined as immutable.
        configuration_problems = []

        def add_problem(problem: str):
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        if self.n_iter < 1:
            add_problem("n_iter must be greater than 0.")

        if self.tol <= 0:
            add_problem("tol must be greater than 0.")

        if self.optimizer not in ["l-bfgs-b", "cg", "powell", "nelder-mead"]:
            add_problem(f"Optimizer {self.optimizer} not recognized.")

        return configuration_problems

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        args_filt = {k: v for k, v in args.items() if k in cls.__dataclass_fields__}

        return cls(
            n_iter=args_filt.get("n_iter", N_ITER),
            tol=args_filt.get("tol", TOL),
            ard=args_filt.get("ard", ARD),
            optimizer=args_filt.get("optimizer", OPTIMIZER),
            l_bfgs_b_l=args_filt.get("l_bfgs_b_l", L_BFGS_B_L),
            l_bfgs_b_epsilon=args_filt.get("l_bfgs_b_epsilon", L_BFGS_B_EPSILON),
            l_bfgs_b_norm=args_filt.get("l_bfgs_b_norm", L_BFGS_B_NORM),
        )

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from a dictionary.
        """
        return cls(
            n_iter=dict["n_iter"],
            tol=dict["tol"],
            ard=dict["ard"],
            optimizer=dict["optimizer"],
            l_bfgs_b_l=dict["l_bfgs_b_l"],
            l_bfgs_b_epsilon=dict["l_bfgs_b_epsilon"],
            l_bfgs_b_norm=dict["l_bfgs_b_norm"],
        )

    def to_dict(self):
        """
        Converts the configuration to a dictionary.
        """
        return {
            "n_iter": self.n_iter,
            "tol": self.tol,
            "ard": self.ard,
            "optimizer": self.optimizer,
            "l_bfgs_b_l": self.l_bfgs_b_l,
            "l_bfgs_b_epsilon": self.l_bfgs_b_epsilon,
            "l_bfgs_b_norm": self.l_bfgs_b_norm,
        }
