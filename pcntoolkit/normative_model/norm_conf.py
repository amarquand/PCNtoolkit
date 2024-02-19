import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)  # Creates an immutable datasclass
class NormConf:
    """
    Configuration for a normative model. Contains parameters for cross-validation, basis expansion, logging, and saving output.
    Performs checks on these configurations.
    This does not care about the underlying regression model.
    """

    savemodel: bool = False
    saveresults: bool = False
    log_dir: str = "./logs"
    save_dir: str = "./saves"

    # DESIGN CHOICE (stijn):
    # Add the basis function type here to keep the regression model agnostic of the basis function type.
    # The regression model should only see the dimensionality of the input data, and handle any shape incompatibilities itself.
    # With the default value of "linear", applied basis expansion will be an identity function.
    # In this way, we can always apply a basis expansion to the input data, even if the regression model does not support multiplee covariate dimensions.
    # possible model types: "linear", "polynomial", "bspline"
    basis_function: str = "linear"
    basis_column: int = (
        0  # the column of the input data that is used for the basis expansion
    )
    order: int = 3  # order of the polynomial or bspline basis functions
    nknots: int = 5  # number of knots for the bspline basis functions

    inscaler: str = "none"  # possible scalers: "none", "standardize", "minmax"
    outscaler: str = "none"  # possible scalers: "none", "standardize", "minmax"

    # Cross-validation parameters
    perform_cv: bool = False
    cv_folds: int = 0

    # The name of the regression model that is used. This is set by the regression model itself.
    normative_model_name: str = None

    def __post_init__(self):
        """
        Checks if the configuration is valid.
        """
        configuration_problems = self.detect_configuration_problems()
        if len(configuration_problems) > 0:
            problem_list = "\n".join(
                [f"{i+1}:\t{v}" for i, v in enumerate(configuration_problems)]
            )
            raise ValueError(
                f"The following problems have been detected in the normative model configuration:\n{problem_list}"
            )
        else:
            print("Configuration of normative model is valid.")

    @classmethod
    def from_args(cls, args) -> "NormConf":
        """
        Creates a configuration from command line arguments.
        """
        # Filter out the arguments that are not relevant for the normative model
        norm_args = {
            k: v for k, v in args.items() if k in cls.__dataclass_fields__.keys()
        }

        return cls(**norm_args)

    @classmethod
    def from_dict(cls, dict) -> "NormConf":
        """
        Creates a configuration from a dictionary.
        """
        return cls.from_args(dict)

    def to_dict(self):
        """
        Converts the configuration to a dictionary.
        """
        return self.__dict__

    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a list.
        """
        # DESIGN CHOICE (stijn):
        # This mutable list needs to be defined here, because the dataclass is defined as immutable, and can not hold mutable fields.
        # The add_problem function is defined here, because it needs access to the mutable configuration_problems variable.
        configuration_problems = []

        def add_problem(problem: str):
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        self.detect_dir_problem(add_problem, "log_dir")
        self.detect_dir_problem(add_problem, "save_dir")
        self.detect_cv_problem(add_problem)
        self.detect_basis_function_problem(add_problem)
        self.detect_scaler_problem(add_problem, "inscaler")
        self.detect_scaler_problem(add_problem, "outscaler")

        return configuration_problems

    def detect_dir_problem(self, add_problem, dir_attr_str):
        dir_attr = self.__getattribute__(dir_attr_str)
        if not isinstance(dir_attr, str):
            add_problem(
                f"{dir_attr_str} is not a string, but {type(dir_attr).__name__}"
            )
        else:
            if os.path.exists(dir_attr):
                if not os.path.isdir(dir_attr):
                    add_problem(
                        f"{dir_attr_str} is not a directory, but {self.get_type_of_object(dir_attr)}"
                    )
            else:
                warnings.warn(
                    f"{dir_attr_str} ({dir_attr}) does not exist, creating it for you"
                )
                os.makedirs(dir_attr)

    def detect_cv_problem(self, add_problem):
        performisbool = isinstance(self.perform_cv, bool)
        foldsisint = isinstance(self.cv_folds, int)
        if not performisbool:
            add_problem(
                f"perform_cv is not a boolean, but {type(self.perform_cv).__name__}"
            )
        if not foldsisint:
            add_problem(
                f"cv_folds is not an integer, but {type(self.cv_folds).__name__}"
            )
        if performisbool and foldsisint:
            if self.perform_cv and self.cv_folds < 2:
                add_problem(f"cv_folds must be at least 2, but is {self.cv_folds}")

    def detect_basis_function_problem(self, add_problem):
        acceptable_basis_functions = ["linear", "polynomial", "bspline"]
        if not isinstance(self.basis_function, str):
            add_problem(
                f"basis_function_type is not a string, but {type(self.basis_function).__name__}"
            )
        else:
            if self.basis_function not in acceptable_basis_functions:
                add_problem(
                    f"basis_function_type is not one of the possible values: {acceptable_basis_functions}"
                )

            if self.basis_function == "polynomial":
                self.detect_polynomial_basis_expansion_problem(add_problem)

            if self.basis_function == "bspline":
                self.detect_bspline_basis_expansion_problem(add_problem)

    def detect_bspline_basis_expansion_problem(self, add_problem):
        nknotsisint = isinstance(self.nknots, int)
        orderisint = isinstance(self.order, int)
        if not nknotsisint:
            add_problem(f"nknots is not an integer, but {type(self.nknots).__name__}")
        else:
            if self.nknots < 2:
                add_problem(f"nknots must be at least 2, but is {self.nknots}")

        if not orderisint:
            add_problem(f"order is not an integer, but {type(self.order).__name__}")

        else:
            if self.order < 1:
                add_problem(f"order must be at least 1, but is {self.order}")
            if nknotsisint:
                if self.order > self.nknots:
                    add_problem(
                        f"order must be smaller than nknots, but order is {self.order} and nknots is {self.nknots}"
                    )

    def detect_polynomial_basis_expansion_problem(self, add_problem):
        orderisint = isinstance(self.order, int)
        if not orderisint:
            add_problem(f"order is not an integer, but {type(self.order).__name__}")
        else:
            if self.order < 1:
                add_problem(f"order must be at least 1, but is {self.order}")

    def detect_scaler_problem(self, add_problem, scaler_attr_str):
        acceptable_scalers = ["none", "standardize", "minmax"]
        scaler_attr = self.__getattribute__(scaler_attr_str)
        if not isinstance(scaler_attr, str):
            add_problem(
                f"{scaler_attr_str} is not a string, but {type(scaler_attr).__name__}"
            )
        else:
            if scaler_attr not in acceptable_scalers:
                add_problem(
                    f"{scaler_attr_str} is not one of the possible values: {acceptable_scalers}"
                )

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

    def set_save_dir(self, path):
        """
        Sets the save directory to the given path.
        """
        object.__setattr__(self, "save_dir", path)

    def set_log_dir(self, path):
        """
        Sets the log directory to the given path.
        """
        object.__setattr__(self, "log_dir", path)

    def copy(self):
        """
        Returns a copy of the configuration.
        """
        return NormConf.from_args(self.to_dict())
