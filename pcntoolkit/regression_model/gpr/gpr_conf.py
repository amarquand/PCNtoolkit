# from dataclasses import dataclass

# from pcntoolkit.regression_model.reg_conf import RegConf


# @dataclass(frozen=True)
# class GPRConf(RegConf):
#     # some configuration parameters
#     # ...

#     # example parameter. This is an int and has a default value of 0.
#     n_iter: int = 100
#     tol: float = 1e-3
#     verbose: bool = False

#     def detect_configuration_problems(self) -> str:
#         """
#         Detects problems in the configuration and returns them as a list of strings.
#         """
#         configuration_problems = []

#         def add_problem(problem: str):
#             nonlocal configuration_problems
#             configuration_problems.append(f"{problem}")

#         # some configuration checks
#         # ...
#         if self.example_parameter < 0:
#             add_problem("Example parameter must be greater than 0.")

#         return configuration_problems

#     @classmethod
#     def from_args(cls, args):
#         """
#         Creates a configuration from command line arguments.
#         """
#         pass

#     @classmethod
#     def from_dict(cls, dict):
#         """
#         Creates a configuration from a dictionary.
#         """
#         pass

#     def to_dict(self):
#         """
#         Creates a dictionary from the configuration.
#         """
#         pass
