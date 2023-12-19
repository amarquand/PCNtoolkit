
from pcntoolkit.regression_model.reg_conf import RegConf


class GPRConf(RegConf):
    # some configuration parameters
    # ...

    example_parameter: int = 0 # example parameter. This is an int and has a default value of 0.

    def detect_configuration_problems(self) -> str:
        """
        Detects problems in the configuration and returns them as a string.
        """
        configuration_problems = ""
        problem_count: int = 0

        def add_problem(problem: str):
            """
            Use this to accumulate the problems into a string
            """
            nonlocal problem_count
            nonlocal configuration_problems
            problem_count += 1
            configuration_problems += f"{problem_count}:\t{problem}\n"


        # some configuration checks
        # ...

        return configuration_problems