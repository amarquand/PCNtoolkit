import numpy as np

from .blr_conf import BLRConf


class BLR:

    def __init__(self, conf: BLRConf):
        self._conf: BLRConf = conf

    def example_function_using_example_parameter(self, my_int: int):
        """
        This is an example function that uses the example parameter.
        """
        return my_int + self._conf.example_parameter

    @property
    def conf(self) -> BLRConf:
        return self._conf
