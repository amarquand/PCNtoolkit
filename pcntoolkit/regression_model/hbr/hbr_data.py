import pymc as pm
import numpy as np
class HBRData:

    def __init__(self, X:np.ndarray, y:np.ndarray, batch_effects:np.ndarray):
        self.X = X
        self.y = y
        self.batch_effects = batch_effects

    def add_to_pymc_model(self, model:pm.Model) -> None:
        """
        Adds the data to the pymc model.
        """
        with model:
            self.pm_X = pm.MutableData("X", self.X)
            self.pm_y = pm.MutableData("y", self.y)
            self.pm_batch_effects = pm.MutableData("batch_effects", self.batch_effects)
