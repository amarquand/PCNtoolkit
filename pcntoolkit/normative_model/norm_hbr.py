import warnings
import numpy as np
from pcntoolkit.dataio.norm_data import NormData

from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


class NormHBR(NormBase):

    def __init__(self, norm_conf: NormConf, reg_conf: HBRConf):
        super().__init__( norm_conf)
        self._reg_conf: HBRConf = reg_conf
        self._model: HBR = HBR(HBRConf)

    def _fit(self, data: NormData):
        """
        Contains all the fitting logic that is specific to the regression model.
        """
        # some fitting logic
        # ...
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}")

    def _predict(self, data: NormData) -> NormData:
        """
        Contains all the prediction logic that is specific to the regression model.
        """
        # some prediction logic
        # ...
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}")

    def _fit_predict(self, data: NormData) -> NormData:
        """
        Contains all the fit_predict logic that is specific to the regression model.
        """
        # some fit_predict logic
        # ...
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}")

    def _transfer(self, data: NormData) -> NormBase:
        """
        Contains all the transfer logic that is specific to the regression model.
        """
        # some transfer logic
        # ...
        raise NotImplementedError(
            f"Transfer method not implemented for {self.__class__.__name__}")

    def _merge(self, other: NormBase):
        """
        Contains all the merge logic that is specific to the regression model.
        """
        # some merge logic
        # ...
        raise NotImplementedError(
            f"Merge method not implemented for {self.__class__.__name__}")

    def _tune(self, data: NormData):
        """
        Contains all the tuning logic that is specific to the regression model.
        """
        # some tuning logic
        # ...
        raise NotImplementedError(
            f"Tune method not implemented for {self.__class__.__name__}")

    def _extend(self, data: NormData):
        """
        Contains all the extension logic that is specific to the regression model.
        """
        # some extension logic
        # ...
        raise NotImplementedError(
            f"Extend method not implemented for {self.__class__.__name__}")

    def evaluate_mse(self, data: NormData) -> np.float32:
        """
        Contains all the evaluation logic that is specific to the regression model.
        """
        # some evaluation logic
        # ...
        warnings.warn(
            f"MSE not implemented for {self.__class__.__name__}, returning NAN")
        return np.NAN

    def evaluate_mae(self, data: NormData) -> np.float32:
        """
        Contains all the evaluation logic that is specific to the regression model.
        """
        # some evaluation logic
        # ...
        warnings.warn(
            f"MAE not implemented for {self.__class__.__name__}, returning NAN")
        return np.NAN

    def evaluate_r2(self, data: NormData) -> np.float32:
        """
        Contains all the evaluation logic that is specific to the regression model.
        """
        # some evaluation logic
        # ...
        warnings.warn(
            f"R2 not implemented for {self.__class__.__name__}, returning NAN")
        return np.NAN

    def load(self) -> NormBase:
        """
        Contains all the loading logic that is specific to the regression model.
        """
        # some loading logic
        # ...
        raise NotImplementedError(
            f"Load method not implemented for {self.__class__.__name__}")

    def save(self):
        """
        Contains all the saving logic that is specific to the regression model.
        """
        # some saving logic
        # ...
        raise NotImplementedError(
            f"Save method not implemented for {self.__class__.__name__}")
