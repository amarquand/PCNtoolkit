import warnings
import numpy as np
import os
import json
import arviz as az
from pcntoolkit.dataio.norm_data import NormData

from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.hbr import hbr_data
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


import pymc as pm


class NormHBR(NormBase):

    def __init__(self, norm_conf: NormConf, reg_conf: HBRConf):
        super().__init__(norm_conf)
        self._reg_conf: HBRConf = reg_conf
        self._model: HBR = HBR(self._reg_conf)

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_args(args)
        hbrconf = HBRConf.from_dict(args)
        self = cls(norm_conf, hbrconf)
        return self

    @property
    def model(self) -> 'HBR':
        return self._model

    @staticmethod
    def normdata_to_hbrdata(data: NormData) -> hbr_data.HBRData:
        hbrdata = hbr_data.HBRData(X=data.X.to_numpy(),
                                   y=data.y.to_numpy(),
                                   batch_effects=data.batch_effects.to_numpy(),
                                   covariate_dims=data.covariates.to_numpy(),
                                   batch_effect_dims=data.batch_effect_dims.to_numpy())
        hbrdata.set_batch_effects_maps(data.batch_effects_maps)
        return hbrdata

    def _fit(self, data: NormData):
        """
        Contains all the fitting logic that is specific to the regression model.
        """
        hbrdata = self.normdata_to_hbrdata(data)
        if not self.model.is_fitted:

            # Sample from pymc model
            if not self.model.model:
                self.model.create_pymc_model(hbrdata)

            with self.model.model:
                self.model.idata = pm.sample(
                    self.model.conf.draws, tune=self.model.conf.tune, cores=self.model.conf.cores)
            self.model.is_fitted = True

        else:
            raise RuntimeError("Model is already fitted.")

    def _predict(self, data: NormData) -> NormData:
        """
        Contains all the prediction logic that is specific to the regression model.
        """
        # some prediction logic
        # ...
        assert self.model.is_fitted, "Model must be fitted before predicting."
        hbrdata = self.normdata_to_hbrdata(data)
        hbrdata.set_data_in_existing_model(self.model.model)
        with self.model.model:
            pm.sample_posterior_predictive(
                self.model.idata, return_inferencedata=True, extend_inferencedata=True)

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Contains all the fit_predict logic that is specific to the regression model.
        """
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        if not self.model.is_fitted:

            # Sample from pymc model
            if not self.model.model:
                self.model.create_pymc_model(fit_hbrdata)

            with self.model.model:
                self.model.idata = pm.sample(
                    self.model.conf.draws, tune=self.model.conf.tune, cores=self.model.conf.cores)
            self.model.is_fitted = True

            predict_hbrdata = self.normdata_to_hbrdata(predict_data)
            predict_hbrdata.set_data_in_existing_model(self.model.model)
            with self.model.model:
                self.model.idata = pm.sample_posterior_predictive(
                    self.model.idata)

        else:
            raise RuntimeError("Model is already fitted.")

    def _transfer(self, data: NormData) -> HBR:
        """
        Contains all the transfer logic that is specific to the regression model.
        """
        transferdata = self.normdata_to_hbrdata(data)
        if not self.model.is_fitted:
            raise RuntimeError(
                "Model needs to be fitted before it can be transferred")
        new_norm_hbr = NormHBR(self.norm_conf, self.model.conf)
        new_norm_hbr.model.create_pymc_model(transferdata, self.model.idata)
        with new_norm_hbr.model.model:
            new_norm_hbr.model.idata = pm.sample(
                new_norm_hbr.model.conf.draws, tune=new_norm_hbr.model.conf.tune, cores=new_norm_hbr.model.conf.cores)
        new_norm_hbr.model.is_fitted = True
        return new_norm_hbr

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

    def save(self):
        """
        Contains all the saving logic that is specific to the regression model.
        Path is a string that points to the directory where the model should be saved.
        """
        model_dict = {}
        model_dict['norm_conf'] = self.norm_conf.to_dict()
        model_dict['model'] = self.model.to_dict()

        if self.model.is_fitted:
            if hasattr(self.model, 'idata'):
                idata_path = os.path.join(self.norm_conf.save_dir, "idata.nc")
                self.model.idata.to_netcdf(idata_path)
                model_dict['model']['idata_path'] = idata_path
            else:
                raise RuntimeError(
                    "Model is fitted but does not have idata. This should not happen.")

        # Save the model_dict as json
        model_dict_path = os.path.join(
            self.norm_conf.save_dir, "normative_model_dict.json")

        with open(model_dict_path, 'w') as f:
            json.dump(model_dict, f)

    @classmethod
    def load(cls, path):
        """
        Contains all the loading logic that is specific to the regression model.
        Path is a string that points to the directory where the model should be loaded from.
        """
        # Load the model dict from the json
        model_path: str = os.path.join(path, "normative_model_dict.json")
        model_dict = json.load(open(model_path, 'r'))

        # Construct the normconf from the dict
        normconf = NormConf.from_dict(model_dict['norm_conf'])

        # Construct the model from the dict
        model = HBR.from_dict(model_dict['model'])

        # Construct the normative model from the normconf and the model
        normative_model = cls(normconf, model.conf)

        normative_model._model = model

        # Load the idata if it exists
        if 'idata_path' in model_dict['model']:
            idata_path = model_dict['model']['idata_path']
            normative_model._model.idata = az.from_netcdf(idata_path)
            normative_model._model.is_fitted = True

        return normative_model
