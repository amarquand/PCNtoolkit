import json
import os
import warnings

import arviz as az
import numpy as np
import pymc as pm

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.hbr import hbr_data
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


class NormHBR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: HBRConf):
        super().__init__(norm_conf)
        self.reg_conf: HBRConf = reg_conf
        self.model_type = HBR
        self.model: HBR = None

    @classmethod
    def from_dict(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_dict(args)
        hbrconf = HBRConf.from_dict(args)
        self = cls(norm_conf, hbrconf)
        return self

    @staticmethod
    def normdata_to_hbrdata(data: NormData) -> hbr_data.HBRData:
        if hasattr(data, "Phi") and data.Phi is not None:
            this_X = data.Phi.to_numpy()
            this_covariate_dims = data.basis_functions.to_numpy()
        elif hasattr(data, "scaled_X") and data.scaled_X is not None:
            this_X = data.scaled_X.to_numpy()
            this_covariate_dims = data.covariates.to_numpy()
        else:
            this_X = data.X.to_numpy()
            this_covariate_dims = data.covariates.to_numpy()

        if hasattr(data, "scaled_y") and data.scaled_y is not None:
            this_y = data.scaled_y.to_numpy()
        else:
            this_y = data.y.to_numpy()

        hbrdata = hbr_data.HBRData(
            X=this_X,
            y=this_y,
            batch_effects=data.batch_effects.to_numpy(),
            covariate_dims=this_covariate_dims,
            batch_effect_dims=data.batch_effect_dims.to_numpy(),
        )
        hbrdata.set_batch_effects_maps(data.batch_effects_maps)
        return hbrdata

    def _fit(self, data: NormData, make_new_model=False):
        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        # Assert that the model is not already fitted
        if not self.model.is_fitted:
            # Make a new model if needed
            if make_new_model or (not self.model.model):
                self.model.create_pymc_model(hbrdata)

            # Sample from pymc model
            with self.model.model:
                self.model.idata = pm.sample(
                    self.reg_conf.draws,
                    tune=self.reg_conf.tune,
                    cores=self.reg_conf.cores,
                    chains=self.reg_conf.chains,
                )

            # Set the is_fitted flag to True
            self.model.is_fitted = True

        else:
            raise RuntimeError("Model is already fitted.")

    def _predict(self, data: NormData) -> NormData:
        # Assert that the model is fitted
        assert self.model.is_fitted, "Model must be fitted before predicting."

        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        # Create a new pymc model if needed
        if self.model.model is None:
            self.model.create_pymc_model(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.model.model)

        # Sample from the posterior predictive
        with self.model.model:
            pm.sample_posterior_predictive(self.model.idata, extend_inferencedata=True)

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        # Transform the data to hbrdata
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        predict_hbrdata = self.normdata_to_hbrdata(predict_data)

        # Assert that the model is not already fitted
        if not self.model.is_fitted:
            # Make a new model if needed
            if not self.model.model:
                self.model.create_pymc_model(fit_hbrdata)

            # Sample from pymc model
            with self.model.model:
                self.model.idata = pm.sample(
                    self.reg_conf.draws,
                    tune=self.reg_conf.tune,
                    cores=self.reg_conf.cores,
                    chains=self.reg_conf.chains,
                )

            # Set the is_fitted flag to True
            self.model.is_fitted = True

            # Set the data in the model
            predict_hbrdata.set_data_in_existing_model(self.model.model)

            # Sample from the posterior predictive
            with self.model.model:
                pm.sample_posterior_predictive(
                    self.model.idata, extend_inferencedata=True
                )
        else:
            raise RuntimeError("Model is already fitted.")

    def _transfer(self, data: NormData) -> "HBR":
        # Transform the data to hbrdata
        transferdata = self.normdata_to_hbrdata(data)

        # Assert that the model is fitted
        if not self.model.is_fitted:
            raise RuntimeError("Model needs to be fitted before it can be transferred")

        new_hbr_model = HBR(self.reg_conf)

        # Create a new model, using the idata from the original model to inform the priors
        new_hbr_model.create_pymc_model(transferdata, self.model.idata)

        # Sample using the new model
        with new_hbr_model.model:
            new_hbr_model.idata = pm.sample(
                draws=self.reg_conf.draws,
                tune=self.reg_conf.tune,
                cores=self.reg_conf.cores,
                chains=self.reg_conf.chains,
            )
            new_hbr_model.is_fitted = True

        # Return the new model
        return new_hbr_model

    def _merge(self, other: NormBase):
        """
        Contains all the merge logic that is specific to the regression model.
        """
        # some merge logic
        # ...
        raise NotImplementedError(
            f"Merge method not implemented for {self.__class__.__name__}"
        )

    def _tune(self, data: NormData):
        """
        Contains all the tuning logic that is specific to the regression model.
        """
        # some tuning logic
        # ...
        raise NotImplementedError(
            f"Tune method not implemented for {self.__class__.__name__}"
        )

    def _extend(self, data: NormData):
        """
        Contains all the extension logic that is specific to the regression model.
        """
        # some extension logic
        # ...
        raise NotImplementedError(
            f"Extend method not implemented for {self.__class__.__name__}"
        )

    def _save(self):
        """
        Contains all the saving logic that is specific to the regression model.
        Path is a string that points to the directory where the model should be saved.
        """
        model_dict = {}
        model_dict["response_vars"] = self.response_vars
        model_dict["norm_conf"] = self.norm_conf.to_dict()
        model_dict["reg_conf"] = self.reg_conf.to_dict()
        model_dict["regression_models"] = {}

        for k, v in self.models.items():
            model_dict["regression_models"][k] = v.to_dict()
            del model_dict["regression_models"][k]["conf"]
            if v.is_fitted:
                if hasattr(v, "idata"):
                    idata_path = os.path.join(self.norm_conf.save_dir, f"idata_{k}.nc")
                    self.model.idata.to_netcdf(idata_path)
                    model_dict["regression_models"][k]["idata_path"] = idata_path
                else:
                    raise RuntimeError(
                        "HBR model is fitted but does not have idata. This should not happen."
                    )

        # Save the model_dict as json
        model_dict_path = os.path.join(
            self.norm_conf.save_dir, "normative_model_dict.json"
        )

        with open(model_dict_path, "w") as f:
            json.dump(model_dict, f, indent=4)

    def regression_model_dict(self):
        regression_model_dict = {}

        for k, v in self.models.items():
            regression_model_dict[k] = v.to_dict()
            del regression_model_dict[k]["conf"]
            if v.is_fitted:
                if hasattr(v, "idata"):
                    idata_path = os.path.join(self.norm_conf.save_dir, f"idata_{k}.nc")
                    self.model.idata.to_netcdf(idata_path)
                    regression_model_dict[k]["idata_path"] = idata_path
                else:
                    raise RuntimeError(
                        "HBR model is fitted but does not have idata. This should not happen."
                    )
        return regression_model_dict

    @classmethod
    def load(cls, path):
        """
        Contains all the loading logic that is specific to the regression model.
        Path is a string that points to the directory where the model should be loaded from.
        """
        # Load the model dict from the json
        model_path: str = os.path.join(path, "normative_model_dict.json")
        model_dict = json.load(open(model_path, "r"))

        # Construct the normconf from the dict
        normconf = NormConf.from_dict(model_dict["norm_conf"])

        # Construct the regression conf from the dict
        regconf = HBRConf.from_dict(model_dict["reg_conf"])

        # Construct the normative model from the normconf and the model
        normative_model = cls(normconf, regconf)

        normative_model.response_vars = []

        # Construct the regression models from the dict
        for k, v in model_dict["regression_models"].items():
            model = HBR(regconf)
            model.is_from_dict = v["is_from_dict"]
            model.is_fitted = v["is_fitted"]
            if "idata_path" in v:
                model.idata = az.from_netcdf(v["idata_path"])
            normative_model.models[k] = model
            normative_model.response_vars.append(k)

        return normative_model

    def evaluate_bic(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate bic method not implemented for {self.__class__.__name__}"
        )

    def evaluate_expv(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate expv method not implemented for {self.__class__.__name__}"
        )

    def evaluate_msll(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate msll method not implemented for {self.__class__.__name__}"
        )

    def evaluate_nll(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate nll method not implemented for {self.__class__.__name__}"
        )

    def evaluate_rho(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate rho method not implemented for {self.__class__.__name__}"
        )

    def evaluate_rmse(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate rmse method not implemented for {self.__class__.__name__}"
        )

    def evaluate_smse(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Evaluate smse method not implemented for {self.__class__.__name__}"
        )

    def evaluate_zscores(self, data: NormData):
        raise NotImplementedError(
            f"Evaluate zscores method not implemented for {self.__class__.__name__}"
        )

    def compute_s2(self, data: NormData) -> float:
        raise NotImplementedError(
            f"Compute s2 method not implemented for {self.__class__.__name__}"
        )

    def compute_yhat(self, data: NormData) -> float:
        return np.random.randn(1).item()
        # raise NotImplementedError(
        #     f"Compute yhat method not implemented for {self.__class__.__name__}"
        # )

    def quantiles(self, data: NormData, quantiles: list):
        raise NotImplementedError(
            f"Quantiles method not implemented for {self.__class__.__name__}"
        )
