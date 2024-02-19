from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from ast import List
from dataclasses import field
from typing import Any, Union

import arviz as az
import numpy as np
import xarray as xr
from scipy import stats
from sklearn.metrics import explained_variance_score

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.dataio.scaler import scaler
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.gpr.gpr import GPR
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.reg_conf import RegConf
from pcntoolkit.regression_model.regression_model import RegressionModel

from .norm_conf import NormConf


class NormBase(ABC):
    """
    The NormBase class is the base class for all normative models.
    This class holds a number of regression models, one for each response variable.
    The class contains methods for fitting, predicting, transferring, extending, tuning, merging, and evaluating the normative model.
    This class contains the general logic that is not specific to the regression model. The subclasses implement the actual logic for fitting, predicting, etc.
    All the bookkeeping is done in this class, such as keeping track of the regression models, the scalers, the response variables, etc.
    """

    def __init__(self, norm_conf: NormConf):
        self._norm_conf: NormConf = norm_conf
        object.__setattr__(
            self._norm_conf, "normative_model_name", self.__class__.__name__
        )

        # Response variables is a list of names of the response variables for which the model is fitted
        self.response_vars: list[str] = None

        # the regression_model_type attribute is used to store the type of regression model
        # should be set by the subclass
        self.regression_model_type = None

        # the self.defult_reg_conf attribute is used whenever a new regression model is created, and no reg_conf is provided
        # should be set by the subclass
        self.default_reg_conf: RegConf = None

        # Regression models is a dictionary that contains the regression models
        # - the keys are the response variables
        # - the values are the regression models
        self.regression_models: dict[str, RegressionModel] = {}

        # the self.current_regression_model attribute is used to store the current regression model
        # this model is used internally by the _fit and _predict methods of the subclass
        self.current_regression_model: RegressionModel = None

        # Inscalers contains a scaler for each covariate
        self.inscalers = {}

        # Outscalers contains a scaler for each response variable
        self.outscalers = {}

    @property
    def norm_conf(self):
        return self._norm_conf

    def fit(self, data: NormData):
        """
        Contains all the general fitting logic that is not specific to the regression model.
        """
        # Preprocess the data
        self.preprocess(data)

        # Set self.response_vars
        self.response_vars = data.response_vars.to_numpy().copy().tolist()

        # Fit the model for each response variable
        print(f"Going to fit {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_fit_data = data.sel(response_vars=responsevar)

            # Set self.current_regression_model to the current model
            self.prepare(responsevar)

            # Fit the model
            print(f"Fitting model for {responsevar}")
            self._fit(resp_fit_data)

            self.reset()

    def predict(self, data: NormData) -> NormData:
        """
        Contains all the general prediction logic that is not specific to the regression model.
        This includes cv, logging, saving, etc. Calls the subclass' _predict method.
        """
        # Preprocess the data
        self.preprocess(data)

        # Predict for each response variable
        print(f"Going to predict {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.regression_models:
                raise ValueError(
                    f"Attempted to predict model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Predict
            print(f"Predicting model for {responsevar}")
            self._predict(resp_predict_data)

            self.reset()

        # Return the results
        results = self.evaluate(data)
        return results

    def fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Contains all the general fit_predict logic that is not specific to the regression model.
        This includes cv, logging, saving, etc. Calls the subclass' _fit_predict method.
        """

        assert fit_data.is_compatible_with(
            predict_data
        ), "Fit data and predict data are not compatible!"

        # Preprocess the data
        self.preprocess(fit_data)
        self.preprocess(predict_data)

        # Set self.response_vars
        self.response_vars = fit_data.response_vars.to_numpy().copy().tolist()

        # Fit and predict for each response variable
        print(f"Going to fit and predict {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_fit_data = fit_data.sel(response_vars=responsevar)
            resp_predict_data = predict_data.sel(response_vars=responsevar)

            # Create a new model if it does not exist yet
            if not responsevar in self.regression_models:
                self.regression_models[responsevar] = self.regression_model_type(
                    responsevar, self.default_reg_conf
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Fit and predict
            print(f"Fitting and predicting model for {responsevar}")
            self._fit_predict(resp_fit_data, resp_predict_data)

            self.reset()

        # predict_data.plot_quantiles()
        # Get the results
        results = self.evaluate(predict_data)
        return results

    def transfer(self, data: NormData, *args, **kwargs) -> "NormBase":
        """
        Transfers the normative model to a new dataset. Calls the subclass' _transfer method.
        """
        # Preprocess the data
        self.preprocess(data)

        transfered_models = {}

        # Transfer for each response variable
        print(f"Going to transfer {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_transfer_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.regression_models:
                raise ValueError(
                    f"Attempted to transfer a model that has not been fitted."
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Transfer
            print(f"Transferring model for {responsevar}")
            transfered_models[responsevar] = self._transfer(
                resp_transfer_data, *args, **kwargs
            )

            self.reset()

        # Create a new normative model
        # Change the reg_conf save_dir and log_dir
        transfered_norm_conf = self.norm_conf.to_dict()
        transfered_norm_conf["save_dir"] = self.norm_conf.save_dir + "_transfer"
        transfered_norm_conf["log_dir"] = self.norm_conf.log_dir + "_transfer"
        transfered_norm_conf = NormConf.from_dict(transfered_norm_conf)

        transfered_normative_model = self.__class__(
            transfered_norm_conf, self.default_reg_conf
        )

        # Set the models
        transfered_normative_model.response_vars = (
            data.response_vars.to_numpy().copy().tolist()
        )
        transfered_normative_model.regression_models = transfered_models

        # Return the new model
        return transfered_normative_model

    def extend(self, data: NormData):
        """
        Extends the normative model with new data. Calls the subclass' _extend method.
        """
        # some preparations and preprocessing
        # ...

        result = self._extend(data)

        # some cleanup and postprocessing
        # ...

        return result

    def tune(self, data: NormData):
        """
        Tunes the normative model. Calls the subclass' _tune method.
        """
        # some preparations and preprocessing
        # ...

        result = self._tune(data)

        # some cleanup and postprocessing
        # ...

        return result

    def merge(self, other: "NormBase"):
        """
        Merges the normative model with another normative model. Calls the subclass' _merge method.
        """
        # some preparations and preprocessing
        # ...

        if not self.__class__ == other.__class__:
            raise ValueError("Attempted to merge two different normative models.")

        result = self._merge(other)

        # some cleanup and postprocessing
        # ...

        return result

    def evaluate(self, data: NormData):
        """
        Contains evaluation logic.
        """
        self.compute_zscores(data)
        self.compute_quantiles(data)
        self.compute_measures(data)

    def compute_measures(self, data: NormData):
        # TODO fix this
        data["Yhat"] = data.quantiles.sel(quantile_zscores=0, method="nearest")
        data["S2"] = (
            data.quantiles.sel(quantile_zscores=1, method="nearest") - data["Yhat"]
        ) ** 2
        self.create_measures_group(data)
        self.evaluate_bic(data)
        self.evaluate_rho(data)
        self.evaluate_rmse(data)
        self.evaluate_smse(data)
        self.evaluate_expv(data)
        # self.evaluate_msll(data)
        self.evaluate_nll(data)

    def create_measures_group(self, data):
        data["measures"] = xr.DataArray(
            np.nan * np.ones((len(self.response_vars), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": self.response_vars,
                "statistics": ["Rho", "RMSE", "SMSE", "ExpV", "MSLL", "BIC"],
            },
        )

    def evaluate_rho(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            rho = self._evaluate_rho(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "Rho"}] = rho

    def evaluate_rmse(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            rmse = self._evaluate_rmse(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "RMSE"}] = (
                rmse
            )

    def evaluate_smse(self, data: NormData):
        data["SMSE"] = self.empty_measure()
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            smse = self._evaluate_smse(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "SMSE"}] = (
                smse
            )

    def evaluate_expv(self, data: NormData):
        data["ExpV"] = self.empty_measure()
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            expv = self._evaluate_expv(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "ExpV"}] = (
                expv
            )

    def evaluate_msll(self, data: NormData):
        data["MSLL"] = self.empty_measure()
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            msll = self._evaluate_msll(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "MSLL"}] = (
                msll
            )

    def evaluate_nll(self, data: NormData):
        data["NLL"] = self.empty_measure()
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            nll = self._evaluate_nll(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "NLL"}] = nll

    def evaluate_bic(self, data: NormData):
        data["BIC"] = self.empty_measure()
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            bic = self._evaluate_bic(resp_predict_data)

            self.prepare(responsevar)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "BIC"}] = bic

            self.reset()

    def _evaluate_rho(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        rho, _ = stats.spearmanr(y, yhat)
        return rho

    def _evaluate_rmse(self, data: NormData):
        y = data["y"].values
        yhat = data["Yhat"].values

        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return rmse

    def _evaluate_smse(self, data: NormData):
        y = data["y"].values
        yhat = data["Yhat"].values

        mse = np.mean((y - yhat) ** 2)
        variance = np.var(y)
        smse = mse / variance if variance != 0 else 0

        return smse

    def _evaluate_expv(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        expv = explained_variance_score(y, yhat)
        return expv

    def _evaluate_msll(self, data: NormData) -> float:
        # TODO check if this is correct

        y = data["y"].values
        yhat = data["Yhat"].values
        yhat_std = data["Yhat_std"]

        # Calculate the log loss of the model's predictions
        log_loss = np.mean((y - yhat) ** 2 / (2 * yhat_std**2) + np.log(yhat_std))

        # Calculate the log loss of the naive model
        naive_std = np.std(y)
        naive_log_loss = np.mean(
            (y - np.mean(y)) ** 2 / (2 * naive_std**2) + np.log(naive_std)
        )

        # Calculate MSLL
        msll = log_loss - naive_log_loss

        return msll

    def _evaluate_nll(self, data: NormData) -> float:
        # TODO check if this is correct

        # assume 'Y' is binary (0 or 1)
        y = data["y"].values
        yhat = data["Yhat"].values

        # Calculate the NLL
        nll = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return nll

    def _evaluate_bic(self, data: NormData) -> float:
        n_params = self.n_params()

        # Assuming 'data' is a NormData object with 'Y' and 'Yhat' DataArrays
        y = data["y"].values
        yhat = data["Yhat"].values

        # Calculate the residual sum of squares
        rss = np.sum((y - yhat) ** 2)

        # Calculate the number of observations
        n = len(y)

        # Calculate the BIC
        bic = n * np.log(rss / n) + n_params * np.log(n)

        return bic

    def empty_measure(self):
        return xr.DataArray(
            np.zeros(len(self.response_vars)),
            dims=("response_vars"),
            coords={"response_vars": self.response_vars},
        )

    def compute_quantiles(self, data: NormData, *args, **kwargs):
        # Preprocess the data
        self.preprocess(data)

        quantiles_zscores = np.arange(-4, 4.1, 1.0)
        # Create an empty array to store the scaledquantiles
        data["scaled_quantiles"] = xr.DataArray(
            np.zeros(
                (len(quantiles_zscores), data.X.shape[0], len(self.response_vars))
            ),
            dims=("quantile_zscores", "datapoints", "response_vars"),
            coords={"quantile_zscores": quantiles_zscores},
        )

        # Predict for each response variable
        for i, responsevar in enumerate(self.response_vars):
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.regression_models:
                raise ValueError(
                    f"Attempted to find quantiles for model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Overwrite quantiles
            print("Computing quantiles for", responsevar)
            data["scaled_quantiles"].loc[{"response_vars": responsevar}] = (
                self._quantiles(resp_predict_data, quantiles_zscores, *args, **kwargs)
            )

            self.reset()

        self.postprocess(data)

    def compute_zscores(self, data: NormData, *args, **kwargs):
        # Preprocess the data
        self.preprocess(data)

        # Create an empty array to store the zscores
        data["zscores"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(self.response_vars))),
            dims=("datapoints", "response_vars"),
            coords={"datapoints": data.datapoints, "response_vars": self.response_vars},
        )

        # Predict for each response variable
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.regression_models:
                raise ValueError(
                    f"Attempted to find zscores for model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Overwrite zscores
            print("Computing zscores for", responsevar)
            data["zscores"].loc[{"response_vars": responsevar}] = self._zscores(
                resp_predict_data, *args, **kwargs
            )

            self.reset()

        self.postprocess(data)

    def preprocess(self, data: NormData) -> NormData:
        """
        Contains all the general preprocessing logic that is not specific to the regression model.
        """
        self.scale_forward(data)

        # data.scale_forward(self.norm_conf.inscaler, self.norm_conf.outscaler)
        data.expand_basis(
            self.norm_conf.basis_function, basis_column=self.norm_conf.basis_column
        )

    def postprocess(self, data: NormData) -> NormData:
        """
        Contains all the general postprocessing logic that is not specific to the regression model.
        """
        self.scale_backward(data)

    def scale_forward(self, data: NormData, overwrite=False):
        """
        Contains all the general scaling logic that is not specific to the regression model.
        """
        for covariate in data.covariates.to_numpy():
            if (not covariate in self.inscalers) or overwrite:
                self.inscalers[covariate] = scaler(self.norm_conf.inscaler)
                self.inscalers[covariate].fit(data.X.sel(covariates=covariate).data)

        for responsevar in data.response_vars.to_numpy():
            if (not responsevar in self.outscalers) or overwrite:
                self.outscalers[responsevar] = scaler(self.norm_conf.outscaler)
                self.outscalers[responsevar].fit(
                    data.y.sel(response_vars=responsevar).data
                )

        data.scale_forward(self.inscalers, self.outscalers)

    def scale_backward(self, data: NormData):
        """
        Contains all the general scaling logic that is not specific to the regression model.
        """
        data.scale_backward(self.inscalers, self.outscalers)

    def save(self):
        model_dict = self.to_dict(self.norm_conf.save_dir)

        # Save the model_dict as json
        model_dict_path = os.path.join(
            self.norm_conf.save_dir, "normative_model_dict.json"
        )
        print("Saving normative model to", model_dict_path)
        with open(model_dict_path, "w") as f:
            json.dump(model_dict, f, indent=4)

    @classmethod
    def load(cls, path):
        """
        Load a normative model from disk.
        """

        # Load the model_dict from json
        print("Loading normative model from", path)
        model_dict_path = os.path.join(path, "normative_model_dict.json")
        with open(model_dict_path, "r") as f:
            model_dict = json.load(f)

        normative_model = cls.from_dict(model_dict, path)

        return normative_model

    @classmethod
    def from_dict(cls, model_dict, path=None):
        # Create the normative model configuration
        normconf = NormConf.from_dict(model_dict["norm_conf"])

        # Create a normative model
        self = cls(normconf)

        # Set the response variables
        self.response_vars = model_dict["response_vars"]

        # Set the regression models
        self.regression_models = self.dict_to_regression_models(
            model_dict["regression_models"], path
        )

        # Set the scalers
        self.inscalers = {
            k: scaler.from_dict(v) for k, v in model_dict["inscalers"].items()
        }
        self.outscalers = {
            k: scaler.from_dict(v) for k, v in model_dict["outscalers"].items()
        }

        return self

    def to_dict(self, path=None):
        """
        Converts the normative model to a dictionary.
        Takes an optional path argument to save large model components
        """
        model_dict = {}
        # Store the response variables
        model_dict["response_vars"] = self.response_vars
        # Store the normative model configuration
        model_dict["norm_conf"] = self.norm_conf.to_dict()

        # Store the regression models
        model_dict["regression_models"] = self.regression_models_to_dict(path)

        # Store the scalers
        model_dict["inscalers"] = {k: v.to_dict() for k, v in self.inscalers.items()}
        model_dict["outscalers"] = {k: v.to_dict() for k, v in self.outscalers.items()}

        return model_dict

    def regression_models_to_dict(self, path) -> dict[str, dict[str, Any]]:
        return {k: v.to_dict(path) for k, v in self.regression_models.items()}

    def dict_to_regression_models(self, model_dict, path) -> dict[str, RegressionModel]:
        return {
            k: self.regression_model_type.from_dict(v, path)
            for k, v in model_dict.items()
        }

    def set_save_dir(self, save_dir):
        self.norm_conf.set_save_dir(save_dir)

    def set_log_dir(self, log_dir):
        self.norm_conf.set_log_dir(log_dir)

    def prepare(self, responsevar):
        self.current_response_var = responsevar
        # Create a new model if it does not exist yet
        if not responsevar in self.regression_models:
            self.regression_models[responsevar] = self.regression_model_type(
                responsevar, self.get_reg_conf(responsevar)
            )
        self.current_regression_model = self.regression_models.get(responsevar, None)

    def get_reg_conf(self, responsevar):
        if responsevar in self.regression_models:
            return self.regression_models[responsevar].reg_conf
        else:
            return self.default_reg_conf

    def reset(self):
        pass

    #######################################################################################################

    # all the methods below are abstract methods, which means they have to be implemented in the subclass

    #######################################################################################################

    @abstractmethod
    def _fit(self, data: NormData) -> NormData:
        """
        Acts as the adapter for fitting the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass

    @abstractmethod
    def _predict(self, data: NormData) -> NormData:
        """
        Acts as the adapter for prediction using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass

    @abstractmethod
    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Acts as the adapter for fit_predict using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass

    @abstractmethod
    def _transfer(self, data: NormData, *args, **kwargs) -> "NormBase":
        pass

    @abstractmethod
    def _extend(self, data: NormData):
        pass

    @abstractmethod
    def _tune(self, data: NormData):
        pass

    @abstractmethod
    def _merge(self, other: "NormBase"):
        pass

    @abstractmethod
    def _quantiles(
        self, data: NormData, quantiles: list[float], *args, **kwargs
    ) -> xr.DataArray:
        """Takes a list of quantiles and returns the corresponding quantiles of the model.
        The return type is an xr.datarray with dimensions (quantile_zscores, datapoints).
        """
        pass

    @abstractmethod
    def _zscores(self, data: NormData, *args, **kwargs) -> xr.DataArray:
        """Returns the zscores of the model.
        The return type is an xr.datarray with dimensions (datapoints)."""
        pass

    @abstractmethod
    def n_params(self):
        """
        Returns the number of parameters of the model.
        """
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, args):
        """
        Creates a normative model from command line arguments.
        """
        pass
