from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from scipy import stats
from sklearn.metrics import explained_variance_score

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.dataio.scaler import scaler

from .norm_conf import NormConf


class NormBase(ABC):  # newer abstract base class syntax, no more python2
    def __init__(self, norm_conf: NormConf):
        self._norm_conf: NormConf = norm_conf
        object.__setattr__(
            self._norm_conf, "normative_model_name", self.__class__.__name__
        )
        self.response_vars: list = None
        self.models = {}
        self.inscalers = {}
        self.outscalers = {}

    @property
    def norm_conf(self):
        return self._norm_conf

    def save(self):
        model_dict = {}
        # Store the response variables
        model_dict["response_vars"] = self.response_vars
        # Store the normative model configuration
        model_dict["norm_conf"] = self.norm_conf.to_dict()
        # Store the regression model configuration
        model_dict["reg_conf"] = self.reg_conf.to_dict()

        # Store the regression models
        model_dict["regression_models"] = self.models_to_dict(self.norm_conf.save_dir)

        # Store the scalers
        model_dict["inscalers"] = {k: v.to_dict() for k, v in self.inscalers.items()}
        model_dict["outscalers"] = {k: v.to_dict() for k, v in self.outscalers.items()}

        # Save the model_dict as json
        model_dict_path = os.path.join(
            self.norm_conf.save_dir, "normative_model_dict.json"
        )
        print("Saving normative model to", model_dict_path)
        with open(model_dict_path, "w") as f:
            json.dump(model_dict, f, indent=4)

    @classmethod
    def load(cls, path):
        # Load the model_dict from json
        print("Loading normative model from", path)
        model_dict_path = os.path.join(path, "normative_model_dict.json")
        with open(model_dict_path, "r") as f:
            model_dict = json.load(f)

        # Create the normative model
        normconf = NormConf.from_args(model_dict["norm_conf"])
        regconf = cls.reg_conf_from_args(model_dict["reg_conf"])
        normative_model = cls(normconf, regconf)

        # Set the response variables
        normative_model.response_vars = model_dict["response_vars"]

        # Set the regression models
        normative_model.dict_to_models(model_dict["regression_models"], path)

        # Set the scalers
        normative_model.inscalers = {
            k: scaler.from_dict(v) for k, v in model_dict["inscalers"].items()
        }
        normative_model.outscalers = {
            k: scaler.from_dict(v) for k, v in model_dict["outscalers"].items()
        }

        return normative_model

    def fit(self, data: NormData):
        """
        Contains all the general fitting logic that is not specific to the regression model.
        """
        # Preprocess the data
        self.preprocess(data)

        # Set self.response_vars
        self.response_vars = data.response_vars.to_numpy().copy().tolist()

        # Fit the model for each response variable
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_fit_data = data.sel(response_vars=responsevar)

            # Create a new model if it does not exist yet
            if not responsevar in self.models:
                self.models[responsevar] = self.model_type(self.reg_conf)

            # Set self.model to the current model
            self.current_responsevar = responsevar
            self.model = self.models[responsevar]

            # Fit the model
            self._fit(resp_fit_data)

    def predict(self, data: NormData) -> NormData:
        """
        Contains all the general prediction logic that is not specific to the regression model.
        This includes cv, logging, saving, etc. Calls the subclass' _predict method.
        """
        # Preprocess the data
        self.preprocess(data)

        # Predict for each response variable
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.models:
                raise ValueError(
                    f"Attempted to predict model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.current_responsevar = responsevar
            self.model = self.models[responsevar]

            # Predict
            self._predict(resp_predict_data)

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
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_fit_data = fit_data.sel(response_vars=responsevar)
            resp_predict_data = predict_data.sel(response_vars=responsevar)

            # Create a new model if it does not exist yet
            if not responsevar in self.models:
                self.models[responsevar] = self.model_type(self.reg_conf)

            # Set self.model to the current model
            self.current_responsevar = responsevar
            self.model = self.models[responsevar]

            # Fit and predict
            self._fit_predict(resp_fit_data, resp_predict_data)

        # predict_data.plot_quantiles()
        # Get the results
        results = self.evaluate(predict_data)
        return results

    def transfer(self, data: NormData) -> "NormBase":
        """
        Transfers the normative model to a new dataset. Calls the subclass' _transfer method.
        """
        # Preprocess the data
        self.preprocess(data)

        transfered_models = {}

        # Transfer for each response variable
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_transfer_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.models:
                raise ValueError(
                    f"Attempted to transfer a model that has not been fitted."
                )

            # Set self.model to the current model
            self.model = self.models[responsevar]

            # Transfer
            transfered_models[responsevar] = self._transfer(resp_transfer_data)

        # Create a new normative model
        transfered_normative_model = self.__class__(self.norm_conf, self.reg_conf)

        # Set the models
        transfered_normative_model.models = transfered_models

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
        self.compute_quantiles(data, [-1.0, 0.0, 1.0])

        results = {}
        data["Yhat"] = data.quantiles.sel(quantile_zscores=0.0)
        data["S2"] = data.quantiles.sel(quantile_zscores=1.0)

        results["Rho"] = self.evaluate_rho(data)
        results["RMSE"] = self.evaluate_rmse(data)
        results["SMSE"] = self.evaluate_smse(data)
        results["EXPV"] = self.evaluate_expv(data)
        # results["MSLL"] = self.evaluate_msll(data)
        # results["NLL"] = self.evaluate_nll(data)
        results["BIC"] = self.evaluate_bic(data)

        return results

    def compute_quantiles(self, data: NormData, z_scores: list[float]):
        # Preprocess the data
        self.preprocess(data)

        # Create an empty array to store the scaledquantiles
        data["scaled_quantiles"] = xr.DataArray(
            np.zeros((len(z_scores), data.X.shape[0], len(self.response_vars))),
            dims=("quantile_zscores", "datapoints", "response_vars"),
            coords={"quantile_zscores": z_scores},
        )

        # Predict for each response variable
        for i, responsevar in enumerate(self.response_vars):
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.models:
                raise ValueError(
                    f"Attempted to find quantiles for model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.current_responsevar = responsevar
            self.model = self.models[responsevar]

            # Overwrite quantiles
            data["scaled_quantiles"].loc[
                {"response_vars": responsevar}
            ] = self._quantiles(resp_predict_data, z_scores)

        self.postprocess(data)

    def compute_zscores(self, data: NormData):
        # Preprocess the data
        self.preprocess(data)

        # Create an empty array to store the zscores
        data["zscores"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(self.response_vars))),
            dims=("datapoints", "response_vars"),
        )

        # Predict for each response variable
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if not responsevar in self.models:
                raise ValueError(
                    f"Attempted to find zscores for model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.current_responsevar = responsevar
            self.model = self.models[responsevar]

            # Overwrite zscores
            data["zscores"].loc[{"response_vars": responsevar}] = self._zscores(
                resp_predict_data
            )

        self.postprocess(data)

    def evaluate_rho(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        rho, _ = stats.spearmanr(y, yhat)
        return rho

    def evaluate_rmse(self, data: NormData):
        y = data["y"].values
        yhat = data["Yhat"].values

        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return rmse

    def evaluate_smse(self, data: NormData):
        y = data["y"].values
        yhat = data["Yhat"].values

        mse = np.mean((y - yhat) ** 2)
        variance = np.var(y)
        smse = mse / variance if variance != 0 else 0

        return smse

    def evaluate_expv(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        expv = explained_variance_score(y, yhat)
        return expv

    def evaluate_msll(self, data: NormData) -> float:
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

    def evaluate_nll(self, data: NormData) -> float:
        # TODO check if this is correct

        # assume 'Y' is binary (0 or 1)
        y = data["y"].values
        yhat = data["Yhat"].values

        # Calculate the NLL
        nll = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return nll

    def evaluate_bic(self, data: NormData) -> float:
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

    def set_save_dir(self, save_dir):
        self.norm_conf.set_save_dir(save_dir)

    def set_log_dir(self, log_dir):
        self.norm_conf.set_log_dir(log_dir)

    #######################################################################################################

    # all the methods below are abstract methods, which means they have to be implemented in the subclass

    #######################################################################################################

    @classmethod
    @abstractmethod
    def from_args(cls, args):
        """
        Creates a normative model from command line arguments.
        """
        pass

    @staticmethod
    @abstractmethod
    def reg_conf_from_args(dict):
        """
        Creates a regression configuration from a dictionary.
        """
        pass

    @abstractmethod
    def models_to_dict(self, path=None):
        """
        Returns a dictionary describing the regression models.
        This dictionary is used to save the normative model to disk.
        Takes an optional path argument, which can be used to save large model components to disk.
        """
        pass

    @abstractmethod
    def dict_to_models(self, dict, path=None):
        """
        Creates the self.models attribute from a dictionary.
        This dictionary is loaded from disk, and is used to restore the normative model.
        Takes an optional path argument, which can be used to load large model components from disk.
        """
        pass

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
    def _fit_predict(self, data: NormData):
        """
        Acts as the adapter for fit_predict using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass

    @abstractmethod
    def _transfer(self, data: NormData) -> "NormBase":
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
    def _quantiles(self, data: NormData, quantiles: list[float]) -> xr.DataArray:
        """Takes a list of quantiles and returns the corresponding quantiles of the model.
        The return type is an xr.datarray with dimensions (quantile_zscores, datapoints).
        """
        pass

    @abstractmethod
    def _zscores(self, data: NormData) -> xr.DataArray:
        """Returns the zscores of the model.
        The return type is an xr.datarray with dimensions (datapoints)."""
        pass

    @abstractmethod
    def n_params(self):
        """
        Returns the number of parameters of the model.
        """
        pass
