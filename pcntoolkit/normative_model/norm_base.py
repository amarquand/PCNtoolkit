from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

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
            self.model = self.models[responsevar]

            # Fit and predict
            self._fit_predict(resp_fit_data, resp_predict_data)

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
        results = {}

        results["Yhat"] = self.compute_yhat(data)
        # results["S2"] = self.compute_s2(data)
        # results["Z"] = self.evaluate_zscores(data)

        # results["Rho"] = self.evaluate_rho(data)
        # results["RMSE"] = self.evaluate_rmse(data)
        # results["SMSE"] = self.evaluate_smse(data)
        # results["EXPV"] = self.evaluate_expv(data)
        # results["MSLL"] = self.evaluate_msll(data)
        # results["NLL"] = self.evaluate_nll(data)
        # results["BIC"] = self.evaluate_bic(data)

        return results

    @abstractmethod
    def compute_yhat(self, data: NormData) -> float:
        pass

    @abstractmethod
    def compute_s2(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_rho(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_rmse(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_smse(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_expv(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_msll(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_nll(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_bic(self, data: NormData) -> float:
        pass

    @abstractmethod
    def evaluate_zscores(self, data: NormData):
        pass

    @abstractmethod
    def _fit_predict(self, data: NormData):
        """
        Acts as the adapter for fit_predict using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
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
    def quantiles(self, data: NormData, quantiles: list[float]):
        pass

    # def save(self):
    #     """
    #     Saves the normative model to a directory.
    #     """
    #     path = self.norm_conf.save_dir
    #     if not os.path.exists(path):
    #         os.makedirs(path, exist_ok=True)
    #     self._save()

    def save(self):
        model_dict = {}
        model_dict["response_vars"] = self.response_vars
        model_dict["norm_conf"] = self.norm_conf.to_dict()
        model_dict["reg_conf"] = self.reg_conf.to_dict()
        model_dict["regression_models"] = self.models_to_dict()

        # Save the model_dict as json
        model_dict_path = os.path.join(
            self.norm_conf.save_dir, "normative_model_dict.json"
        )

        with open(model_dict_path, "w") as f:
            json.dump(model_dict, f, indent=4)

    @classmethod
    def load(cls, path):
        # Load the model_dict from json
        model_dict_path = os.path.join(path, "normative_model_dict.json")

        with open(model_dict_path, "r") as f:
            model_dict = json.load(f)

        normconf = NormConf.from_args(model_dict["norm_conf"])
        regconf = cls.reg_conf_from_dict(model_dict["reg_conf"])
        normative_model = cls(normconf, regconf)

        normative_model.response_vars = model_dict["response_vars"]
        normative_model.dict_to_models(model_dict["regression_models"])
        return normative_model

    @staticmethod
    @abstractmethod
    def reg_conf_from_dict(dict):
        """
        Creates a regression configuration from a dictionary.
        """
        pass

    @abstractmethod
    def models_to_dict():
        """
        Returns a dictionary describing the regression models.
        """
        pass

    @abstractmethod
    def dict_to_models():
        """
        Creates the self.models attribute from a dictionary.
        """
        pass

    # @classmethod
    # @abstractmethod
    # def load(cls, path) -> "NormBase":
    #     """
    #     Contains all the loading logic that is specific to the regression model.
    #     Path is a string that points to the directory where the model should be loaded from.
    #     """
    #     pass

    @property
    def norm_conf(self):
        return self._norm_conf

    def preprocess(self, data: NormData) -> NormData:
        """
        Contains all the general preprocessing logic that is not specific to the regression model.
        """
        self.scale_forward(data)

        # data.scale_forward(self.norm_conf.inscaler, self.norm_conf.outscaler)
        data.expand_basis(self.norm_conf.basis_function)

    def postprocess(self, data: NormData) -> NormData:
        """
        Contains all the general postprocessing logic that is not specific to the regression model.
        """
        data.scale_backward()

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
