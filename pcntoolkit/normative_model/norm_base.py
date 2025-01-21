"""
NormBase Module Documentation
===========================

This module provides the abstract base class for normative modeling implementations.

The module implements a flexible framework for creating, training, and applying normative models
to neuroimaging or other high-dimensional data. It supports various regression models and provides
functionality for data preprocessing, model fitting, prediction, and evaluation.

Classes
-------
NormBase
    Abstract base class for normative modeling implementations.

Notes
-----
The NormBase class is designed to be subclassed to implement specific normative modeling approaches.
It provides a comprehensive interface for:
- Model fitting and prediction
- Data preprocessing and postprocessing
- Model evaluation and visualization
- Model persistence (saving/loading)
- Transfer learning capabilities

The class supports multiple regression model types including:
- Bayesian Linear Regression (BLR)
- Hierarchical Bayesian Regression (HBR)

The class structure of a normative model (using BLR in this example) is:

- NormBLR (abstract base class)
    - NormConf
    - RegressionModels
        1. BLR (feature 1)
            - BLRConf
        2. BLR (feature 2)
            - BLRConf
        3. ...
"""

from __future__ import annotations

import copy
import fcntl
import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData

# pylint: disable=unused-import
from pcntoolkit.regression_model.blr.blr import BLR  # noqa: F401 # type: ignore

# from pcntoolkit.regression_model.gpr.gpr import GPR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.hbr.hbr import HBR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.reg_conf import RegConf
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.basis_function import BasisFunction, create_basis_function
from pcntoolkit.util.evaluator import Evaluator
from pcntoolkit.util.output import Errors, Messages, Output, Warnings
from pcntoolkit.util.plotter import plot_centiles, plot_qq
from pcntoolkit.util.scaler import Scaler

from .norm_conf import NormConf


class NormBase(ABC):
    """
    Abstract base class for normative modeling implementations.

    This class provides the foundation for building normative models, handling multiple
    response variables through separate regression models. It manages data preprocessing,
    model fitting, prediction, and evaluation.

    Parameters
    ----------
    norm_conf : NormConf
        Configuration object containing normative model parameters.

    Attributes
    ----------
    response_vars : list[str]
        List of response variable names.
    regression_model_type : Any
        Type of regression model being used.
    default_reg_conf : RegConf
        Default regression configuration.
    regression_models : dict[str, RegressionModel]
        Dictionary mapping response variables to their regression models.
    focused_var : str
        Currently focused response variable.
    evaluator : Evaluator
        Model evaluation utility instance.
    inscalers : dict
        Input data scalers.
    outscalers : dict
        Output data scalers.
    bspline_basis : Any
        B-spline basis for covariate expansion.

    """

    def __init__(self, norm_conf: NormConf):
        self._norm_conf: NormConf = norm_conf
        object.__setattr__(self._norm_conf, "normative_model_name", self.__class__.__name__)

        self.response_vars: list[str] = None  # type: ignore
        self.regression_model_type: Any = None  # type: ignore
        self.default_reg_conf: RegConf = None  # type: ignore
        self.regression_models: dict[str, RegressionModel] = {}
        self.focused_var: str = None  # type: ignore
        self.evaluator = Evaluator()
        self.inscalers: dict = {}
        self.outscalers: dict = {}
        self.basis_function: BasisFunction
        self.basis_column: Optional[int] = None
        self.is_fitted: bool = False

    def fit(self, data: NormData) -> None:
        """
        Fits a regression model for each response variable in the data.

        This method performs the following steps:
        1. Preprocesses the input data (scaling and basis expansion)
        2. Extracts response variables
        3. Fits individual regression models for each response variable

        Parameters
        ----------
        data : NormData
            Training data containing covariates (X) and response variables (y).
            Must be a valid NormData object with properly formatted dimensions:
            - X: (n_samples, n_covariates)
            - y: (n_samples, n_response_vars)

        Notes
        -----
        - The method fits one regression model per response variable
        - Each model is stored in self.regression_models with response variable name as key
        - Preprocessing includes scaling and basis expansion based on norm_conf settings

        """
        self.register_data_info(data)
        self.preprocess(data)
        Output.print(Messages.FITTING_MODELS, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            resp_fit_data = data.sel({"response_vars": responsevar})
            self.focus(responsevar)
            Output.print(Messages.FITTING_MODEL, model_name=responsevar)
            self._fit(resp_fit_data)
            self.reset()
        self.is_fitted = True
        if self.norm_conf.savemodel:
            self.save()

    def predict(self, data: NormData) -> NormData:
        """
        Makes predictions for each response variable using fitted regression models.

        This method performs the following steps:
        1. Preprocesses the input data
        2. Generates predictions for each response variable
        3. Evaluates prediction performance
        4. Postprocesses the predictions

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions.
            Must have the same covariate structure as training data.

        Returns
        -------
        NormData
            Prediction results containing:
            - yhat: predicted values
            - ys2: prediction variances (if applicable)
            - Additional metrics (z-scores, centiles, etc.)

        Notes
        -----
        - Requires models to be previously fitted using fit()
        - Predictions are made independently for each response variable
        - Automatically computes evaluation metrics after prediction
        - Predictions are stored in the data object

        """
        assert self.is_fitted, "Model is not fitted!"
        self.preprocess(data)
        assert self.check_compatibility(data), "Data is not compatible with the model!"

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)

        Output.print(Messages.PREDICTING_MODELS, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            self.focus(responsevar)
            Output.print(Messages.PREDICTING_MODEL, model_name=responsevar)
            self._predict(resp_predict_data)
            self.reset()
        self.evaluate(data)
        if self.norm_conf.saveresults:
            self.save_results(data)
        return data

    def fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Combines model fitting and prediction in a single operation.

        This method provides a convenient way to fit models and generate predictions
        in one step, which can be more efficient than separate fit() and predict()
        calls.

        Parameters
        ----------
        fit_data : NormData
            Training data containing covariates and response variables
            for model fitting.

        predict_data : NormData
            Test data containing covariates for prediction. Must have the same
            covariate structure as fit_data with dimensions:
            - X: (n_test_samples, n_covariates)

        Returns
        -------
        NormData
            Prediction results containing:
            - yhat: predicted values (n_test_samples, n_response_vars)
            - ys2: prediction variances (n_test_samples, n_response_vars)
            - zscores: standardized residuals
            - centiles: prediction percentiles
            - Additional evaluation metrics

        Notes
        -----
        - Performs compatibility check between fit_data and predict_data
        - Automatically handles preprocessing and postprocessing
        - Computes evaluation metrics after prediction
        """

        assert predict_data.check_compatibility(fit_data), "Fit data and predict data are not compatible!"

        self.register_data_info(fit_data)
        self.preprocess(fit_data)
        self.preprocess(predict_data)
        Output.print(Messages.FITTING_AND_PREDICTING_MODELS, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            resp_fit_data = fit_data.sel({"response_vars": responsevar})
            resp_predict_data = predict_data.sel({"response_vars": responsevar})
            if responsevar not in self.regression_models:
                self.regression_models[responsevar] = self.regression_model_type(responsevar, self.default_reg_conf)
            self.focus(responsevar)
            Output.print(Messages.FITTING_AND_PREDICTING_MODEL, model_name=responsevar)
            self._fit_predict(resp_fit_data, resp_predict_data)
            self.reset()
        self.is_fitted = True
        if self.norm_conf.savemodel:
            self.save()
        self.evaluate(predict_data)
        if self.norm_conf.saveresults:
            self.save_results(predict_data)
        return predict_data

    def transfer(self, data: NormData, *args: Any, **kwargs: Any) -> "NormBase":
        """
        Transfers the normative model to new data, creating a new adapted model.

        This method performs transfer learning by adapting the existing model to new data
        while preserving knowledge from the original training. It creates a new normative
        model instance with transferred parameters for each response variable.

        Parameters
        ----------
        data : NormData
            Transfer data containing covariates (X) and response variables (y).
            Must have compatible structure with the original training data.

        Returns
        -------
        NormBase
            A new normative model instance adapted to the transfer data, containing:
            - Transferred regression models for each response variable
            - Preserved preprocessing transformations from original model

        Notes
        -----
        The transfer process:
        1. Preprocesses transfer data using original scalers
        2. Transfers each response variable's model separately
        3. Creates new model instance with transferred parameters
        4. Maintains original model's configuration with transfer-specific adjustments
        """
        self.preprocess(data)
        transfered_norm_conf_dict = copy.deepcopy(self.norm_conf.to_dict())
        transfered_norm_conf_dict["save_dir"] = kwargs.get("save_dir", self.norm_conf.save_dir + "_transfer")
        transfered_norm_conf = NormConf.from_dict(transfered_norm_conf_dict)
        # pylint: disable=too-many-function-args
        transfered_normative_model = self.__class__(
            transfered_norm_conf,
            self.default_reg_conf,  # type: ignore
        )
        transfered_normative_model.register_data_info(data)
        transfered_models = {}

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        Output.print(Messages.TRANSFERRING_MODELS, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_transfer_data = data.sel({"response_vars": responsevar})
            self.focus(responsevar)
            Output.print(Messages.TRANSFERRING_MODEL, model_name=responsevar)
            transfered_models[responsevar] = self._transfer(transfered_normative_model, resp_transfer_data, *args, **kwargs)
            self.reset()
        transfered_normative_model.regression_models = transfered_models
        self.transfer_basis_function(transfered_normative_model)
        self.transfer_scalers(transfered_normative_model)
        transfered_normative_model.batch_effects_counts = data.batch_effects_counts
        transfered_normative_model.is_fitted = True
        if transfered_normative_model.norm_conf.savemodel:
            transfered_normative_model.save()
        return transfered_normative_model

    def transfer_predict(self, fit_data: NormData, predict_data: NormData, *args: Any, **kwargs: Any) -> NormBase:
        """Transfer the normative model to new data and make predictions.

        Parameters
        ----------
        fit_data : NormData
            Data to fit the model to.
        predict_data : NormData
            Data to make predictions for.

        Returns
        -------
        NormBase
            The transferred model.
        """
        assert fit_data.check_compatibility(predict_data), "Fit data and predict data are not compatible!"
        transfered_model = self.transfer(fit_data, *args, **kwargs)
        transfered_model.predict(predict_data)
        return transfered_model

    def transfer_basis_function(self, transfered_normative_model: "NormBase") -> None:
        """
        Transfers the basis expansion from the original model to the transferred model.
        """
        transfered_normative_model.basis_function = copy.deepcopy(self.basis_function)
        transfered_normative_model.basis_function.compute_max = False
        transfered_normative_model.basis_function.compute_min = False

    def transfer_scalers(self, transfered_normative_model: "NormBase") -> None:
        """
        Transfers the scalers from the original model to the transferred model.
        """
        transfered_normative_model.inscalers = copy.deepcopy(self.inscalers)
        transfered_normative_model.outscalers = copy.deepcopy(self.outscalers)

    def extend(self, data: NormData, *args, **kwargs) -> "NormBase":
        """Extend the normative model with new data.

        Parameters
        ----------
        data : NormData
            Data to extend the model with.

        Returns
        -------
        NormBase
            The extended normative model.
        """

        synthetic_data = self._generate_synthetic_data(data)
        self.postprocess(synthetic_data)
        merged = synthetic_data.merge(data)
        reg_conf_copy = copy.deepcopy(self.default_reg_conf)
        norm_conf_copy = copy.deepcopy(self._norm_conf)
        norm_conf_copy.set_save_dir(kwargs.get("save_dir", self.norm_conf.save_dir + "_extend"))
        extended_model = self.__class__(norm_conf_copy, reg_conf_copy)  # type: ignore
        extended_model.fit(merged)
        return extended_model

    def extend_predict(self, fit_data: NormData, predict_data: NormData, *args: Any, **kwargs: Any) -> NormBase:
        """Extend the normative model with new data and make predictions.

        Parameters
        ----------
        fit_data : NormData
            Data to extend the model with.
        predict_data : NormData
            Data to make predictions for.

        Returns
        -------
        NormBase
            The extended normative model.
        """
        synthetic_data = self._generate_synthetic_data(fit_data)
        self.postprocess(synthetic_data)
        merged = synthetic_data.merge(fit_data)
        reg_conf_copy = copy.deepcopy(self.default_reg_conf)
        norm_conf_copy = copy.deepcopy(self._norm_conf)
        norm_conf_copy.set_save_dir(kwargs.get("save_dir", self.norm_conf.save_dir + "_extend"))
        extended_model = self.__class__(norm_conf_copy, reg_conf_copy)  # type: ignore
        extended_model.fit_predict(merged, predict_data)
        return extended_model

    def evaluate(self, data: NormData) -> None:
        """
        Evaluates model performance by computing z-scores, centiles, and additional evaluation metrics.

        This method performs a comprehensive evaluation of the normative model by:
        1. Computing standardized residuals (z-scores)
        2. Computing prediction centiles
        3. Calculating various evaluation metrics through the Evaluator class

        Parameters
        ----------
        data : NormData
            Data object containing prediction results to evaluate, including:
            - yhat: predicted values
            - ys2: prediction variances
            - y: actual values (if available for evaluation)

        Returns
        -------
        None
            Modifies the input data object in-place by adding evaluation metrics:
            - zscores: standardized residuals
            - centiles: prediction percentiles
            - Additional metrics from Evaluator (e.g., MSE, R², etc.)

        Notes
        -----
        The evaluation process includes:
        - Z-score computation to identify outliers
        - Centile computation for distributional analysis
        - Performance metrics calculation through Evaluator
        """
        data = self.compute_logp(data)
        data = self.compute_zscores(data)
        data = self.compute_centiles(data)
        data = self.evaluator.evaluate(data)

    def compute_centiles(
        self,
        data: NormData,
        cdf: Optional[List | np.ndarray] = None,
        **kwargs: Any,
    ) -> NormData:
        """
        Computes prediction centiles for each response variable in the data.

        Parameters
        ----------
        data : NormData
            Input data containing predictions for which to compute centiles.
            Must contain:
            - X: covariates array

        cdf : array-like, optional
            Cumulative distribution function values at which to compute centiles.
            Default values are [0.05, 0.25, 0.5, 0.75, 0.95], corresponding to:
            - 5th percentile
            - 25th percentile (Q1)
            - 50th percentile (median)
            - 75th percentile (Q3)
            - 95th percentile

        **kwargs : Any
            Additional keyword arguments passed to the underlying _centiles implementation.
            Model-specific parameters that affect centile computation.

        Returns
        -------
        NormData
            Input data extended with centile computations, adding:
            - scaled_centiles: centiles in scaled space
            - centiles: centiles in original space
            Both arrays have dimensions (cdf, datapoints, response_vars)
        """

        self.preprocess(data)

        if cdf is None:
            cdf = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        if isinstance(cdf, list):
            cdf = np.array(cdf)

        # Drop the centiles and dimensions if they already exist
        centiles_already_computed = "scaled_centiles" in data or "centiles" in data or "cdf" in data.coords
        if centiles_already_computed:
            data = data.drop_vars(["scaled_centiles", "centiles"])
            data = data.drop_dims(["cdf"])

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["scaled_centiles"] = xr.DataArray(
            np.zeros((cdf.shape[0], data.X.shape[0], len(respvar_intersection))),
            dims=("cdf", "datapoints", "response_vars"),
            coords={"cdf": cdf},
        )
        Output.print(Messages.COMPUTING_CENTILES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            self.focus(responsevar)
            Output.print(Messages.COMPUTING_CENTILES_MODEL, model_name=responsevar)
            data["scaled_centiles"].loc[{"response_vars": responsevar}] = self._centiles(resp_predict_data, cdf, **kwargs)
            self.reset()
        self.postprocess(data)
        return data

    def compute_zscores(self, data: NormData) -> NormData:
        """
        Computes standardized z-scores for each response variable in the data.

        Z-scores represent the number of standard deviations an observation is from the model's
        predicted mean. The specific computation depends on the underlying regression model.

        Parameters
        ----------
        data : NormData
            Input data containing:
            - X: covariates array (n_samples, n_features)
            - y: observed responses (n_samples, n_response_vars)
            - yhat: predicted means (n_samples, n_response_vars)
            - ys2: predicted variances (n_samples, n_response_vars)

        Returns
        -------
        NormData
            Input data extended with:
            - zscores: array of z-scores (n_samples, n_response_vars)
            Original data structure is preserved with additional z-score information.

        """

        self.preprocess(data)
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)

        data["zscores"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("datapoints", "response_vars"),
            coords={
                "datapoints": data.datapoints,
                "response_vars": list(respvar_intersection),
            },
        )
        Output.print(Messages.COMPUTING_ZSCORES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            self.focus(responsevar)
            Output.print(Messages.COMPUTING_ZSCORES_MODEL, model_name=responsevar)
            data["zscores"].loc[{"response_vars": responsevar}] = self._zscores(resp_predict_data)
            self.reset()
        self.postprocess(data)
        return data

    def compute_logp(self, data: NormData) -> NormData:
        """
        Computes standardized log-probabilities for each response variable in the data.

        Log-probabilities represent the log-probability of each observation under the model's
        predicted distribution. The specific computation depends on the underlying regression model.

        Parameters
        ----------
        data : NormData
            Input data containing:
            - X: covariates array (n_samples, n_features)
            - y: observed responses (n_samples, n_response_vars)
            - yhat: predicted means (n_samples, n_response_vars)
            - ys2: predicted variances (n_samples, n_response_vars)

        Returns
        -------
        NormData
            Input data extended with:
            - logp: array of log-probabilities (n_samples, n_response_vars)
            Original data structure is preserved with additional log-probability information.

        """

        self.preprocess(data)
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["logp"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("datapoints", "response_vars"),
            coords={
                "datapoints": data.datapoints,
                "response_vars": list(respvar_intersection),
            },
        )
        Output.print(Messages.COMPUTING_LOGP, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            self.focus(responsevar)
            Output.print(Messages.COMPUTING_LOGP_MODEL, model_name=responsevar)
            data["logp"].loc[{"response_vars": responsevar}] = self._logp(resp_predict_data)
            self.reset()
        return data

    def preprocess(self, data: NormData) -> None:
        """
        Applies preprocessing transformations to the input data.

        This method performs two main preprocessing steps:
        1. Data scaling using configured scalers
        2. Basis expansion of covariates (if specified)

        Parameters
        ----------
        data : NormData
            Data to preprocess containing:
            - X: covariates array
            - y: response variables array (optional)
            Must be a valid NormData object with proper dimensions.

        Returns
        -------
        None
            Modifies the input data object in-place by adding:
            - scaled_X: scaled covariates
            - scaled_y: scaled responses (if y exists)
            - expanded_X: basis-expanded covariates (if basis_function specified)

        Notes
        -----
        Scaling operations:
        - Creates and fits scalers if they don't exist
        - Applies existing scalers if already created
        - Supports different scaling methods via norm_conf.inscaler/outscaler

        Basis expansion options:
        - B-spline expansion: Creates basis using specified knots and order
        - Other basis functions as specified in norm_conf.basis_function
        """
        self.scale_forward(data)
        self.expand_basis(data, "scaled_X")

    def postprocess(self, data: NormData) -> None:
        """Apply postprocessing to the data.

        Args:
            data (NormData): Data to postprocess.
        """
        self.scale_backward(data)

    def check_compatibility(self, data: NormData) -> bool:
        """
        Check if the data is compatible with the model.

        Parameters
        ----------
        data : NormData
            Data to check compatibility with.

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        missing_covariates = [i for i in self.inscalers.keys() if i not in data.covariates.values]
        if len(missing_covariates) > 0:
            Output.warning(
                Warnings.MISSING_COVARIATES,
                covariates=missing_covariates,
                dataset_name=data.name,
            )

        extra_covariates = [i for i in data.covariates.values if i not in self.inscalers.keys()]
        if len(extra_covariates) > 0:
            Output.warning(
                Warnings.EXTRA_COVARIATES,
                covariates=extra_covariates,
                dataset_name=data.name,
            )

        extra_response_vars = [i for i in data.response_vars.values if i not in self.outscalers.keys()]
        if len(extra_response_vars) > 0:
            Output.warning(
                Warnings.EXTRA_RESPONSE_VARS,
                response_vars=extra_response_vars,
                dataset_name=data.name,
            )

        compatible = True
        unknown_batch_effects = {
            be: [u for u in unique if u not in self.unique_batch_effects[be]] for be, unique in data.unique_batch_effects.items()
        }
        compatible = sum([len(unknown) for unknown in unknown_batch_effects.values()]) == 0
        if len(unknown_batch_effects) > 0:
            Output.warning(
                Warnings.UNKNOWN_BATCH_EFFECTS,
                batch_effects=unknown_batch_effects,
                dataset_name=data.name,
            )
        return (
            (len(missing_covariates) == 0) and (len(extra_covariates) == 0) and (len(extra_response_vars) == 0) and (compatible)
        )

    def register_data_info(self, data: NormData) -> None:
        self.covariates = data.covariates.to_numpy().copy().tolist()
        self.response_vars = data.response_vars.to_numpy().copy().tolist()
        self.register_batch_effects(data)

    def register_batch_effects(self, data: NormData) -> None:
        self.unique_batch_effects = copy.deepcopy(data.unique_batch_effects)
        self.batch_effects_maps = {
            be: {k: i for i, k in enumerate(self.unique_batch_effects[be])} for be in self.unique_batch_effects.keys()
        }
        self.batch_effects_counts = data.batch_effects_counts

    def map_batch_effects(self, data: NormData) -> np.ndarray:
        mapped_batch_effects = np.zeros(data.batch_effects.values.shape)
        for i, be in enumerate(self.unique_batch_effects.keys()):
            for j, v in enumerate(data.batch_effects.values[:, i]):
                mapped_batch_effects[j, i] = self.batch_effects_maps[be][v]
        return mapped_batch_effects.astype(int)

    def sample_batch_effects(self, n_samples: int) -> pd.DataFrame:
        """
        Sample the batch effects from the estimated distribution.
        """
        max_batch_effect_count = max([len(v) for v in self.unique_batch_effects.values()])
        if n_samples < max_batch_effect_count:
            raise Output.error(
                Errors.SAMPLE_BATCH_EFFECTS,
                n_samples=n_samples,
                max_batch_effect_count=max_batch_effect_count,
            )

        bes = pd.DataFrame()
        for be in self.batch_effects_counts.keys():
            countsum = np.sum(list(self.batch_effects_counts[be].values()))
            bes[be] = np.random.choice(
                list(self.batch_effects_counts[be].keys()),
                size=n_samples,
                p=[c / countsum for c in list(self.batch_effects_counts[be].values())],
            )
            levels = self.unique_batch_effects[be]
            bes.loc[0 : len(levels) - 1, be] = levels
        return bes

    def scale_forward(self, data: NormData, overwrite: bool = False) -> None:
        """
        Scales input data to standardized form using configured scalers.

        This method handles the forward scaling transformation of both covariates (X)
        and response variables (y) using separate scalers for each variable. It creates
        and fits new scalers if they don't exist or if overwrite is True.

        Parameters
        ----------
        data : NormData
            Data object containing arrays to be scaled:
            - X : array-like, shape (n_samples, n_covariates)
                Covariate data to be scaled
            - y : array-like, shape (n_samples, n_response_vars), optional
                Response variable data to be scaled

        overwrite : bool, default=False
            If True, creates new scalers even if they already exist.
            If False, uses existing scalers when available.
        """
        for covariate in data.covariates.to_numpy():
            if (covariate not in self.inscalers) or overwrite:
                self.inscalers[covariate] = Scaler.from_string(self.norm_conf.inscaler)
                self.inscalers[covariate].fit(data.X.sel(covariates=covariate).data)

        for responsevar in data.response_vars.to_numpy():
            if (responsevar not in self.outscalers) or overwrite:
                self.outscalers[responsevar] = Scaler.from_string(self.norm_conf.outscaler)
                self.outscalers[responsevar].fit(data.y.sel(response_vars=responsevar).data)

        data.scale_forward(self.inscalers, self.outscalers)

    def scale_backward(self, data: NormData) -> None:
        """
        Scales data back to its original scale using stored scalers.

        This method performs inverse scaling transformation on the data using previously
        fitted scalers. It reverses the scaling applied during preprocessing to return
        predictions and other computed values to their original scale.

        Parameters
        ----------
        data : NormData
            Data object containing scaled values to be transformed back. May include:
            - scaled_X: scaled covariates
            - scaled_y: scaled responses
            - scaled_yhat: scaled predictions
            - scaled_ys2: scaled prediction variances
            - scaled_centiles: scaled prediction centiles
        """
        data.scale_backward(self.inscalers, self.outscalers)

    def expand_basis(self, data: NormData, source_array: str):
        """Expand the basis of a source array using a specified basis function.

        Parameters
        ----------
        source_array_name : str
            The name of the source array to expand ('X' or 'scaled_X')
        basis_function : str
            The basis function to use ('polynomial', 'bspline', 'linear', or 'none')
        basis_column : int, optional
            The column index to apply the basis function, by default 0
        linear_component : bool, optional
            Whether to include a linear component, by default True
        **kwargs : dict
            Additional arguments for basis functions

        Raises
        ------
        ValueError
            If the source array does not exist or if required parameters are missing
        """
        if not hasattr(self, "basis_function"):
            self.basis_function = create_basis_function(
                self.norm_conf.basis_function,
                source_array,
                **self.norm_conf.basis_function_kwargs,
            )
        if not self.basis_function.is_fitted:
            self.basis_function.fit(data)
        self.basis_function.transform(data)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model to a file.

        Args:
            path (str, optional): The path to save the model to. If None, the model is saved to the save_dir provided in the norm_conf.
        """
        Output.print(Messages.SAVING_MODEL, save_dir=self.norm_conf.save_dir)
        if path is not None:
            self.norm_conf.set_save_dir(path)
        metadata = {
            "norm_conf": self.norm_conf.to_dict(),
            "regression_model_type": self.regression_model_type.__name__,
            "default_reg_conf": self.default_reg_conf.to_dict(),
            "inscalers": {k: v.to_dict() for k, v in self.inscalers.items()},
            "unique_batch_effects": self.unique_batch_effects,
            "is_fitted": self.is_fitted,
        }

        if hasattr(self, "covariates"):
            metadata["covariates"] = self.covariates

        if hasattr(self, "bspline_basis"):
            metadata["basis_function"] = copy.deepcopy(self.basis_function)

        # JSON keys are always string, so we have use a trick to save the batch effects distributions, which may have string or int keys.
        # We invert the map, so that the original keys are stored as the values
        # The original integer values are then converted to strings by json, but we can safely convert them back to ints when loading
        # We also add an index to the keys to make sure they are unique
        if hasattr(self, "batch_effects_counts"):
            metadata["inverse_batch_effects_counts"] = {
                be: {f"{i}_{k}": v for i, (v, k) in enumerate(self.batch_effects_counts[be].items())}
                for be, mp in self.batch_effects_counts.items()
            }
        if hasattr(self, "batch_effects_maps"):
            metadata["batch_effects_maps"] = {
                be: {f"{i}_{k}": v for i, (v, k) in enumerate(self.batch_effects_maps[be].items())}
                for be in self.batch_effects_maps.keys()
            }

        model_save_path = os.path.join(self.norm_conf.save_dir, "model")
        os.makedirs(model_save_path, exist_ok=True)
        with open(
            os.path.join(model_save_path, "normative_model.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=4)

        for responsevar, model in self.regression_models.items():
            reg_model_dict = {}
            reg_model_save_path = os.path.join(model_save_path, f"{responsevar}")
            os.makedirs(reg_model_save_path, exist_ok=True)
            reg_model_dict["model"] = model.to_dict(reg_model_save_path)
            reg_model_dict["outscaler"] = self.outscalers[responsevar].to_dict()
            json_save_path = os.path.join(reg_model_save_path, "regression_model.json")
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(reg_model_dict, f, indent=4)

    @classmethod
    def load(cls, path: str) -> NormBase:
        model_path = os.path.join(path, "model", "normative_model.json")
        with open(model_path, mode="r", encoding="utf-8") as f:
            metadata = json.load(f)

        self = cls(NormConf.from_dict(metadata["norm_conf"]))

        if "basis_function" in metadata:
            self.basis_function = create_basis_function(metadata["basis_function"])
        self.inscalers = {k: Scaler.from_dict(v) for k, v in metadata["inscalers"].items()}
        if "batch_effects_maps" in metadata:
            self.batch_effects_maps = {
                be: {v: int(k.split("_")[1]) for k, v in mp.items()} for be, mp in metadata["batch_effects_maps"].items()
            }
        if "inverse_batch_effects_counts" in metadata:
            self.batch_effects_counts = {
                be: {v: int(k.split("_")[1]) for k, v in mp.items()}
                for be, mp in metadata["inverse_batch_effects_counts"].items()
            }
        if "unique_batch_effects" in metadata:
            self.unique_batch_effects = metadata["unique_batch_effects"]

        if "covariates" in metadata:
            self.covariates = metadata["covariates"]

        self.regression_model_type = globals()[metadata["regression_model_type"]]
        self.response_vars = []
        self.outscalers = {}
        self.regression_models = {}
        self.is_fitted = metadata["is_fitted"]
        reg_models_path = os.path.join(path, "model", "*")
        for path in glob.glob(reg_models_path):
            if os.path.isdir(path):
                with open(
                    os.path.join(path, "regression_model.json"),
                    mode="r",
                    encoding="utf-8",
                ) as f:
                    reg_model_dict = json.load(f)
                    responsevar = reg_model_dict["model"]["name"]
                    self.response_vars.append(responsevar)
                    self.regression_models[responsevar] = self.regression_model_type.from_dict(reg_model_dict["model"], path)
                    self.outscalers[responsevar] = Scaler.from_dict(reg_model_dict["outscaler"])
        self.default_reg_conf = type(self.regression_models[self.response_vars[0]].reg_conf).from_dict(
            metadata["default_reg_conf"]
        )

        return self

    def save_results(self, data: NormData) -> None:
        Output.print(Messages.SAVING_RESULTS, save_dir=self.norm_conf.save_dir)
        os.makedirs(os.path.join(self.norm_conf.save_dir, "results"), exist_ok=True)
        self.save_zscores(data)
        self.save_centiles(data)
        self.save_measures(data)
        os.makedirs(os.path.join(self.norm_conf.save_dir, "plots"), exist_ok=True)
        plot_centiles(
            self,
            data,
            save_dir=os.path.join(self.norm_conf.save_dir, "plots"),
            show_data=True,
        )
        plot_qq(data, save_dir=os.path.join(self.norm_conf.save_dir, "plots"))

    def save_zscores(self, data: NormData) -> None:
        zdf = data.zscores.to_dataframe().unstack(level="response_vars")
        zdf.columns = zdf.columns.droplevel(0)
        res_path = os.path.join(self.norm_conf.save_dir, "results", "zscores.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=0) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    # Merge on datapoints, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(zdf, on="datapoints", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = zdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save_centiles(self, data: NormData) -> None:
        cdf = data.centiles.to_dataframe().unstack(level="response_vars")
        cdf.columns = cdf.columns.droplevel(0)
        res_path = os.path.join(self.norm_conf.save_dir, "results", "centiles.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=[0, 1]) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    # Merge on datapoints, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(cdf, on=["datapoints", "cdf"], how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = cdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save_measures(self, data: NormData) -> None:
        mdf = data.measures.to_dataframe().unstack(level="response_vars")
        mdf.columns = mdf.columns.droplevel(0)
        res_path = os.path.join(self.norm_conf.save_dir, "results", "measures.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=0) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    # Merge on datapoints, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(mdf, on="measure", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = mdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def focus(self, responsevar: str) -> None:
        """
        Prepares the model for operations on a specific response variable by setting up
        the corresponding regression model.

        This method serves as an internal state manager that:
        1. Sets the current focused response variable
        2. Creates a new regression model if one doesn't exist for this variable
        3. Makes the focused model easily accessible for subsequent operations

        Parameters
        ----------
        responsevar : str
            The name of the response variable to focus on. Must be one of the variables
            present in the training data.
        """
        self.focused_var = responsevar
        if responsevar not in self.regression_models:
            self.regression_models[responsevar] = self.regression_model_type(responsevar, self.get_reg_conf(responsevar))

    def get_reg_conf(self, responsevar: str) -> RegConf:
        """
        Get regression configuration for a specific response variable.

        This method retrieves the regression configuration for a given response variable,
        either returning an existing configuration if the model exists, or the default
        configuration if it doesn't.

        Parameters
        ----------
        responsevar : str
            Name of the response variable to get configuration for.

        Returns
        -------
        RegConf
            Regression configuration object for the specified response variable.

        Notes
        -----
        The method implements a simple lookup strategy:
        1. Check if model exists for response variable
        2. If yes, return its configuration
        3. If no, return default configuration
        """
        if responsevar in self.regression_models:
            return self.regression_models[responsevar].reg_conf
        else:
            return self.default_reg_conf

    def set_save_dir(self, save_dir: str) -> None:
        """Override the save_dir in the norm_conf.

        Args:
            save_dir (str): New save directory.
        """
        self.norm_conf.set_save_dir(save_dir)

    def reset(self) -> None:
        """Does nothing. Can be overridden by subclasses."""

    def __getitem__(self, key: str) -> RegressionModel:
        return self.regression_models[key]

    #######################################################################################################

    # Abstract methods

    #######################################################################################################

    @abstractmethod
    def _fit(self, data: NormData, make_new_model: bool = False) -> None:
        """
        Fits the specific regression model to the provided data.

        Parameters
        ----------
        data : NormData
            The data to fit the model to, containing covariates and response variables.
        make_new_model : bool, optional
            If True, creates a new model instance for fitting. Default is False.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model = ConcreteNormModel(norm_conf)
        >>> model._fit(training_data)
        """

    @abstractmethod
    def _predict(self, data: NormData) -> None:
        """
        Predicts response variables using the fitted regression model.

        Parameters
        ----------
        data : NormData
            The data for which to make predictions, containing covariates.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model = ConcreteNormModel(norm_conf)
        >>> model._predict(test_data)
        """

    @abstractmethod
    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> None:
        """
        Fits the model to the fit_data and predicts on the predict_data.

        Parameters
        ----------
        fit_data : NormData
            The data to fit the model to, containing covariates and response variables.
        predict_data : NormData
            The data for which to make predictions, containing covariates.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model = ConcreteNormModel(norm_conf)
        >>> model._fit_predict(training_data, test_data)
        """

    @abstractmethod
    def _transfer(self, model: NormBase, data: NormData, **kwargs: Any) -> RegressionModel:
        """
        Transfers the current regression model to new data.

        Parameters
        ----------
        data : NormData
            The data to transfer the model to, containing covariates and response variables.
        **kwargs : Any
            Additional keyword arguments for the transfer process.

        Returns
        -------
        RegressionModel
            A new regression model adapted to the new data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model = ConcreteNormModel(norm_conf)
        >>> new_model = model._transfer(transfer_data)
        """

    @abstractmethod
    def _extend(self, data: NormData) -> NormBase:
        """
        Extends the current regression model with new data.

        Parameters
        ----------
        data : NormData
            The data to extend the model with, containing covariates and response variables.

        Returns
        -------
        NormBase
            The extended normative model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model = ConcreteNormModel(norm_conf)
        >>> extended_model = model._extend(additional_data)
        """

    @abstractmethod
    def _generate_synthetic_data(self, data: NormData, n_synthetic_samples: int = 1000) -> NormData:
        """
        Generates synthetic data from the model. Required for model extension.

        Parameters
        ----------
        data : NormData
            The data to generate synthetic data for. Must contain covariates and empty response variables.

        Returns
        -------
        NormData
            The data with the predicted response variables and batch effects.
        """

    @abstractmethod
    def _centiles(self, data: NormData, cdf: np.ndarray, **kwargs: Any) -> xr.DataArray:
        """Computes centiles of the model predictions for given cumulative density values.

        Parameters
        ----------
        data : NormData
            Data object containing model predictions to compute centiles for
        cdf : np.ndarray
            Array of cumulative density values to compute centiles at (between 0 and 1)
        **kwargs : Any
            Additional keyword arguments passed to the centile computation

        Returns
        -------
        xr.DataArray
            DataArray containing the computed centiles with dimensions (cdf, datapoints)
            where:
            - cdf dimension corresponds to the input cumulative density values
            - datapoints dimension corresponds to the samples in the input data

        Notes
        -----
        Centiles represent the values below which a given percentage of observations fall.
        For example, the 50th centile is the median value.

        The centile computation depends on the specific regression model implementation
        and its underlying distributional assumptions.

        Examples
        --------
        >>> # Compute 25th, 50th and 75th centiles
        >>> centiles = model._centiles(data, np.array([0.25, 0.5, 0.75]))
        >>> median = centiles.sel(cdf=0.5)
        """

    @abstractmethod
    def _zscores(self, data: NormData) -> xr.DataArray:
        """Computes standardized residuals (z-scores) for model predictions.

        Parameters
        ----------
        data : NormData
            Data object containing model predictions to compute z-scores for,
            including predicted values (yhat), prediction variances (ys2),
            and actual values (y)

        Returns
        -------
        xr.DataArray
            DataArray containing the computed z-scores with dimensions (datapoints)
            where datapoints corresponds to the samples in the input data

        Notes
        -----
        Z-scores measure how many standard deviations an observation is from the mean
        of the predicted distribution. They are useful for:
        - Identifying outliers
        - Assessing prediction accuracy
        - Standardizing residuals across different scales

        The computation depends on the specific regression model implementation
        and its underlying distributional assumptions.

        Examples
        --------
        >>> # Compute z-scores for predictions
        >>> zscores = model._zscores(predictions)
        >>> outliers = abs(zscores) > 2  # Find outliers beyond 2 SD
        """

    @abstractmethod
    def _logp(self, data: NormData) -> None:
        """
        Computes log-probabilities for each response variable in the data.
        """

    @abstractmethod
    def n_params(self) -> int:
        """
        Returns the number of parameters of the model.

        Returns
        -------
        int
            The number of parameters in the model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> num_params = model.n_params()
        """

    @classmethod
    @abstractmethod
    def from_args(cls, args: dict) -> NormBase:
        """
        Creates a normative model from command line arguments.

        Parameters
        ----------
        args : dict
            Dictionary of command line arguments.

        Returns
        -------
        NormBase
            An instance of the normative model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> args = {"param1": value1, "param2": value2}
        >>> model = ConcreteNormModel.from_args(args)
        """

    #######################################################################################################

    # Properties

    #######################################################################################################

    @property
    def norm_conf(self) -> NormConf:
        """Returns the norm_conf attribute.

        Returns:
            NormConf: The norm_conf attribute.
        """
        return self._norm_conf

    @property
    def has_random_effect(self) -> bool:
        """Returns whether the model has a random effect.

        Returns:
            bool: True if the model has a random effect, False otherwise.
        """
        return self.focused_model.has_random_effect

    @property
    @abstractmethod
    def focused_model(self) -> RegressionModel:
        """Returns the regression model that is currently focused on."""

    @focused_model.setter
    def focused_model(self, value: str) -> None:
        self.focused_var = value
