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
- Gaussian Process Regression (GPR)
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


Examples
--------
>>> from pcntoolkit.normative_model import NormBase
>>> from pcntoolkit.normative_model.norm_conf import NormConf
>>>
>>> # Create configuration for normative model
>>> norm_conf = NormConf(save_dir="./models", log_dir="./logs")

>>> # Create configuration for regression models
>>> hbr_conf = HBRConf()

>>> # Create normative model
>>> model = NormHBR(norm_conf, hbr_conf)

>>>
>>> # Fit model
>>> model.fit(training_data)
>>>
>>> # Make predictions
>>> predictions = model.predict(test_data)

See Also
--------
pcntoolkit.regression_model : Package containing various regression model implementations
pcntoolkit.dataio : Package for data input/output operations
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

    Methods
    -------
    fit(data: NormData) -> None
        Fit the normative model to training data.

    predict(data: NormData) -> NormData
        Make predictions using the fitted model.

    fit_predict(fit_data: NormData, predict_data: NormData) -> NormData
        Fit the model and make predictions in one step.

    transfer(data: NormData, *args: Any, **kwargs: Any) -> NormBase
        Transfer the model to new data.

    extend(data: NormData) -> None
        Extend the model with additional data.

    tune(data: NormData) -> None
        Tune model parameters using validation data.

    merge(other: NormBase) -> None
        Merge current model with another normative model.

    evaluate(data: NormData) -> None
        Evaluate model performance.

    compute_centiles(data: NormData, cdf: Optional[List | np.ndarray] = None) -> NormData
        Compute prediction centiles
    compute_zscores(data: NormData) -> NormData
        Compute z-scores for predictions.

    save(path: Optional[str] = None) -> None
        Save model to disk.

    load(path: str) -> NormBase
        Load model from disk.

    Notes
    -----
    The NormBase class implements the Template Method pattern, where the main workflow
    is defined in the base class, but specific implementations are delegated to
    subclasses through abstract methods.

    Examples
    --------
    Example of implementing a concrete normative model:

    >>> class ConcreteNormModel(NormBase):
    ...     def _fit(self, data):
    ...         # Implementation
    ...         pass
    ...
    ...     def _predict(self, data):
    ...         # Implementation
    ...         pass
    ...
    ...     # Implement other abstract methods...

    Example of using a normative model:

    >>> from pcntoolkit.dataio import NormData
    >>>
    >>> # Prepare data
    >>> train_data = NormData(X=train_covariates, y=train_responses)
    >>> test_data = NormData(X=test_covariates, y=test_responses)
    >>>
    >>> # Create and fit model
    >>> model = ConcreteNormModel(norm_conf, reg_conf)
    >>> model.fit(train_data)
    >>>
    >>> # Make predictions
    >>> predictions = model.predict(test_data)
    >>>
    >>> # Compute evaluation metrics
    >>> model.evaluate(predictions)
    """

    def __init__(self, norm_conf: NormConf):
        self._norm_conf: NormConf = norm_conf
        object.__setattr__(
            self._norm_conf, "normative_model_name", self.__class__.__name__
        )

        self.response_vars: list[str] = None  # type: ignore
        self.regression_model_type: Any = None  # type: ignore
        self.default_reg_conf: RegConf = None  # type: ignore
        self.regression_models: dict[str, RegressionModel] = {}
        self.focused_var: str = None  # type: ignore
        self.evaluator = Evaluator()
        self.inscalers: dict = {}
        self.outscalers: dict = {}
        self.basis_function: BasisFunction
        self.basis_column:Optional[int] = None

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

        Returns
        -------
        None
            The method modifies the model's internal state by fitting regression models.

        Notes
        -----
        - The method fits one regression model per response variable
        - Each model is stored in self.regression_models with response variable name as key
        - Preprocessing includes scaling and basis expansion based on norm_conf settings

        Examples
        --------
        >>> from pcntoolkit.dataio import NormData
        >>> model = NormBase(norm_conf)
        >>> train_data = NormData(X=covariates, y=responses)
        >>> model.fit(train_data)

        Raises
        ------
        ValueError
            If data is not properly formatted or contains invalid values
        RuntimeError
            If preprocessing or model fitting fails
        """
        self.preprocess(data)
        self.response_vars = data.response_vars.to_numpy().copy().tolist()
        print(f"{os.getpid()} - Going to fit {len(self.response_vars)} models\n")
        for responsevar in self.response_vars:
            resp_fit_data = data.sel(response_vars=responsevar)
            self.focus(responsevar)
            print(f"{os.getpid()} - Fitting model for {responsevar}\n")
            self._fit(resp_fit_data)
            self.reset()
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

        Examples
        --------
        >>> test_data = NormData(X=test_covariates)
        >>> predictions = model.predict(test_data)
        >>> print(predictions.yhat)  # access predictions
        >>> print(predictions.ys2)   # access variances

        Raises
        ------
        ValueError
            If model hasn't been fitted or data format is invalid
        RuntimeError
            If prediction process fails for any response variable
        """
        self.preprocess(data)
        print(f"Going to predict {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            resp_predict_data = data.sel(response_vars=responsevar)
            if responsevar not in self.regression_models:
                raise ValueError(
                    f"Attempted to predict model {responsevar}, but it does not exist."
                )
            self.focus(responsevar)
            print(f"Predicting model for {responsevar}")
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

        Examples
        --------
        >>> train_data = NormData(X=train_covariates, y=train_responses)
        >>> test_data = NormData(X=test_covariates)
        >>> results = model.fit_predict(train_data, test_data)
        >>>
        >>> # Access predictions
        >>> zscores = results.zscores
        >>> centiles = results.centiles
        >>> measures = results.measures

        Raises
        ------
        ValueError
            If data formats are incompatible or invalid
        AssertionError
            If fit_data and predict_data have incompatible structures
        RuntimeError
            If fitting or prediction process fails

        See Also
        --------
        fit : Method for model fitting only
        predict : Method for prediction using pre-fitted model
        compute_zscores : Method for computing standardized residuals
        compute_centiles : Method for computing prediction percentiles
        """

        assert fit_data.is_compatible_with(
            predict_data
        ), "Fit data and predict data are not compatible!"

        self.preprocess(fit_data)
        self.preprocess(predict_data)
        self.response_vars = fit_data.response_vars.to_numpy().copy().tolist()
        print(f"Going to fit and predict {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            resp_fit_data = fit_data.sel(response_vars=responsevar)
            resp_predict_data = predict_data.sel(response_vars=responsevar)
            if responsevar not in self.regression_models:
                self.regression_models[responsevar] = self.regression_model_type(
                    responsevar, self.default_reg_conf
                )
            self.focus(responsevar)
            print(f"Fitting and predicting model for {responsevar}")
            self._fit_predict(resp_fit_data, resp_predict_data)
            self.reset()
        if self.norm_conf.savemodel:
            self.save()
        # Get the results
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
        *args : Any
            Additional positional arguments passed to the underlying transfer implementation.
        **kwargs : Any
            Additional keyword arguments passed to the underlying transfer implementation.
            Common options include:
            - 'transfer_type': str, type of transfer learning to perform
            - 'learning_rate': float, adaptation rate for transfer
            - 'regularization': float, strength of regularization during transfer

        Returns
        -------
        NormBase
            A new normative model instance adapted to the transfer data, containing:
            - Transferred regression models for each response variable
            - Updated configuration with transfer-specific settings
            - Preserved preprocessing transformations from original model

        Notes
        -----
        The transfer process:
        1. Preprocesses transfer data using original scalers
        2. Transfers each response variable's model separately
        3. Creates new model instance with transferred parameters
        4. Maintains original model's configuration with transfer-specific adjustments

        The method supports different transfer learning approaches depending on the
        underlying regression model implementation.

        Examples
        --------
        >>> # Original model trained on source domain
        >>> original_model = NormBase(norm_conf)
        >>> original_model.fit(source_data)
        >>>
        >>> # Transfer to target domain
        >>> transfer_data = NormData(X=target_covariates, y=target_responses)
        >>> transferred_model = original_model.transfer(
        ...     transfer_data,
        ...     transfer_type='fine_tune',
        ...     learning_rate=0.01
        ... )
        >>>
        >>> # Make predictions with transferred model
        >>> predictions = transferred_model.predict(test_data)

        Raises
        ------
        ValueError
            If the model hasn't been fitted before transfer attempt
        AssertionError
            If transfer data is incompatible with original model structure
        RuntimeError
            If transfer process fails for any response variable

        See Also
        --------
        _transfer : Abstract method implementing specific transfer logic
        fit : Method for initial model fitting
        predict : Method for making predictions

        Notes
        -----
        Transfer learning considerations:
        - Ensures knowledge preservation from source domain
        - Adapts to target domain characteristics
        - Maintains model structure and constraints
        - Supports various transfer strategies

        The effectiveness of transfer depends on:
        - Similarity between source and target domains
        - Amount of transfer data available
        - Choice of transfer learning parameters
        - Underlying model architecture
        """
        self.preprocess(data)
        transfered_models = {}
        print(f"Going to transfer {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            resp_transfer_data = data.sel(response_vars=responsevar)
            if responsevar not in self.regression_models:
                raise ValueError(
                    "Attempted to transfer a model that has not been fitted."
                )
            self.focus(responsevar)
            print(f"Transferring model for {responsevar}")
            transfered_models[responsevar] = self._transfer(
                resp_transfer_data, *args, **kwargs
            )
            self.reset()

        transfered_norm_conf_dict = self.norm_conf.to_dict()
        transfered_norm_conf_dict["save_dir"] = self.norm_conf.save_dir + "_transfer"
        transfered_norm_conf = NormConf.from_dict(transfered_norm_conf_dict)

        # pylint: disable=too-many-function-args
        transfered_normative_model = self.__class__(
            transfered_norm_conf,
            self.default_reg_conf,  # type: ignore
        )
        transfered_normative_model.response_vars = (
            data.response_vars.to_numpy().copy().tolist()
        )
        transfered_normative_model.regression_models = transfered_models
        self.transfer_basis_function(transfered_normative_model)
        self.transfer_scalers(transfered_normative_model)
        if transfered_normative_model.norm_conf.savemodel:
            transfered_normative_model.save()
        return transfered_normative_model
    
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


    def extend(self, data: NormData) -> None:
        """Extends the normative model with new data.

        Args:
            data (NormData): Data containing the covariates and response variables to extend the model with.
        """
        self._extend(data)

    def tune(self, data: NormData) -> None:
        """Tunes the normative model with new data.

        Args:
            data (NormData): Data containing the covariates and response variables to tune the model with.
        """
        self._tune(data)

    def merge(self, other: "NormBase") -> None:
        """Merges the current normative model with another normative model.

        Args:
            other (NormBase): The other normative model to merge with.

        Raises:
            ValueError: Error if the models are not of the same type.
        """
        if not self.__class__ == other.__class__:
            raise ValueError("Attempted to merge normative models of different types.")
        self._merge(other)

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

        The method modifies the input data object by adding computed metrics
        as new variables/attributes.

        Examples
        --------
        >>> # After making predictions
        >>> predictions = model.predict(test_data)
        >>> model.evaluate(predictions)
        >>>
        >>> # Access evaluation results
        >>> zscores = predictions.zscores
        >>> centiles = predictions.centiles
        >>> metrics = predictions.metrics  # Additional evaluation metrics

        See Also
        --------
        compute_zscores : Method for computing standardized residuals
        compute_centiles : Method for computing prediction percentiles
        Evaluator : Class handling additional evaluation metrics

        Notes
        -----
        Evaluation metrics typically include:
        - Mean Squared Error (MSE)
        - R-squared (R²)
        - Mean Absolute Error (MAE)
        - Additional metrics defined in Evaluator

        The exact metrics computed depend on:
        - Availability of true values (y)
        - Model type and capabilities
        - Evaluator configuration

        Warnings
        --------
        - Ensure data contains necessary fields for evaluation
        - Some metrics may be unavailable without true values
        - Large datasets may require significant computation time
        """
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

        This method calculates percentile values for model predictions, providing a way to
        assess the distribution of predicted values and identify potential outliers.

        Parameters
        ----------
        data : NormData
            Input data containing predictions for which to compute centiles.
            Must contain:
            - X: covariates array
            - yhat: predicted values (if already predicted)
            - ys2: prediction variances (if applicable)

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

        Notes
        -----
        The computation process:
        1. Preprocesses input data (scaling)
        2. Computes centiles for each response variable
        3. Postprocesses results (inverse scaling)
        4. Handles existing centile computations by removing them first

        The method supports both parametric and non-parametric centile computations
        depending on the underlying model implementation.

        Examples
        --------
        >>> # Compute default centiles
        >>> results = model.compute_centiles(prediction_data)
        >>> centiles = results.centiles
        >>>
        >>> # Compute specific centiles
        >>> custom_centiles = model.compute_centiles(
        ...     prediction_data,
        ...     cdf=[0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
        ... )
        >>>
        >>> # Access specific centile (e.g., median)
        >>> median = results.centiles.sel(cdf=0.5)

        Raises
        ------
        ValueError
            If attempting to compute centiles for a model that hasn't been fitted
        AssertionError
            If input data format is invalid or incompatible

        See Also
        --------
        compute_zscores : Method for computing standardized scores
        _centiles : Abstract method implementing specific centile computation
        scale_forward : Method for data preprocessing
        scale_backward : Method for data postprocessing

        Notes
        -----
        Centile interpretation:
        - Values below 5th or above 95th percentile may indicate outliers
        - Median (50th) provides central tendency
        - Q1-Q3 range (25th-75th) shows typical variation

        Performance considerations:
        - Computation time scales with number of response variables
        - Memory usage depends on number of centile points requested
        - Consider using fewer centile points for large datasets
        """

        self.preprocess(data)

        if cdf is None:
            cdf = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        if isinstance(cdf, list):
            cdf = np.array(cdf)

        # Drop the centiles and dimensions if they already exist
        centiles_already_computed = (
            "scaled_centiles" in data or "centiles" in data or "cdf" in data.coords
        )
        if centiles_already_computed:
            data = data.drop_vars(["scaled_centiles", "centiles"])
            data = data.drop_dims(["cdf"])

        data["scaled_centiles"] = xr.DataArray(
            np.zeros((cdf.shape[0], data.X.shape[0], len(self.response_vars))),
            dims=("cdf", "datapoints", "response_vars"),
            coords={"cdf": cdf},
        )
        for responsevar in self.response_vars:
            resp_predict_data = data.sel(response_vars=responsevar)
            if responsevar not in self.regression_models:
                raise ValueError(
                    f"Attempted to find quantiles for model {responsevar}, but it does not exist."
                )
            self.focus(responsevar)
            print("Computing centiles for", responsevar)
            data["scaled_centiles"].loc[{"response_vars": responsevar}] = (
                self._centiles(resp_predict_data, cdf, **kwargs)
            )
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

        Notes
        -----
        The method:
        1. Preprocesses input data using model's scalers
        2. Computes z-scores independently for each response variable
        3. Uses model-specific z-score computation (_zscores abstract method)
        4. Postprocesses results back to original scale

        Z-scores can be interpreted as:
        - |z| < 1: observation within 1 SD of prediction
        - |z| < 2: observation within 2 SD of prediction
        - |z| > 3: potential outlier

        Examples
        --------
        >>> # Compute z-scores for test data
        >>> test_data = NormData(X=test_covariates, y=test_responses)
        >>> results = model.compute_zscores(test_data)
        >>>
        >>> # Access z-scores
        >>> zscores = results.zscores
        >>>
        >>> # Identify potential outliers
        >>> outliers = np.abs(zscores) > 3

        Raises
        ------
        ValueError
            If model hasn't been fitted for any response variable

        See Also
        --------
        _zscores : Abstract method implementing specific z-score computation
        compute_centiles : Method for computing prediction percentiles
        preprocess : Method for data preprocessing
        postprocess : Method for data postprocessing

        Notes
        -----
        Implementation considerations:
        - Handles missing data appropriately
        - Supports multiple response variables
        - Maintains data dimensionality
        - Computationally efficient for large datasets

        The z-scores are particularly useful for:
        - Identifying outliers
        - Assessing prediction accuracy
        - Comparing across different response variables
        - Quality control in clinical applications
        """

        self.preprocess(data)
        data["zscores"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(self.response_vars))),
            dims=("datapoints", "response_vars"),
            coords={"datapoints": data.datapoints, "response_vars": self.response_vars},
        )
        for responsevar in self.response_vars:
            resp_predict_data = data.sel(response_vars=responsevar)
            if responsevar not in self.regression_models:
                raise ValueError(
                    f"Attempted to find zscores for self {responsevar}, but it does not exist."
                )
            self.focus(responsevar)
            print("Computing zscores for", responsevar)
            data["zscores"].loc[{"response_vars": responsevar}] = self._zscores(
                resp_predict_data
            )
            self.reset()
        self.postprocess(data)
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

        Examples
        --------
        >>> data = NormData(X=covariates, y=responses)
        >>> model.preprocess(data)
        >>>
        >>> # Access preprocessed data
        >>> scaled_X = data.scaled_X
        >>> scaled_y = data.scaled_y
        >>> expanded_X = data.expanded_X  # if basis expansion enabled

        See Also
        --------
        scale_forward : Method handling data scaling
        scale_backward : Method for inverse scaling
        NormData.expand_basis : Method for covariate basis expansion

        Notes
        -----
        B-spline basis expansion:
        - Creates basis only once and reuses for subsequent calls
        - Basis is created using min/max of specified column
        - Supports linear component inclusion

        Other basis expansions:
        - Applied directly without storing basis
        - Linear component handling configurable

        Warnings
        --------
        - Ensure consistent preprocessing between training and test data
        - B-spline basis requires sufficient data range in basis column
        - Memory usage increases with basis expansion complexity
        """
        self.scale_forward(data)
        # TODO: pass kwargs from config to expand_basis
        self.expand_basis(data, "scaled_X")

    def postprocess(self, data: NormData) -> None:
        """Apply postprocessing to the data.

        Args:
            data (NormData): Data to postprocess.
        """
        self.scale_backward(data)

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

        Returns
        -------
        None
            Modifies the input data object in-place by adding:
            - scaled_X : scaled covariate data
            - scaled_y : scaled response data (if y exists)

        Notes
        -----
        Scaling operations:
        1. For each covariate:
            - Creates/retrieves scaler using norm_conf.inscaler type
            - Fits scaler if new or overwrite=True
            - Transforms data using scaler

        2. For each response variable:
            - Creates/retrieves scaler using norm_conf.outscaler type
            - Fits scaler if new or overwrite=True
            - Transforms data using scaler

        The scalers are stored in:
        - self.inscalers : dict mapping covariate names to their scalers
        - self.outscalers : dict mapping response variable names to their scalers

        Examples
        --------
        >>> # Basic usage
        >>> data = NormData(X=covariates, y=responses)
        >>> model.scale_forward(data)
        >>> scaled_X = data.scaled_X
        >>> scaled_y = data.scaled_y

        >>> # Force new scalers
        >>> model.scale_forward(data, overwrite=True)

        See Also
        --------
        scale_backward : Method for inverse scaling transformation
        NormData : Class containing data structures
        scaler : Factory function for creating scalers

        Notes
        -----
        Supported scaler types (configured in norm_conf):
        - 'StandardScaler': zero mean, unit variance
        - 'MinMaxScaler': scales to specified range
        - 'RobustScaler': scales using statistics robust to outliers
        - Custom scalers implementing fit/transform interface

        References
        ----------
        .. [1] Scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
        """
        for covariate in data.covariates.to_numpy():
            if (covariate not in self.inscalers) or overwrite:
                self.inscalers[covariate] = Scaler.from_string(self.norm_conf.inscaler)
                self.inscalers[covariate].fit(data.X.sel(covariates=covariate).data)

        for responsevar in data.response_vars.to_numpy():
            if (responsevar not in self.outscalers) or overwrite:
                self.outscalers[responsevar] = Scaler.from_string(
                    self.norm_conf.outscaler
                )
                self.outscalers[responsevar].fit(
                    data.y.sel(response_vars=responsevar).data
                )

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

        Returns
        -------
        None
            Modifies the input data object in-place by adding or updating:
            - X: unscaled covariates
            - y: unscaled responses
            - yhat: unscaled predictions
            - ys2: unscaled prediction variances
            - centiles: unscaled prediction centiles

        Notes
        -----
        The method:
        - Uses stored inscalers for covariates
        - Uses stored outscalers for responses and predictions
        - Preserves the original data structure and coordinates
        - Handles missing data appropriately
        - Maintains data consistency across all variables

        The inverse scaling is applied to all relevant variables that were previously
        scaled during preprocessing. The original scalers must have been created and
        stored during the forward scaling process.

        Examples
        --------
        >>> # Assuming model has been fitted and predictions made
        >>> predictions = model.predict(test_data)
        >>> # Scale predictions back to original scale
        >>> model.scale_backward(predictions)
        >>> # Access unscaled predictions
        >>> unscaled_yhat = predictions.yhat
        >>> unscaled_ys2 = predictions.ys2

        See Also
        --------
        scale_forward : Method for forward scaling of data
        preprocess : Method handling complete preprocessing pipeline
        NormData.scale_backward : Underlying scaling implementation

        Warnings
        --------
        - Requires scalers to be previously fitted
        - May produce unexpected results if applied multiple times
        - Should be used only on data scaled with corresponding forward scalers
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
            self.basis_function = create_basis_function(self.norm_conf.basis_function, source_array, **self.norm_conf.basis_function_kwargs)
        if not self.basis_function.is_fitted:
            self.basis_function.fit(data)
        self.basis_function.transform(data)
       
    def save(self, path: Optional[str] = None) -> None:
        if path is not None:
            self.norm_conf.set_save_dir(path)

        os.makedirs(os.path.join(self.norm_conf.save_dir, "model"), exist_ok=True)

        metadata = {
            "norm_conf": self.norm_conf.to_dict(),
            "regression_model_type": self.regression_model_type.__name__,
            "default_reg_conf": self.default_reg_conf.to_dict(),
            "inscalers": {k: v.to_dict() for k, v in self.inscalers.items()},
        }

        # TODO make this cleaner and more general
        if self.norm_conf.basis_function == "bspline" and hasattr(
            self, "bspline_basis"
        ):
            metadata["basis_function"] = copy.deepcopy(self.basis_function)

        os.makedirs(self.norm_conf.save_dir, exist_ok=True)
        print(os.getpid(), f"Saving model to {self.norm_conf.save_dir}")

        model_save_path = os.path.join(self.norm_conf.save_dir, "model")
        with open(
            os.path.join(model_save_path,"normative_model.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=4)

        for responsevar, model in self.regression_models.items():
            reg_model_dict = {}
            reg_model_save_path = os.path.join(model_save_path, f"{responsevar}")
            os.makedirs(reg_model_save_path, exist_ok=True)
            reg_model_dict["model"] = model.to_dict(reg_model_save_path)
            reg_model_dict['outscaler'] = self.outscalers[responsevar].to_dict()
            with open(os.path.join(reg_model_save_path,"regression_model.json"), "w", encoding="utf-8") as f:
                json.dump(reg_model_dict, f, indent=4)

        abspath = os.path.abspath(self.norm_conf.save_dir)
        print(f"Model saved to {abspath}/model")

        

    @classmethod
    def load(cls, path: str) -> NormBase:
        model_path = os.path.join(path, "model", "normative_model.json")   
        with open(model_path, mode="r", encoding="utf-8") as f:
            metadata = json.load(f)

        self = cls(NormConf.from_dict(metadata["norm_conf"]))

        self.response_vars = []
        self.outscalers ={}
        self.regression_models = {}
        reg_models_path = os.path.join(path, "model","*")
        for path in glob.glob(reg_models_path):
            if os.path.isdir(path):
                with open(os.path.join(path,"regression_model.json"), mode="r", encoding="utf-8") as f:
                    reg_model_dict = json.load(f)
                    responsevar = reg_model_dict["model"]["name"]
                    self.response_vars.append(responsevar)
                    self.regression_models[responsevar] = self.regression_model_type.from_dict(reg_model_dict["model"], path)
                    self.outscalers[responsevar] = Scaler.from_dict(reg_model_dict["outscaler"])

        
        self.regression_model_type = globals()[metadata["regression_model_type"]]

        if "basis_function" in metadata:
            self.basis_function = create_basis_function(metadata["basis_function"])
        self.inscalers = {
            k: Scaler.from_dict(v) for k, v in metadata["inscalers"].items()
        }
        self.default_reg_conf = type(
            self.regression_models[self.response_vars[0]].reg_conf
        ).from_dict(metadata["default_reg_conf"])
        return self


    def save_results(self, data: NormData) -> None:
        os.makedirs(os.path.join(self.norm_conf.save_dir, "results"), exist_ok=True)
        self.save_zscores(data)
        self.save_centiles(data)
        self.save_measures(data)
        os.makedirs(os.path.join(self.norm_conf.save_dir, "plots"), exist_ok=True)
        plot_centiles(self, data, save_dir=os.path.join(self.norm_conf.save_dir, "plots"), show_data=True)
        plot_qq(data, save_dir=os.path.join(self.norm_conf.save_dir, "plots"))
        abspath = os.path.abspath(self.norm_conf.save_dir)
        print(f"Results and plots saved to {abspath}/results and {abspath}/plots")

    def save_zscores(self, data: NormData) -> None:
        zdf = data.zscores.to_dataframe().unstack(level="response_vars")
        zdf.columns=zdf.columns.droplevel(0)
        res_path = os.path.join(self.norm_conf.save_dir, "results", "zscores.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try: 
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=0) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    new_results = pd.concat([old_results, zdf], axis=1)
                else:
                    new_results = zdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    

    def save_centiles(self, data: NormData) -> None:
        cdf = data.centiles.to_dataframe().unstack(level="response_vars")
        cdf.columns=cdf.columns.droplevel(0)
        res_path = os.path.join(self.norm_conf.save_dir, "results", "centiles.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try: 
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=[0,1]) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    new_results = pd.concat([old_results, cdf], axis=1)
                else:
                    new_results = cdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


    def save_measures(self, data: NormData) -> None:
        mdf = data.measures.to_dataframe().unstack(level="response_vars")
        mdf.columns=mdf.columns.droplevel(0)
        res_path = os.path.join(self.norm_conf.save_dir, "results", "measures.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try: 
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=0) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    new_results = pd.concat([old_results, mdf], axis=1)
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

        Returns
        -------
        None
            Modifies the internal state of the normative model by:
            - Setting self.focused_var
            - Creating/accessing self.regression_models[responsevar]
            - Setting self.focused_model reference

        Notes
        -----
        This method is typically called before performing operations that work with
        individual response variables, such as:
        - Model fitting
        - Prediction
        - Transfer learning
        - Model evaluation

        The method implements a lazy initialization strategy for regression models,
        creating them only when needed.

        Examples
        --------
        >>> model = NormBase(norm_conf)
        >>> model.focus('brain_region_1')
        >>> # Now model.focused_model refers to the regression model for 'brain_region_1'
        >>> model.focused_model.fit(data)
        >>>
        >>> # Switch focus to another variable
        >>> model.focus('brain_region_2')
        >>> # Now working with the model for 'brain_region_2'
        >>> predictions = model.focused_model.predict(test_data)

        See Also
        --------
        reset : Method to clear the current focus
        get_reg_conf : Method to get regression configuration for a response variable

        Notes
        -----
        - The focused model is accessible through self.focused_model property
        - New regression models are initialized with default configuration
        - Focus state persists until explicitly changed or reset
        - Thread safety should be considered in concurrent operations

        Warnings
        --------
        - Ensure responsevar exists in the training data
        - Be mindful of memory usage when creating many regression models
        - Consider resetting focus when switching between response variables
        """
        self.focused_var = responsevar
        if responsevar not in self.regression_models:
            self.regression_models[responsevar] = self.regression_model_type(
                responsevar, self.get_reg_conf(responsevar)
            )

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

        Examples
        --------
        >>> model = NormBase(norm_conf)
        >>> # Get config for existing model
        >>> config = model.get_reg_conf('brain_region_1')
        >>>
        >>> # Get default config for new variable
        >>> default_config = model.get_reg_conf('new_region')

        See Also
        --------
        focus : Method to set focus on a response variable
        RegConf : Regression configuration class
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
    def _transfer(self, data: NormData, **kwargs: Any) -> RegressionModel:
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
    def _tune(self, data: NormData) -> NormBase:
        """
        Tunes the model parameters using the provided data.

        Parameters
        ----------
        data : NormData
            The data to use for tuning the model, containing covariates and response variables.

        Returns
        -------
        NormBase
            The tuned normative model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model = ConcreteNormModel(norm_conf)
        >>> tuned_model = model._tune(validation_data)
        """

    @abstractmethod
    def _merge(self, other: NormBase) -> NormBase:
        """
        Merges the current model with another normative model.

        Parameters
        ----------
        other : NormBase
            The other normative model to merge with.

        Returns
        -------
        NormBase
            The merged normative model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Examples
        --------
        >>> model1 = ConcreteNormModel(norm_conf)
        >>> model2 = ConcreteNormModel(norm_conf)
        >>> merged_model = model1._merge(model2)
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
        >>> args = {'param1': value1, 'param2': value2}
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
    def focused_model(self) -> RegressionModel:
        """Returns the regression model that is currently focused on."""
        return self[self.focused_var]

    @focused_model.setter
    def focused_model(self, value: str) -> None:
        self.focused_var = value
