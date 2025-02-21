from __future__ import annotations

import copy
import fcntl
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math.scaler import Scaler

# pylint: disable=unused-import
from pcntoolkit.regression_model.blr import BLR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.hbr import HBR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.evaluator import Evaluator
from pcntoolkit.util.output import Errors, Messages, Output, Warnings
from pcntoolkit.util.plotter import plot_centiles, plot_qq


class NormativeModel:
    """
    This class provides the foundation for building normative models, handling multiple
    response variables through separate regression models. It manages data preprocessing,
    model fitting, prediction, and evaluation.

    Parameters
    ----------
    template_reg_model : RegressionModel
        Regression model used as a template to create all regression models.
    savemodel : bool
        Whether to save the model.
    evaluate_model : bool
        Whether to evaluate the model.
    saveresults : bool
        Whether to save the results.
    saveplots : bool
        Whether to save the plots.
    save_dir : str
        Directory to save the model, results, and plots.
    inscaler : str
        Input scaler to use.
    """

    def __init__(
        self,
        template_reg_model: RegressionModel,
        savemodel: bool = True,
        evaluate_model: bool = True,
        saveresults: bool = True,
        saveplots: bool = True,
        save_dir: str = "./saves",
        inscaler: str = "none",
        outscaler: str = "none",
        normative_model_name: Optional[str] = None,
    ):
        self.savemodel: bool = savemodel
        self.evaluate_model: bool = evaluate_model
        self.saveresults: bool = saveresults
        self.saveplots: bool = saveplots
        self.save_dir: str = save_dir
        self.inscaler: str = inscaler
        self.outscaler: str = outscaler
        self.normative_model_name: Optional[str] = normative_model_name
        self.response_vars: list[str] = None  # type: ignore
        self.template_regression_model: RegressionModel = template_reg_model
        self.regression_models: dict[str, RegressionModel] = {}
        self.evaluator = Evaluator()
        self.inscalers: dict = {}
        self.outscalers: dict = {}
        self.is_fitted: bool = False

    def fit(self, data: NormData) -> None:
        """
        Fits a regression model for each response variable in the data.

        Parameters
        ----------
        data : NormData
            Training data containing covariates (X), batch effects (batch_effects), and response variables (Y).
            Must be a valid NormData object with properly formatted dimensions:
            - X: (n_samples, n_covariates)
            - batch_effects: (n_samples, n_batch_effects)
            - Y: (n_samples, n_response_vars)

        """
        self.register_data_info(data)
        self.preprocess(data)
        Output.print(Messages.FITTING_MODELS, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            Output.print(Messages.FITTING_MODEL, model_name=responsevar)
            resp_fit_data = data.sel({"response_vars": responsevar})
            X, be, be_maps, Y = self.extract_data(resp_fit_data)
            self[responsevar].fit(X, be, be_maps, Y)
        self.is_fitted = True

        self.postprocess(data)

        if self.savemodel:
            self.save()
        if self.evaluate_model:
            self.evaluate(data)
        if self.saveresults:
            self.save_results(data)
        if self.saveplots:
            self.save_plots(data)

    def predict(self, data: NormData) -> NormData:
        """Computes Z-scores for each response variable using fitted regression models."""
        self.compute_zscores(data)
        if self.evaluate_model:
            self.evaluate(data)
        if self.saveresults:
            self.save_results(data)
        if self.saveplots:
            self.save_plots(data)
        return data

    def synthesize(self, data: NormData | None = None, n_samples: int | None = None) -> NormData:  # type: ignore
        """Returns synthetic Data

        Parameters
        ----------
        data : NormData, optional
            X and batch_effects are optional. If provided, they are used to generate the synthetic data.
            If not provided, the model is used to generate the synthetic data.
            If only batch_effects are provided, batch_effects are sampled from the model.
            If only X is provided, covariates are sampled from the model.
            If neither X nor batch_effects are provided, the model is used to generate the synthetic data.
        n_samples : int, optional
            Number of samples to synthesize. If this is None, the number of samples that were in the train data is used.
        """
        assert self.is_fitted
        if data:
            self.check_compatibility(data)
            data = copy.deepcopy(data)
            if n_samples is not None:
                Output.warning(Warnings.SYNTHESIZE_N_SAMPLES_IGNORED, n_samples=n_samples)
            n_samples = data.X.shape[0] if data.X is not None else data.batch_effects.shape[0]
            if not hasattr(data, "batch_effects") or data.batch_effects is None:
                data["batch_effects"] = self.sample_batch_effects(n_samples)  # type: ignore
            if not hasattr(data, "X") or data.X is None:
                data["X"] = self.sample_covariates(data.batch_effects)
        else:
            if n_samples is None:
                n_samples = self.n_fit_datapoints
            bes = self.sample_batch_effects(n_samples)
            X = self.sample_covariates(bes)
            data = NormData(
                name="synthesized",
                data_vars={"X": X, "batch_effects": bes},
                coords={
                    "datapoints": np.arange(n_samples),
                    "response_vars": self.response_vars,
                },
            )

        data["Z"] = xr.DataArray(
            np.random.normal(0, 1, size=(n_samples, len(self.response_vars))),  # type: ignore
            dims=("datapoints", "response_vars"),
        )
        data["Y"] = xr.DataArray(
            np.zeros((n_samples, len(self.response_vars))),  # type: ignore
            dims=("datapoints", "response_vars"),
        )
        self.preprocess(data)
        Output.print(Messages.SYNTHESIZING_DATA, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            Output.print(Messages.SYNTHESIZING_DATA_MODEL, model_name=responsevar)
            resp_fit_data = data.sel({"response_vars": responsevar})
            resp_Z_data = data.Z.sel({"response_vars": responsevar})
            X, be, be_maps, _ = self.extract_data(resp_fit_data)
            Z_pred = self[responsevar].backward(X, be, be_maps, resp_Z_data)
            data["Y"].loc[{"response_vars": responsevar}] = Z_pred
        self.postprocess(data)
        return data

    def harmonize(self, data: NormData, reference_batch_effect: dict[str, str] | None = None) -> NormData:
        """Harmonizes the data to a reference batch effect. Harmonizes to the provided reference batch effect if provided,
        otherwise, harmonizes to the first batch effect alphabetically.

        Parameters
        ----------
        data : NormData
            Data to harmonize.
        reference_batch_effect : dict[str, str]
            Reference batch effect.
        """
        self.preprocess(data)
        _, be, _, _ = self.extract_data(data)
        ref_be_array = be.astype(str)
        if not reference_batch_effect:
            ref_be = {k: v[0] for k, v in data.get_single_batch_effect().items()}
        else:
            for k, v in data.get_single_batch_effect().items():
                if k not in reference_batch_effect:
                    reference_batch_effect[k] = v[0]
            ref_be = reference_batch_effect
        for k, v in ref_be.items():
            ref_be_array.loc[{"batch_effect_dims": k}] = v
        ref_be_array = self.map_batch_effects(ref_be_array)
        Output.print(Messages.HARMONIZING_DATA, n_models=len(self.response_vars))

        data["Y_harmonized"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(self.response_vars))),
            dims=("datapoints", "response_vars"),
            coords={"datapoints": data.datapoints, "response_vars": self.response_vars},
        )
        for responsevar in self.response_vars:
            Output.print(Messages.HARMONIZING_DATA_MODEL, model_name=responsevar)
            resp_fit_data = data.sel({"response_vars": responsevar})
            X, be, be_maps, Y = self.extract_data(resp_fit_data)
            Z_pred = self[responsevar].forward(X, be, be_maps, Y)
            Y_harmonized = self[responsevar].backward(X, ref_be_array, be_maps, Z_pred)
            data["Y_harmonized"].loc[{"response_vars": responsevar}] = Y_harmonized
        self.is_fitted = True

        self.postprocess(data)
        return data

    def compute_zscores(self, data: NormData) -> NormData:
        """
        Computes Z-scores for each response variable using fitted regression models.

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).

        Returns
        -------
        NormData
            Prediction results containing:
            - Zscores: z-scores of the response variables
        """
        assert self.is_fitted, "Model is not fitted!"
        assert self.check_compatibility(data), "Data is not compatible with the model!"

        self.preprocess(data)
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        Output.print(Messages.PREDICTING_MODELS, n_models=len(respvar_intersection))

        data["Z"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("datapoints", "response_vars"),
            coords={
                "datapoints": data.datapoints,
                "response_vars": list(respvar_intersection),
            },
        )
        Output.print(Messages.COMPUTING_ZSCORES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            Output.print(Messages.COMPUTING_ZSCORES_MODEL, model_name=responsevar)
            resp_predict_data = data.sel({"response_vars": responsevar})
            X, be, be_maps, Y = self.extract_data(resp_predict_data)
            data["Z"].loc[{"response_vars": responsevar}] = self[responsevar].forward(X, be, be_maps, Y)

        self.postprocess(data)
        return data

    def compute_centiles(self, data: NormData, centiles: Optional[List[float] | np.ndarray] = None, **kwargs) -> NormData:
        """
        Computes the centiles for each response variable in the data.

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).
        centiles : np.ndarray, optional
            The centiles to compute. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].

        Returns
        -------
        NormData
            Prediction results containing:
            - Centiles: centiles of the response variables
        """
        self.preprocess(data)

        if centiles is None:
            centiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        if isinstance(centiles, list):
            centiles = np.array(centiles)

        ppf = stats.norm.ppf(centiles)

        # Drop the centiles and dimensions if they already exist
        centiles_already_computed = "centiles" in data or "centile" in data.coords
        if centiles_already_computed:
            data = data.drop_vars(["centiles"])
            data = data.drop_dims(["centile"])

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["centiles"] = xr.DataArray(
            np.zeros((centiles.shape[0], data.X.shape[0], len(respvar_intersection))),
            dims=("centile", "datapoints", "response_vars"),
            coords={"centile": centiles},
        )

        Output.print(Messages.COMPUTING_CENTILES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            Output.print(Messages.COMPUTING_CENTILES_MODEL, model_name=responsevar)
            X, be, be_maps, _ = self.extract_data(resp_predict_data)
            for p, c in zip(ppf, centiles):
                Z = xr.DataArray(np.full(resp_predict_data.X.shape[0], p), dims=("datapoints",))
                data["centiles"].loc[{"response_vars": responsevar, "centile": c}] = self[responsevar].backward(X, be, be_maps, Z)

        self.postprocess(data)
        return data

    def compute_logp(self, data: NormData) -> NormData:
        """
        Computes the log-probability of the data under the model.

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).

        Returns
        -------
        NormData
            Prediction results containing:
            - Logp: log-probability of the response variables per datapoint
        """
        self.preprocess(data)

        # Drop the centiles and dimensions if they already exist
        centiles_already_computed = "logp" in data
        if centiles_already_computed:
            data = data.drop_vars(["logp"])

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["logp"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("datapoints", "response_vars"),
            coords={"datapoints": data.datapoints},
        )

        Output.print(Messages.COMPUTING_CENTILES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            X, be, be_maps, Y = self.extract_data(resp_predict_data)
            Output.print(Messages.COMPUTING_CENTILES_MODEL, model_name=responsevar)
            data["logp"].loc[{"response_vars": responsevar}] = self[responsevar].elemwise_logp(X, be, be_maps, Y)

        self.postprocess(data)
        return data

    def evaluate(self, data: NormData) -> None:
        """
        Evaluates the model performance.
        This method performs the following steps:
        1. Preprocesses the data
        2. Computes the Z-scores
        3. Computes the centiles
        4. Computes the log-probability of the data
        5. Evaluates the model performance
        6. Postprocesses the data
        """
        self.preprocess(data)
        self.compute_zscores(data)
        self.compute_centiles(data)
        self.compute_logp(data)
        self.evaluator.evaluate(data)
        # self.model_specific_evaluation()
        self.postprocess(data)

    def model_specific_evaluation(self) -> None:
        """
        Save model-specific evaluation metrics.
        """
        for responsevar in self.response_vars:
            self[responsevar].model_specific_evaluation(self.save_dir)

    def fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Combines model.fit and model.predict in a single operation.
        """
        self.fit(fit_data)
        self.predict(predict_data)
        return predict_data

    def extract_data(self, data: NormData) -> Tuple[xr.DataArray, xr.DataArray, dict[str, dict[str, int]], xr.DataArray]:
        if hasattr(data, "X"):
            X = data.X
        else:
            X = None
        if hasattr(data, "batch_effects"):
            batch_effects = self.map_batch_effects(data.batch_effects)
            batch_effects_maps = self.batch_effects_maps
        else:
            batch_effects = None
            batch_effects_maps = None
        if hasattr(data, "Y"):
            Y = data.Y
        else:
            Y = None
        return X, batch_effects, batch_effects_maps, Y  # type: ignore

    def transfer(self, transfer_data: NormData, save_dir: str | None = None, **kwargs) -> NormativeModel:
        """
        Transfers the model to a new dataset.
        """
        new_model = NormativeModel(
            copy.deepcopy(self.template_regression_model),
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            inscaler=self.inscaler,
            outscaler=self.outscaler,
            save_dir=self.save_dir,
            
        )
        if save_dir is not None:
            new_model.save_dir = save_dir
        else:
            new_model.save_dir = self.save_dir+"_transfer"
        new_model.covariates = copy.deepcopy(self.covariates)
        new_model.response_vars = copy.deepcopy(self.response_vars)
        new_model.inscalers = copy.deepcopy(self.inscalers)
        new_model.outscalers = copy.deepcopy(self.outscalers)
        new_model.preprocess(transfer_data)
        new_model.register_batch_effects(transfer_data)
        Output.print(Messages.TRANSFERRING_MODELS, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            Output.print(Messages.TRANSFERRING_MODEL, model_name=responsevar)
            resp_transfer_data = transfer_data.sel({"response_vars": responsevar})
            X, be, be_maps, Y = new_model.extract_data(resp_transfer_data)
            new_model[responsevar] = self[responsevar].transfer(X, be, be_maps, Y, **kwargs)
        new_model.is_fitted = True
        new_model.postprocess(transfer_data)
        return new_model
    
    def transfer_predict(self, transfer_data: NormData, predict_data: NormData, save_dir: str | None = None) -> NormativeModel:
        """
        Transfers the model to a new dataset and predicts the data.
        """
        new_model = self.transfer(transfer_data, save_dir )
        new_model.predict(predict_data)
        return new_model
    
    def extend(self, data: NormData, save_dir: str | None = None, n_synth_samples: int | None = None) -> NormativeModel:
        """
        Extends the model to a new dataset.
        """
        synth = self.synthesize(n_samples=n_synth_samples)
        self.postprocess(synth)
        self.postprocess(data)
        merged_data = data.merge(synth)
        if save_dir is None:
            save_dir = self.save_dir+"_extend"
            
        new_model = NormativeModel(
            copy.deepcopy(self.template_regression_model),
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            inscaler=self.inscaler,
            outscaler=self.outscaler,
            save_dir=save_dir,
        )

        new_model.inscalers = copy.deepcopy(self.inscalers)
        new_model.outscalers = copy.deepcopy(self.outscalers)
        new_model.fit(merged_data)
        return new_model

    def extend_predict(self, extend_data: NormData, predict_data: NormData, save_dir: str | None = None, n_synth_samples: int | None = None) -> NormativeModel:
        """
        Extends the model to a new dataset and predicts the data.
        """
        new_model = self.extend(extend_data, save_dir, n_synth_samples)
        new_model.predict(predict_data) 
        return new_model
    
    def preprocess(self, data: NormData) -> None:
        """
        Applies preprocessing transformations to the input data.

        Args:
            data (NormData): Data to preprocess.
        """
        self.scale_forward(data)

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
        missing_covariates = [i for i in self.covariates if i not in data.covariates.values]
        if len(missing_covariates) > 0:
            Output.warning(
                Warnings.MISSING_COVARIATES,
                covariates=missing_covariates,
                dataset_name=data.name,
            )

        extra_covariates = [i for i in data.covariates.values if i not in self.covariates]
        if len(extra_covariates) > 0:
            Output.warning(
                Warnings.EXTRA_COVARIATES,
                covariates=extra_covariates,
                dataset_name=data.name,
            )

        extra_response_vars = [i for i in data.response_vars.values if i not in self.response_vars]
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
        if not compatible:
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
        self.unique_batch_effects = {k: list(v) for k, v in self.unique_batch_effects.items()}
        self.batch_effects_maps = {
            be: {k: i for i, k in enumerate(self.unique_batch_effects[be])} for be in self.unique_batch_effects.keys()
        }
        self.batch_effect_counts = copy.deepcopy(data.batch_effect_counts)
        self.batch_effect_covariate_ranges = copy.deepcopy(data.batch_effect_covariate_ranges)

    def map_batch_effects(self, batch_effects: xr.DataArray) -> xr.DataArray:
        mapped_batch_effects = np.zeros(batch_effects.values.shape).astype(int)
        for i, be in enumerate(self.unique_batch_effects.keys()):
            for j, v in enumerate(batch_effects.values[:, i]):
                mapped_batch_effects[j, i] = self.batch_effects_maps[be][v]
        mapped_batch_effects = xr.DataArray(
            mapped_batch_effects,
            dims=("datapoints", "batch_effect_dims"),
            coords={"batch_effect_dims": list(self.unique_batch_effects.keys())},
        )
        return mapped_batch_effects

    def sample_batch_effects(self, n_samples: int) -> xr.DataArray:
        """
        Sample the batch effects from the estimated distribution.
        """
        max_batch_effect_count = max([len(v) for v in self.unique_batch_effects.values()])
        if n_samples < max_batch_effect_count:
            raise ValueError(Output.error(
                Errors.SAMPLE_BATCH_EFFECTS,
                n_samples=n_samples,
                max_batch_effect_count=max_batch_effect_count,
            )) 

        bes = xr.DataArray(
            np.zeros((n_samples, len(self.batch_effect_counts.keys()))).astype(str),
            dims=("datapoints", "batch_effect_dims"),
            coords={"datapoints": np.arange(n_samples), "batch_effect_dims": self.batch_effect_dims},
        )
        for be in self.batch_effect_dims:
            countsum = np.sum(list(self.batch_effect_counts[be].values()))
            bes.loc[{"batch_effect_dims": be}] = np.random.choice(
                list(self.batch_effect_counts[be].keys()),
                size=n_samples,
                p=[c / countsum for c in list(self.batch_effect_counts[be].values())],
            )
        return bes

    def sample_covariates(self, bes: xr.DataArray) -> xr.DataArray:
        """
        Sample the covariates from the estimated distribution.

        Uses ranges of observed covariates matched with batch effects to create a representative sample
        """
        X = xr.DataArray(
            np.zeros((bes.shape[0], len(self.covariates))),
            dims=("datapoints", "covariates"),
            coords={"datapoints": np.arange(bes.shape[0]), "covariates": self.covariates},
        )
        for c in self.covariates:
            for i in range(X.shape[0]):
                running_min, running_max = -np.inf, np.inf
                for k in self.batch_effect_dims:
                    my_be = bes.sel({"datapoints": i, "batch_effect_dims": k}).values.item()
                    running_min = max(running_min, self.batch_effect_covariate_ranges[k][my_be][c]["min"])
                    running_max = min(running_max, self.batch_effect_covariate_ranges[k][my_be][c]["max"])

                X.loc[{"datapoints": i, "covariates": c}] = np.random.uniform(running_min, running_max, size=1).item()
        return X

    def scale_forward(self, data: NormData, overwrite: bool = False) -> None:
        """
        Scales input data to standardized form using configured scalers.

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
                self.inscalers[covariate] = Scaler.from_string(self.inscaler)
                self.inscalers[covariate].fit(data.X.sel(covariates=covariate).data)

        for responsevar in data.response_vars.to_numpy():
            if (responsevar not in self.outscalers) or overwrite:
                self.outscalers[responsevar] = Scaler.from_string(self.outscaler)
                self.outscalers[responsevar].fit(data.Y.sel(response_vars=responsevar).data)

        data.scale_forward(self.inscalers, self.outscalers)

    def scale_backward(self, data: NormData) -> None:
        """
        Scales data back to its original scale using stored scalers.

        Parameters
        ----------
        data : NormData
            Data object containing arrays to be scaled back:
            - X : array-like, shape (n_samples, n_covariates)
                Covariate data to be scaled back
            - y : array-like, shape (n_samples, n_response_vars), optional
                Response variable data to be scaled back
        """
        data.scale_backward(self.inscalers, self.outscalers)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model to a file.

        Args:
            path (str, optional): The path to save the model to. If None, the model is saved to the save_dir provided in the norm_conf.
        """
        savepath = path if path is not None else self.save_dir
        savepath = os.path.join(savepath, "model")
        Output.print(Messages.SAVING_MODEL, save_dir=savepath)
        metadata = {
            "save_dir": self.save_dir,
            "savemodel": self.savemodel,
            "saveresults": self.saveresults,
            "saveplots": self.saveplots,
            "evaluate_model": self.evaluate_model,
            "normative_model_name": self.normative_model_name,
            "template_regression_model": self.template_regression_model.to_dict(),
            "inscalers": {k: v.to_dict() for k, v in self.inscalers.items()},
            "unique_batch_effects": self.unique_batch_effects,
            "is_fitted": self.is_fitted,
            "inscaler": self.inscaler,
            "outscaler": self.outscaler,
        }

        if hasattr(self, "covariates"):
            metadata["covariates"] = self.covariates

        # JSON keys are always string, so we have use a trick to save the batch effects distributions, which may have string or int keys.
        # We invert the map, so that the original keys are stored as the values
        # The original integer values are then converted to strings by json, but we can safely convert them back to ints when loading
        # We also add an index to the keys to make sure they are unique
        if hasattr(self, "batch_effect_counts"):
            metadata["inverse_batch_effect_counts"] = {
                be: {f"{i}_{k}": v for i, (v, k) in enumerate(self.batch_effect_counts[be].items())}
                for be, mp in self.batch_effect_counts.items()
            }
        if hasattr(self, "batch_effects_maps"):
            metadata["batch_effects_maps"] = {
                be: {f"{i}_{k}": v for i, (v, k) in enumerate(self.batch_effects_maps[be].items())}
                for be in self.batch_effects_maps.keys()
            }
        if hasattr(self, "batch_effect_covariate_ranges"):
            metadata["batch_effect_covariate_ranges"] = copy.deepcopy(self.batch_effect_covariate_ranges)

        os.makedirs(savepath, exist_ok=True)
        with open(
            os.path.join(savepath, "normative_model.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=4)

        for responsevar, model in self.regression_models.items():
            reg_model_save_path = os.path.join(savepath, f"{responsevar}")
            os.makedirs(reg_model_save_path, exist_ok=True)
            reg_model_dict = {}
            reg_model_dict["model"] = model.to_dict(reg_model_save_path)
            reg_model_dict["outscaler"] = self.outscalers[responsevar].to_dict()
            json_save_path = os.path.join(reg_model_save_path, "regression_model.json")
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(reg_model_dict, f, indent=4)

    @classmethod
    def load(cls, path: str, into: NormativeModel | None = None) -> NormativeModel:
        """
        Load a normative model from a path.

        Parameters
        ----------
        path : str
            The path to the normative model.
        into : NormBase, optional
            The normative model to load the data into. If None, a new normative model is created.
            This is useful if you want to load a normative model into an existing normative model, for example in the runner.
        """
        model_path = os.path.join(path, "model", "normative_model.json")
        with open(model_path, mode="r", encoding="utf-8") as f:
            metadata = json.load(f)

        savemodel = metadata["savemodel"]
        saveresults = metadata["saveresults"]
        save_dir = metadata["save_dir"]
        inscaler = metadata["inscaler"]
        outscaler = metadata["outscaler"]
        saveplots = metadata["saveplots"]
        evaluate_model = metadata["evaluate_model"]
        normative_model_name = metadata["normative_model_name"]

        response_vars = []
        outscalers = {}
        regression_models = {}
        reg_models_path = os.path.join(path, "model", "*")
        for path in glob.glob(reg_models_path):
            if os.path.isdir(path):
                with open(
                    os.path.join(path, "regression_model.json"),
                    mode="r",
                    encoding="utf-8",
                ) as f:
                    reg_model_dict = json.load(f)
                    responsevar = reg_model_dict["model"]["_name"]
                    response_vars.append(responsevar)
                    regression_model_type = globals()[reg_model_dict["model"]["type"]]
                    regression_models[responsevar] = regression_model_type.from_dict(reg_model_dict["model"], path)
                    outscalers[responsevar] = Scaler.from_dict(reg_model_dict["outscaler"])
        template_regression_model = type(regression_models[response_vars[0]]).from_dict(metadata["template_regression_model"])
        if into is None:
            self = cls(
                template_reg_model=template_regression_model,
                savemodel=savemodel,
                evaluate_model=evaluate_model,
                saveresults=saveresults,
                saveplots=saveplots,
                save_dir=save_dir,
                inscaler=inscaler,
                outscaler=outscaler,
                normative_model_name=normative_model_name,
            )
        else:
            self = into

        self.regression_models = regression_models
        self.outscalers = outscalers
        self.response_vars = response_vars
        self.inscalers = {k: Scaler.from_dict(v) for k, v in metadata["inscalers"].items()}

        if "batch_effects_maps" in metadata:
            self.batch_effects_maps = {
                be: {v: int(k.split("_")[1]) for k, v in mp.items()} for be, mp in metadata["batch_effects_maps"].items()
            }
        if "inverse_batch_effect_counts" in metadata:
            self.batch_effect_counts = {
                be: {v: int(k.split("_")[1]) for k, v in mp.items()} for be, mp in metadata["inverse_batch_effect_counts"].items()
            }
        if "batch_effect_covariate_ranges" in metadata:
            self.batch_effect_covariate_ranges = metadata["batch_effect_covariate_ranges"]
        if "unique_batch_effects" in metadata:
            self.unique_batch_effects = metadata["unique_batch_effects"]

        if "covariates" in metadata:
            self.covariates = metadata["covariates"]

        self.is_fitted = metadata["is_fitted"]

        return self

    def save_results(self, data: NormData) -> None:
        Output.print(Messages.SAVING_RESULTS, save_dir=self.save_dir)
        os.makedirs(os.path.join(self.save_dir, "results"), exist_ok=True)
        if hasattr(data, "Z"):
            self.save_zscores(data)
        if hasattr(data, "centiles"):
            self.save_centiles(data)
        if hasattr(data, "measures"):
            self.save_measures(data)

    def save_plots(self, data: NormData) -> None:
        os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)
        if hasattr(data, "Z"):
            plot_qq(data, save_dir=os.path.join(self.save_dir, "plots"))
        if hasattr(data, "centiles"):
            plot_centiles(self, 
                          data, 
                          save_dir=os.path.join(self.save_dir, "plots"), show_data=True,
                          show_other_data=True,
                          harmonize_data=True)

    def save_zscores(self, data: NormData) -> None:
        zdf = data.Z.to_dataframe().unstack(level="response_vars")
        zdf.columns = zdf.columns.droplevel(0)
        res_path = os.path.join(self.save_dir, "results", f"Z_{data.name}.csv")
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
        centiles = data.centiles.to_dataframe().unstack(level="response_vars")
        centiles.columns = centiles.columns.droplevel(0)
        res_path = os.path.join(self.save_dir, "results", f"centiles_{data.name}.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=[0, 1]) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    # Merge on datapoints, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(centiles, on=["datapoints", "centile"], how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = centiles
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save_measures(self, data: NormData) -> None:
        mdf = data.measures.to_dataframe().unstack(level="response_vars")
        mdf.columns = mdf.columns.droplevel(0)
        res_path = os.path.join(self.save_dir, "results", f"measures_{data.name}.csv")
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

    def set_save_dir(self, save_dir: str) -> None:
        """Override the save_dir in the norm_conf.

        Args:
            save_dir (str): New save directory.
        """
        self.save_dir = save_dir

    def __getitem__(self, key: str) -> RegressionModel:
        if key not in self.regression_models:
            self.regression_models[key] = copy.deepcopy(self.template_regression_model)
            self.regression_models[key]._name = key
        return self.regression_models[key]

    def __setitem__(self, key: str, value: RegressionModel) -> None:
        self.regression_models[key] = value

    @property
    def has_batch_effect(self) -> bool:
        """Returns whether the model has a batch effect.
        Returns:
            bool: True if the model has a batch effect, False otherwise. This currently looks at the template reg conf
        """
        return self.template_regression_model.has_batch_effect

    @property
    def batch_effect_dims(self) -> list[str]:
        """Returns the batch effect dimensions.
        Returns:
            list[str]: The batch effect dimensions.
        """
        return list(self.unique_batch_effects.keys())

    @property
    def n_fit_datapoints(self) -> int:
        """Returns the number of batch effects.
        Returns:
            int: The number of batch effects.
        """
        return sum(self.batch_effect_counts[self.batch_effect_dims[0]].values())


# Factory methods #########################################################################################


def create_normative_model_from_args(args: dict[str, str]) -> NormativeModel:
    """
    Create a normative model from command line arguments.

    Parameters
    ----------
    args : dict[str, str]
        A dictionary of command line arguments.

    Returns
    -------
    NormBase
        An instance of a normative model.

    Raises
    ------
    ValueError
        If the regression model specified in the arguments is unknown.
    """
    savemodel = args["savemodel"] == "True"
    saveresults = args["saveresults"] == "True"
    save_dir = args["save_dir"]
    inscaler = args["inscaler"]
    outscaler = args["outscaler"]
    normative_model_name = args["normative_model_name"]
    if args["alg"] == "blr":
        template_reg_model = BLR.from_args("template", args)
    elif args["alg"] == "hbr":
        template_reg_model = HBR.from_args("template", args)
    else:
        raise ValueError(Output.error(Errors.ERROR_UNKNOWN_CLASS, class_name=args["alg"]))
    return NormativeModel(
        template_reg_model=template_reg_model,
        savemodel=savemodel,
        saveresults=saveresults,
        save_dir=save_dir,
        inscaler=inscaler,
        outscaler=outscaler,
        normative_model_name=normative_model_name,
    )
