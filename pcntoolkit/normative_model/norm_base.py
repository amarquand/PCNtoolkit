from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import explained_variance_score

from pcntoolkit.dataio.basis_expansions import create_bspline_basis, create_poly_basis
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.dataio.scaler import scaler
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.gpr.gpr import GPR
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.reg_conf import RegConf
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.evaluator import Evaluator

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

        # The evaluator is used to evaluate the normative model
        self.evaluator = Evaluator()

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
        data = self.compute_zscores(data)
        data = self.compute_centiles(data)
        data = self.evaluator.evaluate(data)

    def compute_centiles(
        self, data: NormData, cummulative_densities=None, *args, **kwargs
    ) -> NormData:
        # Preprocess the data
        self.preprocess(data)

        if cummulative_densities is None:
            cummulative_densities = np.array(
                [0.05, 0.1587, 0.25, 0.5, 0.75, 0.8413, 0.95]
            )

        # If centiles are already computed, remove them
        centiles_already_computed = (
            "scaled_centiles" in data
            or "centiles" in data
            or "cummulative_densities" in data.coords
        )
        if centiles_already_computed:
            data = data.drop_vars(["scaled_centiles", "centiles"])
            data = data.drop_dims(["cummulative_densities"])

        # Create an empty array to store the scaled centiles
        data["scaled_centiles"] = xr.DataArray(
            np.zeros(
                (len(cummulative_densities), data.X.shape[0], len(self.response_vars))
            ),
            dims=("cummulative_densities", "datapoints", "response_vars"),
            coords={"cummulative_densities": cummulative_densities},
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

            # Overwrite centiles
            print("Computing centiles for", responsevar)
            data["scaled_centiles"].loc[{"response_vars": responsevar}] = (
                self._centiles(
                    resp_predict_data, cummulative_densities, *args, **kwargs
                )
            )

            self.reset()

        self.postprocess(data)

        return data

    # def plot_centiles(self, data:NormData):

    def plot_centiles(
        self,
        data: NormData,
        covariate: str = None,
        batch_effects: Dict[str, List[str]] = None,
        cummul_densities=None,
        show_data: bool = False,
        plt_kwargs=None,
    ):
        """Plot the centiles for all response variables."""
        synth_data = data.create_synthetic_data(
            n_datapoints=150,
            range_dim=covariate,
            batch_effects_to_sample=batch_effects,
        )
        self.compute_centiles(synth_data, cummulative_densities=cummul_densities)
        for response_var in data.coords["response_vars"].to_numpy():
            self._plot_centiles(
                data,
                synth_data,
                response_var,
                covariate,
                batch_effects,
                show_data,
                plt_kwargs,
            )

    def _plot_centiles(
        _self,
        data: NormData,
        synth_data: NormData,
        response_var: str,
        covariate: str = None,
        batch_effects: Dict[str, List[str]] = None,
        show_data: bool = False,
        plt_kwargs=None,
    ):
        """Plot the centiles for a single response variable."""
        # Use the first covariate, if not specified
        if covariate is None:
            covariate = data.covariates[0].to_numpy().item()

        if batch_effects is None:
            batch_effects = data.get_single_batch_effect()

        # Set the plt kwargs to an empty dictionary if they are not provided
        if plt_kwargs is None:
            plt_kwargs = {}

        new_batch_effects = []

        # Create a list of batch effects
        for be in batch_effects:
            if isinstance(be, str):
                new_batch_effects.append(be)
            elif isinstance(be, list):
                new_batch_effects.append(be[0])
            else:
                try:
                    new_batch_effects.append(be.item())
                except AttributeError:
                    new_batch_effects.append(be)
        batch_effects_list = new_batch_effects

        # Filter the covariate and responsevar that are to be plotted
        filter_dict = {
            "covariates": covariate,
            "response_vars": response_var,
        }
        filtered = synth_data.sel(filter_dict)

        plt.figure(**plt_kwargs)
        lines = []
        for cdf in synth_data.coords["cummulative_densities"][::-1]:
            d_mean = abs(cdf - 0.5)
            thickness = 3 - 4 * d_mean

            if d_mean < 0.25:
                style = "-"
            elif d_mean < 0.475:
                style = "--"
            else:
                style = ":"

            cmap = plt.get_cmap("viridis")
            color = cmap(cdf)

            lines.extend(
                plt.plot(
                    filtered.X,
                    filtered.centiles.sel(cummulative_densities=cdf),
                    color=color,
                    linewidth=thickness,
                    linestyle=style,
                )
            )
        line_legend = plt.legend(
            lines,
            synth_data.coords["cummulative_densities"].values,
            loc="upper right",
            title="Centiles",
        )
        plt.gca().add_artist(line_legend)

        if show_data:
            filtered_scatter = data.sel(filter_dict)
            idx = np.all(
                np.stack(
                    [
                        filtered_scatter.batch_effects[:, i] == batch_effects_list[i]
                        for i in range(filtered_scatter.batch_effects.shape[1])
                    ],
                    axis=1,
                ),
                axis=1,
            )
            idx = xr.DataArray(idx)
            filt1: xr.Dataset = filtered_scatter.isel(datapoints=list(np.where(idx)[0]))
            be_string = ", ".join([f"{k}={v}" for k, v in batch_effects.items()])
            plt.scatter(
                filt1.X,
                filt1.y,
                label=f"{{{be_string}}})",
            )
            filt2: xr.Dataset = filtered_scatter.isel(
                datapoints=list(np.where(~idx)[0])
            )
            plt.scatter(
                filt2.X,
                filt2.y,
                label=f"other batches",
            )
        if show_data:
            plt.title(f"Centiles of {response_var} with {data.attrs['name']} scatter")
            plt.legend(loc="upper left", title="Batch effect")
        else:
            plt.title(f"Centiles of {response_var}")
        plt.xlabel(covariate)
        plt.ylabel(response_var)
        plt.show()

    def compute_zscores(self, data: NormData, *args, **kwargs) -> NormData:
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
        return data

    def preprocess(self, data: NormData) -> NormData:
        """
        Contains all the general preprocessing logic that is not specific to the regression model.
        """
        self.scale_forward(data)

        # data.scale_forward(self.norm_conf.inscaler, self.norm_conf.outscaler)
        self.expand_basis_new(data, source_array="scaled_X")

    def expand_basis_new(
        self, data: NormData, source_array: str, intercept: bool = False
    ):
        # Expand the basis of the source array
        if source_array == "scaled_X":
            if "scaled_X" not in data.data_vars:
                raise ValueError(
                    "scaled_X does not exist. Please scale the data first using the scale_forward method."
                )
            source_array = data.scaled_X
        elif source_array == "X":
            source_array = data.X

        all_arrays = [source_array.data]
        all_dims = list(data.covariates.to_numpy())

        # Create a new array with the expanded basis
        basis_expansion = self.norm_conf.basis_function
        basis_column = self.norm_conf.basis_column
        if basis_expansion == "polynomial":
            order = self.norm_conf.order
            expanded_basis = create_poly_basis(
                source_array.data[:, basis_column], order
            )
            all_arrays.append(expanded_basis)
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )
        elif basis_expansion == "bspline":
            if not hasattr(self, "bspline_basis"):
                order = self.norm_conf.order
                nknots = self.norm_conf.nknots
                xmin = np.min(source_array.data[:, basis_column])
                xmax = np.max(source_array.data[:, basis_column])
                diff = xmax - xmin
                xmin = xmin - 0.2 * diff
                xmax = xmax + 0.2 * diff
                self.bspline_basis = create_bspline_basis(
                    xmin=xmin,
                    xmax=xmax,
                    p=order,
                    nknots=nknots,
                )
            expanded_basis = np.array(
                [self.bspline_basis(c) for c in source_array.data[:, basis_column]]
            )
            all_arrays.append(expanded_basis)
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )

        if intercept:
            all_dims.append("intercept")
            all_arrays.append(np.ones((expanded_basis.shape[0], 1)))

        Phi = np.concatenate(all_arrays, axis=1)
        data["Phi"] = xr.DataArray(
            Phi,
            coords={"basis_functions": all_dims},
            dims=["datapoints", "basis_functions"],
        )
        pass

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

        # Set the regression model type
        self.regression_model_type = globals()[model_dict["regression_model_type"]]

        # Set the regression models
        self.regression_models = self.dict_to_regression_models(
            model_dict["regression_models"], path
        )

        # Load the default regression model configuration
        # Get the first regression model
        first_regression_model = next(iter(self.regression_models.values()))

        # Get the class of the reg_conf object
        reg_conf_class = first_regression_model.reg_conf.__class__

        # Create a new instance of the reg_conf class from the dictionary
        self.default_reg_conf = reg_conf_class.from_dict(model_dict["default_reg_conf"])

        # Load the bspline basis expansion
        if normconf.basis_function == "bspline":
            if "bspline_basis" in model_dict:
                self.bspline_basis = create_bspline_basis(**model_dict["bspline_basis"])

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
        # store the regression model type
        model_dict["regression_model_type"] = self.regression_model_type.__name__
        # store the default regression model configuration
        model_dict["default_reg_conf"] = self.default_reg_conf.to_dict()

        # Store the bspline_basis expansion if it exists
        if self.norm_conf.basis_function == "bspline":
            if hasattr(self, "bspline_basis"):
                knots = self.bspline_basis.knot_vector
                model_dict["bspline_basis"] = {
                    "xmin": knots[0],
                    "xmax": knots[-1],
                    "nknots": self.norm_conf.nknots,
                    "p": self.norm_conf.order,
                }

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
    def _centiles(
        self, data: NormData, centiles: list[float], *args, **kwargs
    ) -> xr.DataArray:
        """Takes a list of cummulative densities and returns the corresponding centiles of the model.
        The return type is an xr.datarray with dimensions (cummulative_densities, datapoints).
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
