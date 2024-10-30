"""
The NormBase class is the base class for all normative models.
This class holds a number of regression models, one for each response variable.
The class contains methods for fitting, predicting, transferring, extending, tuning, merging, and evaluating the normative model.
This class contains the general logic that is not specific to the regression model. The subclasses implement the actual logic for fitting, predicting, etc.
All the bookkeeping is done in this class, such as keeping track of the regression models, the scalers, the response variables, etc.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import seaborn as sns  # type: ignore
import xarray as xr
from bspline import splinelab  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from pcntoolkit.dataio.basis_expansions import create_bspline_basis, create_poly_basis
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.dataio.scaler import scaler

# pylint: disable=unused-import
from pcntoolkit.regression_model.blr.blr import BLR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.gpr.gpr import GPR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.hbr.hbr import HBR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.reg_conf import RegConf
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.evaluator import Evaluator

from .norm_conf import NormConf


class NormBase(ABC):
    """
    The base class for all normative models.

    Contains a number of regression models, one for each response variable. Has methods for fitting, predicting, transferring, extending,
    tuning, merging, and evaluating. All methods contains general logic that is not specific to the regression model. The subclasses
    implement the actual numerical methods for fitting, predicting, etc. All the bookkeeping is done in this class, such as keeping track
    of the regression models, the scalers, the response variables, etc.
    """

    def __init__(self, norm_conf: NormConf):
        self._norm_conf: NormConf = norm_conf
        object.__setattr__(
            self._norm_conf, "normative_model_name", self.__class__.__name__
        )

        # Response variables is a list of names of the response variables for which the model is fitted
        self.response_vars: list[str] = None  # type: ignore

        # the regression_model_type attribute is used to store the type of regression model
        # should be set by the subclass
        self.regression_model_type: Any = None  # type: ignore

        # the self.defult_reg_conf attribute is used whenever a new regression model is created, and no reg_conf is provided
        # should be set by the subclass
        self.default_reg_conf: RegConf = None  # type: ignore

        # Regression models is a dictionary that contains the regression models
        # - the keys are the response variables
        # - the values are the regression models
        self.regression_models: dict[str, RegressionModel] = {}

        # the self.current_regression_model attribute is used to store the current regression model
        # this model is used internally by the _fit and _predict methods of the subclass
        self.current_response_var: str = None  # type: ignore
        self.current_regression_model: RegressionModel = None  # type: ignore

        # The evaluator is used to evaluate the normative model
        self.evaluator = Evaluator()

        # Inscalers contains a scaler for each covariate
        self.inscalers: dict = {}

        # Outscalers contains a scaler for each response variable
        self.outscalers: dict = {}

        # The basis expansion for the covariates
        self.bspline_basis: splinelab.bspline.Bspline = None  # type: ignore

    def fit(self, data: NormData) -> None:
        """Fits a regression model for each response variable in the data.

        Args:
            data (NormData): Fit data containing the covariates and response variables.
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
        """Makes predictions for each response variable in the data.

        Args:
            data (NormData): Data containing the covariates for which to make predictions.

        Raises:
            ValueError: Error if the model has not been fitted yet.

        Returns:
            NormData: Data containing the predictions.
        """
        # Preprocess the data
        self.preprocess(data)

        # Predict for each response variable
        print(f"Going to predict {len(self.response_vars)} models")
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if responsevar not in self.regression_models:
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
        self.evaluate(data)
        return data

    def fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """Fits and predicts for each response variable in the data.

        Args:
            fit_data (NormData): Data containing the covariates and response variables for fitting.
            predict_data (NormData): Data containing the covariates for which to make predictions.

        Returns:
            NormData: Data containing the predictions.
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
            if responsevar not in self.regression_models:
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
        self.evaluate(predict_data)
        return predict_data

    def transfer(self, data: NormData, *args: Any, **kwargs: Any) -> "NormBase":
        """Transfers the normative model to new data, returning a new normative model.

        Args:
            data (NormData): Transfer data containing the covariates and response variables.

        Raises:
            ValueError: Error if the model has not been fitted yet.

        Returns:
            NormBase: A new normative model transferred to the new data.
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
            if responsevar not in self.regression_models:
                raise ValueError(
                    "Attempted to transfer a model that has not been fitted."
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

        # pylint: disable=too-many-function-args
        transfered_normative_model = self.__class__(
            transfered_norm_conf,
            self.default_reg_conf,  # type: ignore
        )

        # Set the models
        transfered_normative_model.response_vars = (
            data.response_vars.to_numpy().copy().tolist()
        )
        transfered_normative_model.regression_models = transfered_models

        # Return the new model
        return transfered_normative_model

    def extend(self, data: NormData) -> None:
        """Extends the normative model with new data.

        Args:
            data (NormData): Data containing the covariates and response variables to extend the model with.
        """

        # some preparations and preprocessing
        # ...

        self._extend(data)

        # some cleanup and postprocessing
        # ...

    def tune(self, data: NormData) -> None:
        """Tunes the normative model with new data.

        Args:
            data (NormData): Data containing the covariates and response variables to tune the model with.
        """
        # some preparations and preprocessing
        # ...

        self._tune(data)

        # some cleanup and postprocessing
        # ...

    def merge(self, other: "NormBase") -> None:
        """Merges the current normative model with another normative model.

        Args:
            other (NormBase): The other normative model to merge with.

        Raises:
            ValueError: Error if the models are not of the same type.
        """
        # some preparations and preprocessing
        # ...

        if not self.__class__ == other.__class__:
            raise ValueError("Attempted to merge normative models of different types.")

        self._merge(other)

        # some cleanup and postprocessing
        # ...

    def evaluate(self, data: NormData) -> None:
        """Calls evaluation methods on the data.

        Args:
            data (NormData): Data containing the results of the evaluation.
        """
        data = self.compute_zscores(data)
        data = self.compute_centiles(data)
        data = self.evaluator.evaluate(data)

    def compute_centiles(
        self,
        data: NormData,
        cdf: List | np.ndarray | None = None,
        **kwargs: Any,
    ) -> NormData:
        """Computes centiles for each response variable in the data.

        Args:
            data (NormData): Data containing the covariates and response variables.
            cdf (list, optional): A list containing the CDF values corresponding to the centiles. Setting this to [0.5], for instance, would result in computation of the median. Defaults to None.

        Raises:
            ValueError: Error raised if the model for the response variable does not exist.

        Returns:
            NormData: Data extended with the centiles.
        """

        # Preprocess the data
        self.preprocess(data)

        if cdf is None:
            cdf = np.array([0.05, 0.1587, 0.25, 0.5, 0.75, 0.8413, 0.95])
        if isinstance(cdf, list):
            cdf = np.array(cdf)

        # If centiles are already computed, remove them
        centiles_already_computed = (
            "scaled_centiles" in data or "centiles" in data or "cdf" in data.coords
        )
        if centiles_already_computed:
            data = data.drop_vars(["scaled_centiles", "centiles"])
            data = data.drop_dims(["cdf"])

        # Create an empty array to store the scaled centiles
        data["scaled_centiles"] = xr.DataArray(
            np.zeros((cdf.shape[0], data.X.shape[0], len(self.response_vars))),
            dims=("cdf", "datapoints", "response_vars"),
            coords={"cdf": cdf},
        )

        # Predict for each response variable
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # raise an error if the model has not been fitted yet
            if responsevar not in self.regression_models:
                raise ValueError(
                    f"Attempted to find quantiles for model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Overwrite centiles
            print("Computing centiles for", responsevar)
            data["scaled_centiles"].loc[{"response_vars": responsevar}] = (
                self._centiles(resp_predict_data, cdf, **kwargs)
            )

            self.reset()

        self.postprocess(data)

        return data

    def plot_qq(self, data: NormData) -> None:
        """
        Plot QQ plots for all response variables in the data.

        This function calls the plot_qq method of the NormData object.

        Parameters:
        -----------
        data : NormData
            The NormData object containing the data to be plotted.
        """
        data.plot_qq()

    def plot_centiles(
        self,
        data: NormData,
        covariate: str | None = None,
        cummul_densities: list | None = None,
        show_data: bool = False,
        plt_kwargs: dict | None = None,
        hue_data: str = "site",
        markers_data: str = "sex",
        batch_effects: Dict[str, List[str]] | None = None,
    ) -> None:
        """Plot centiles for all response variables in the data.

        Args:
            data (NormData): Data containing the covariates and response variables.
            covariate (str | None, optional): Name of the covariate on the x-axis. Defaults to None.
            cummul_densities (list | None, optional): Which CDF values correspond to the centiles. Defaults to None.
            show_data (bool, optional): Scatter data along with centiles. Defaults to False.
            plt_kwargs (dict | None, optional): Additional kwargs to pt. Defaults to None.
            hue_data (str, optional): Column to use for coloring. Defaults to "site".
            markers_data (str, optional): Column to use for marker styling. Defaults to "sex".
            batch_effects (Dict[str, List[str]] | None, optional): Models with a random effect have different centiles for different batch effects. This parameter allows you to specify for which batch effects to plot the centiles, by providing a dictionary with the batch effect name as key and a list of batch effect values as value. The first values in the lists will be used for computing the centiles. If no list is provided, the batch effect that occurs first in the data will be used. Addtionally, if show_data=True, the dictionary values specify which batch effects are highlighted in the scatterplot.

        Raises:
            ValueError: _description_
        """
        # Use the first covariate, if not specified
        if covariate is None:
            covariate = data.covariates[0].to_numpy().item()
            assert isinstance(covariate, str)

        # Use the first batch effect if not specified
        if batch_effects is None:
            if self.has_random_effect:
                batch_effects = data.get_single_batch_effect()
            else:
                batch_effects = {}

        # Ensure that the batch effects are in the correct format
        if batch_effects:
            for k, v in batch_effects.items():
                if isinstance(v, str):
                    batch_effects[k] = [v]
                elif not isinstance(v, list):
                    raise ValueError(
                        f"Items of the batch_effect dict be a list or a string, not {type(v)}"
                    )

        # Set the plt kwargs to an empty dictionary if they are not provided
        if plt_kwargs is None:
            plt_kwargs = {}

        palette = plt_kwargs.pop("cmap", "viridis")

        synth_data = data.create_synthetic_data(
            n_datapoints=150,
            range_dim=covariate,
            batch_effects_to_sample={k: [v[0]] for k, v in batch_effects.items()}
            if batch_effects
            else None,
        )
        self.compute_centiles(synth_data, cdf=cummul_densities)
        for response_var in data.coords["response_vars"].to_numpy():
            self._plot_centiles(
                data=data,
                synth_data=synth_data,
                response_var=response_var,
                covariate=covariate,
                batch_effects=batch_effects,
                show_data=show_data,
                plt_kwargs=plt_kwargs,
                hue_data=hue_data,
                markers_data=markers_data,
                palette=palette,
            )

    def _plot_centiles(
        self,
        data: NormData,
        synth_data: NormData,
        response_var: str,
        batch_effects: Dict[str, List[str]],
        covariate: str = None,  # type: ignore
        show_data: bool = False,
        plt_kwargs: dict = None,  # type: ignore
        hue_data: str = "site",
        markers_data: str = "sex",
        palette: str = "viridis",
    ) -> None:
        """Plot the centiles for a single response variable."""

        # Set up the plot style
        sns.set_style("whitegrid")
        plt.figure(**plt_kwargs)
        cmap = plt.get_cmap(palette)

        # Filter the covariate and responsevar that are to be plotted
        filter_dict = {
            "covariates": covariate,
            "response_vars": response_var,
        }
        filtered = synth_data.sel(filter_dict)

        # Create centile lines with seaborn
        for cdf in synth_data.coords["cdf"][::-1]:
            # Calculate the style of the line
            d_mean = abs(cdf - 0.5)
            if d_mean < 0.25:
                style = "-"
            elif d_mean < 0.475:
                style = "--"
            else:
                style = ":"

            # Plot centile line
            sns.lineplot(
                x=filtered.X,
                y=filtered.centiles.sel(cdf=cdf),
                color=cmap(cdf),
                linestyle=style,
                linewidth=1,
                zorder=2,
                legend="brief",
            )
            # Add text annotation at the terminal points of the line
            color = cmap(cdf)
            font = FontProperties()
            font.set_weight("bold")
            plt.text(
                s=cdf.item(),
                x=filtered.X[0] - 1,
                y=filtered.centiles.sel(cdf=cdf)[0],
                color=color,
                horizontalalignment="right",
                verticalalignment="center",
                fontproperties=font,
            )
            plt.text(
                s=cdf.item(),
                x=filtered.X[-1] + 1,
                y=filtered.centiles.sel(cdf=cdf)[-1],
                color=color,
                horizontalalignment="left",
                verticalalignment="center",
                fontproperties=font,
            )

        # Increase xlim by 10%
        minx, maxx = plt.xlim()
        plt.xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))

        # # Add legend for centile lines
        # line_legend = plt.legend(
        #     lines,
        #     synth_data.coords["cdf"].values,
        #     title="Percentile",
        # )
        # plt.gca().add_artist(line_legend)

        if show_data:
            df = data.sel(filter_dict).to_dataframe()
            columns = [("X", covariate), ("y", response_var)]
            columns.extend(
                [("batch_effects", be.item()) for be in data.batch_effect_dims]
            )
            df = df[columns]
            df.columns = [c[1] for c in df.columns]

            if batch_effects == {}:
                # If no batch effects are provided, plot all data with slightly larger points
                sns.scatterplot(
                    df,
                    x=covariate,
                    y=response_var,
                    label=data.name,
                    color="black",
                    s=20,  # Slightly larger point size
                    alpha=0.6,
                    zorder=1,
                )
            else:
                # Filter data based on batch effects
                idx = np.full(len(df), True)
                for j in batch_effects:
                    idx = np.logical_and(
                        idx,
                        df[j].isin(batch_effects[j]),
                    )
                be_df = df[idx]
                scatter = sns.scatterplot(
                    data=be_df,
                    x=covariate,
                    y=response_var,
                    hue=hue_data if hue_data in df else None,
                    style=markers_data if markers_data in df else None,
                    s=50,
                    alpha=0.7,
                    zorder=1,
                )

                non_be_df = df[~idx]
                # Plot other data as small black points
                sns.scatterplot(
                    data=non_be_df,
                    x=covariate,
                    y=response_var,
                    color="black",
                    s=20,  # Smaller point size
                    alpha=0.4,
                    zorder=0,
                )

                legend = scatter.get_legend()
                if legend:
                    handles = legend.legend_handles
                    labels = [t.get_text() for t in legend.get_texts()]
                    plt.legend(
                        handles,
                        labels,
                        title=data.name + " data",
                        title_fontsize=10,
                    )
        # Set title and labels
        plt.title(f"Centiles of {response_var}")
        plt.xlabel(covariate)
        plt.ylabel(response_var)

        # Show the plot
        plt.show()

    def compute_zscores(self, data: NormData) -> NormData:
        """Computes zscores for each response variable in the data.

        Args:
            data (NormData): Data containing the covariates and response variables.

        Raises:
            ValueError: Error raised if the model for the response variable does not exist.

        Returns:
            NormData: NormData extended with the zscores.
        """

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
            if responsevar not in self.regression_models:
                raise ValueError(
                    f"Attempted to find zscores for model {responsevar}, but it does not exist."
                )

            # Set self.model to the current model
            self.prepare(responsevar)

            # Overwrite zscores
            print("Computing zscores for", responsevar)
            data["zscores"].loc[{"response_vars": responsevar}] = self._zscores(
                resp_predict_data
            )

            self.reset()

        self.postprocess(data)
        return data

    def preprocess(self, data: NormData) -> None:
        """Applies preprocessing to the data.

        Consists of scaling the data and expanding the basis of the covariates.

        Args:
            data (NormData): Data to preprocess.
        """
        self.scale_forward(data)

        # data.scale_forward(self.norm_conf.inscaler, self.norm_conf.outscaler)
        self.expand_basis(data, source_array_name="scaled_X")

    def expand_basis(
        self, data: NormData, source_array_name: str, intercept: bool = False
    ) -> None:
        """Expand the basis of the source array.

        Args:
            data (NormData): Data to expand the basis of.
            source_array_name (str): Name of the source array to expand the basis of.
            intercept (bool, optional): Whether to include a column of ones for modeling an intercept in a linear regression. Defaults to False.

        Raises:
            ValueError: Error if the source array does not exist.
        """

        # Expand the basis of the source array
        source_array: xr.DataArray = None  # type: ignore
        if source_array_name == "scaled_X":
            if "scaled_X" not in data.data_vars:
                raise ValueError(
                    "scaled_X does not exist. Please scale the data first using the scale_forward method."
                )
            source_array = data.scaled_X
        elif source_array_name == "X":
            source_array = data.X

        # TODO: Should this be like this?
        # Every basis expansion has a linear component, so we always include the original data
        all_arrays = [source_array.data]

        # Get the original named dimensions of the data
        all_dims = list(data.covariates.to_numpy())

        # Create a new array with the expanded basis
        basis_expansion = self.norm_conf.basis_function
        basis_column = self.norm_conf.basis_column
        if basis_expansion == "polynomial":
            # Expand the basis with polynomial basis functions
            expanded_basis = create_poly_basis(
                source_array.data[:, basis_column], self.norm_conf.order
            )
            # Add the expanded basis to the list of arrays
            all_arrays.append(expanded_basis)
            # Add the names of the new dimensions to the list of dimensions
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )
        elif basis_expansion == "bspline":
            # Expand the basis with bspline basis functions
            if not hasattr(self, "bspline_basis"):
                order = self.norm_conf.order
                nknots = self.norm_conf.nknots
                xmin = np.min(source_array.data[:, basis_column])
                xmax = np.max(source_array.data[:, basis_column])
                diff = xmax - xmin
                xmin = xmin - 0.2 * diff
                xmax = xmax + 0.2 * diff
                self.bspline_basis: splinelab.bspline.Bspline = create_bspline_basis(  # type: ignore
                    xmin=xmin,
                    xmax=xmax,
                    p=order,
                    nknots=nknots,
                )
            expanded_basis = np.array(
                [self.bspline_basis(c) for c in source_array.data[:, basis_column]]  # type: ignore
            )
            # Add the expanded basis to the list of arrays
            all_arrays.append(expanded_basis)
            # Add the names of the new dimensions to the list of dimensions
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )
        elif basis_expansion in ["none", "linear"]:
            # Do not expand the basis
            pass

        if intercept:
            all_dims.append("intercept")
            all_arrays.append(np.ones((expanded_basis.shape[0], 1)))

        Phi = np.concatenate(all_arrays, axis=1)
        data["Phi"] = xr.DataArray(
            Phi,
            coords={"basis_functions": all_dims},
            dims=["datapoints", "basis_functions"],
        )

    def postprocess(self, data: NormData) -> None:
        """Apply postprocessing to the data.

        Args:
            data (NormData): Data to postprocess.
        """
        self.scale_backward(data)

    def scale_forward(self, data: NormData, overwrite: bool = False) -> None:
        for covariate in data.covariates.to_numpy():
            if (covariate not in self.inscalers) or overwrite:
                self.inscalers[covariate] = scaler(self.norm_conf.inscaler)
                self.inscalers[covariate].fit(data.X.sel(covariates=covariate).data)

        for responsevar in data.response_vars.to_numpy():
            if (responsevar not in self.outscalers) or overwrite:
                self.outscalers[responsevar] = scaler(self.norm_conf.outscaler)
                self.outscalers[responsevar].fit(
                    data.y.sel(response_vars=responsevar).data
                )

        data.scale_forward(self.inscalers, self.outscalers)

    def scale_backward(self, data: NormData) -> None:
        """Scale the data back to the original scale.

        Args:
            data (NormData): Data to scale back.
        """
        data.scale_backward(self.inscalers, self.outscalers)

    def save(self) -> None:
        """Save the normative model to disk, at the location specified in the norm_conf."""
        # TODO pass a path to save the model, which takes precedence over the save_dir in the norm_conf

        # Save metadata and small components in JSON
        metadata = {
            "norm_conf": self.norm_conf.to_dict(),
            "response_vars": self.response_vars,
            "regression_model_type": self.regression_model_type.__name__,
            "default_reg_conf": self.default_reg_conf.to_dict(),
        }

        # Include bspline basis expansion if it exists
        if self.norm_conf.basis_function == "bspline" and hasattr(
            self, "bspline_basis"
        ):
            knots = self.bspline_basis.knot_vector
            metadata["bspline_basis"] = {
                "xmin": knots[0],
                "xmax": knots[-1],
                "nknots": self.norm_conf.nknots,
                "p": self.norm_conf.order,
            }

        # Save inscalers and outscalers
        metadata["inscalers"] = {k: v.to_dict() for k, v in self.inscalers.items()}
        metadata["outscalers"] = {k: v.to_dict() for k, v in self.outscalers.items()}

        with open(
            os.path.join(self.norm_conf.save_dir, "normative_model_dict.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=4)

        # Save regression models as JSON -> use the to_dict method of the regression model
        for responsevar, model in self.regression_models.items():
            model_dict = model.to_dict(self.norm_conf.save_dir)
            with open(
                os.path.join(self.norm_conf.save_dir, f"model_{responsevar}.json"),
                mode="w",
                encoding="utf-8",
            ) as f:
                json.dump(model_dict, f, indent=4)

    @classmethod
    def load(cls, path: str) -> NormBase:
        """Load a normative model from a specified path.

        Args:
            path (str): The path to the directory containing the normative model.

        Returns:
            NormBase: A normative model loaded from the specified path.
        """
        with open(
            os.path.join(path, "normative_model_dict.json"), mode="r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        self = cls(NormConf.from_dict(metadata["norm_conf"]))
        self.response_vars = metadata["response_vars"]
        self.regression_model_type = globals()[metadata["regression_model_type"]]

        # Load bspline basis if it exists
        if "bspline_basis" in metadata:
            self.bspline_basis = create_bspline_basis(**metadata["bspline_basis"])

        # Load inscalers and outscalers
        self.inscalers = {
            k: scaler.from_dict(v) for k, v in metadata["inscalers"].items()
        }
        self.outscalers = {
            k: scaler.from_dict(v) for k, v in metadata["outscalers"].items()
        }

        # Load regression models
        self.regression_models = {}
        for responsevar in self.response_vars:
            model_path = os.path.join(path, f"model_{responsevar}.json")
            with open(model_path, mode="r", encoding="utf-8") as f:
                model_dict = json.load(f)
            self.regression_models[responsevar] = self.regression_model_type.from_dict(
                model_dict, path
            )

        self.default_reg_conf = type(
            self.regression_models[responsevar].reg_conf
        ).from_dict(metadata["default_reg_conf"])
        return self

    def set_save_dir(self, save_dir: str) -> None:
        """Override the save_dir in the norm_conf.

        Args:
            save_dir (str): New save directory.
        """
        self.norm_conf.set_save_dir(save_dir)

    def set_log_dir(self, log_dir: str) -> None:
        """Override the log_dir in the norm_conf.

        Args:
            log_dir (str): New log directory.
        """
        self.norm_conf.set_log_dir(log_dir)

    def prepare(self, responsevar: str) -> None:
        """Prepare the model for a specific response variable.
        Sets the current_regression_model attribute and the current_response_var attribute.
        Creates a new regression model if it does not exist yet.

        Args:
            responsevar (str): The response variable to prepare the model for.
        """
        self.current_response_var = responsevar
        # Create a new model if it does not exist yet
        if responsevar not in self.regression_models:
            self.regression_models[responsevar] = self.regression_model_type(
                responsevar, self.get_reg_conf(responsevar)
            )
        self.current_regression_model = self.regression_models.get(responsevar, None)  # type: ignore

    def get_reg_conf(self, responsevar: str) -> RegConf:
        """Get the regression configuration for a specific response variable.

        Args:
            responsevar (str): The response variable to get the regression configuration for.

        Returns:
            RegConf: The regression configuration for the response variable.
        """
        if responsevar in self.regression_models:
            return self.regression_models[responsevar].reg_conf
        else:
            return self.default_reg_conf

    def reset(self) -> None:
        """Does nothing. Can be overridden by subclasses."""
        pass

    def __item__(self, key: str) -> RegressionModel:
        return self.regression_models[key]

    #######################################################################################################

    # Abstract methods

    #######################################################################################################

    @abstractmethod
    def _fit(self, data: NormData) -> NormData:
        """
        Acts as the adapter for fitting the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """

    @abstractmethod
    def _predict(self, data: NormData) -> NormData:
        """
        Acts as the adapter for prediction using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """

    @abstractmethod
    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Acts as the adapter for fit_predict using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """

    @abstractmethod
    def _transfer(self, data: NormData, **kwargs: Any) -> RegressionModel:
        """Transfers the current regression model to new data. Returns a new regression model.

        Args:
            data (NormData): Data to transfer the model to.

        Returns:
            RegressionModel: A new regression model transferred to the new data.
        """

    @abstractmethod
    def _extend(self, data: NormData) -> NormBase:
        """Extends the current regression model with new data. Returns a new normative model.

        Args:
            data (NormData): _description_

        Returns:
            NormBase: _description_
        """

    @abstractmethod
    def _tune(self, data: NormData) -> NormBase:
        pass

    @abstractmethod
    def _merge(self, other: NormBase) -> NormBase:
        pass

    @abstractmethod
    def _centiles(self, data: NormData, centiles: np.ndarray) -> xr.DataArray:
        """Takes a list of cummulative densities and returns the corresponding centiles of the model.
        The return type is an xr.datarray with dimensions (cdf, datapoints).
        """

    @abstractmethod
    def _zscores(self, data: NormData) -> xr.DataArray:
        """Returns the zscores of the model.
        The return type is an xr.datarray with dimensions (datapoints)."""

    @abstractmethod
    def n_params(self) -> int:
        """
        Returns the number of parameters of the model.
        """

    @classmethod
    @abstractmethod
    def from_args(cls, args: dict) -> NormBase:
        """
        Creates a normative model from command line arguments.
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
        return self.current_regression_model.has_random_effect
