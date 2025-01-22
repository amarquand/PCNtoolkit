"""
Hierarchical Bayesian Regression model implementation.

This class implements a Bayesian hierarchical regression model using PyMC for
posterior sampling. It supports multiple likelihood functions and provides
methods for model fitting, prediction, and analysis.

Parameters
----------
name : str
    Unique identifier for the model instance
reg_conf : HBRConf
    Configuration object containing model hyperparameters and structure
is_fitted : bool, optional
    Flag indicating if the model has been fitted, by default False
is_from_dict : bool, optional
    Flag indicating if model was created from dictionary, by default False

Attributes
----------
idata : arviz.InferenceData
    Contains the MCMC samples and model inference data
pymc_model : pm.Model
    PyMC model object containing the computational graph
reg_conf : HBRConf
    Model configuration object
is_fitted : bool
    Indicates if model has been fitted
name : str
    Model identifier

Methods
-------
fit(data: HBRData, idata: Optional[az.InferenceData] = None, freedom: float = 1)
    Fit the model to training data
predict(data: HBRData)
    Generate predictions for new data
centiles(data: HBRData, cdf: np.ndarray, resample: bool = True)
    Calculate centile values for observations
zscores(data: HBRData, resample: bool = False)
    Calculate z-scores for observations
compile_model(data: HBRData, idata: Optional[az.InferenceData] = None, freedom: float = 1)
    Create the PyMC model computational graph
to_dict(path: Optional[str] = None)
    Serialize model to dictionary format
from_dict(dct: Dict[str, Any], path: Optional[str] = None)
    Create model instance from serialized dictionary
from_args(name: str, args: Dict[str, Any])
    Create model instance from command line arguments

Notes
-----
The model supports Normal, SHASHb and SHASHo likelihood functions. The model structure
is defined through the HBRConf configuration object which specifies the parameters
(mu, sigma, epsilon, delta) and their hierarchical relationships.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import arviz as az  # type: ignore
import numpy as np
import pymc as pm  # type: ignore
import scipy.stats as stats  # type: ignore
import xarray as xr

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.hbr_util import centile, zscore
from pcntoolkit.regression_model.hbr.prior import Prior
from pcntoolkit.regression_model.hbr.shash import SHASHb, SHASHo
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.output import Errors, Output

from .hbr_conf import HBRConf


class HBR(RegressionModel):
    """
    Hierarchical Bayesian Regression model implementation.

    This class implements a Bayesian hierarchical regression model using PyMC for
    posterior sampling. It supports multiple likelihood functions and provides
    methods for model fitting, prediction, and analysis.

    Parameters
    ----------
    name : str
        Unique identifier for the model instance
    reg_conf : HBRConf
        Configuration object containing model hyperparameters and structure
    is_fitted : bool, optional
        Flag indicating if the model has been fitted, by default False
    is_from_dict : bool, optional
        Flag indicating if model was created from dictionary, by default False

    Attributes
    ----------
    idata : arviz.InferenceData
        Contains the MCMC samples and model inference data
    pymc_model : pm.Model
        PyMC model object containing the computational graph
    reg_conf : HBRConf
        Model configuration object
    is_fitted : bool
        Indicates if model has been fitted

    Methods
    -------
    fit(hbrdata, make_new_model=True)
        Fit the model to training data using MCMC sampling
    predict(hbrdata)
        Generate predictions for new data
    fit_predict(fit_hbrdata, predict_hbrdata)
        Fit model and generate predictions in one step
    transfer(hbrconf, transferdata, freedom)
        Perform transfer learning using existing model as prior
    centiles(hbrdata, cdf, resample=True)
        Calculate centile values for given cumulative densities
    zscores(hbrdata, resample=False)
        Calculate z-scores for observations
    """

    def __init__(
        self,
        name: str,
        reg_conf: HBRConf,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        super().__init__(name, reg_conf, is_fitted, is_from_dict)
        self.idata: az.InferenceData = None  # type: ignore
        self.pymc_model: pm.Model = None  # type: ignore

    def fit(self, hbrdata: HBRData, make_new_model: bool = True, progressbar: bool = True) -> None:
        """
        Fit the model to training data using MCMC sampling.

        Parameters
        ----------
        hbrdata : HBRData
            Training data object containing features and targets
        make_new_model : bool, optional
            Whether to create a new PyMC model, by default True

        Returns
        -------
        None
        """
        if make_new_model or (not self.pymc_model):
            self.compile_model(hbrdata)
        with self.pymc_model:
            self.idata = pm.sample(
                self.draws,
                tune=self.tune,
                cores=self.pymc_cores,
                chains=self.chains,
                nuts_sampler=self.nuts_sampler,  # type: ignore
                init=self.init,
                progressbar=progressbar,
            )
        self.is_fitted = True

    def predict(
        self,
        hbrdata: HBRData,
        extend_inferencedata: bool = True,
        progressbar: bool = True,
    ) -> az.InferenceData | dict[str, np.ndarray[Any, Any]]:
        """
        Generate predictions for new data.

        Parameters
        ----------
        hbrdata : HBRData
            Data object containing features to predict on

        Returns
        -------
        None
            Updates the model's inference data with predictions
        """
        if not self.pymc_model:
            self.compile_model(hbrdata)
        else:
            hbrdata.set_data_in_existing_model(self.pymc_model)  # Model already compiled, only need to update the data
        if extend_inferencedata and hasattr(self.idata, "predictions"):
            del self.idata.predictions
            del self.idata.predictions_constant_data
        with self.pymc_model:
            idata = pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=extend_inferencedata,
                var_names=self.get_var_names() + ["y_pred"],
                predictions=True,
                progressbar=progressbar,
            )
        return idata

    def fit_predict(self, fit_hbrdata: HBRData, predict_hbrdata: HBRData) -> None:
        """
        Fit model and generate predictions in one step.

        Parameters
        ----------
        fit_hbrdata : HBRData
            Training data for model fitting
        predict_hbrdata : HBRData
            Data to generate predictions for

        Returns
        -------
        None
            Updates model's inference data with fitted parameters and predictions
        """
        if not self.pymc_model:
            self.compile_model(fit_hbrdata)
        with self.pymc_model:
            self.idata = pm.sample(
                self.draws,
                tune=self.tune,
                cores=self.pymc_cores,
                chains=self.chains,
                nuts_sampler=self.nuts_sampler,  # type: ignore
                init=self.init,
            )
        self.is_fitted = True
        predict_hbrdata.set_data_in_existing_model(self.pymc_model)
        with self.pymc_model:
            pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names() + ["y_pred"],
                predictions=True,
            )

    def transfer(
        self,
        hbrconf: HBRConf,
        transferdata: HBRData,
        freedom: float,
        progressbar: bool = True,
    ) -> HBR:
        """
        Perform transfer learning using existing model as prior.

        Parameters
        ----------
        hbrconf : HBRConf
            Configuration for new model
        transferdata : HBRData
            Data for transfer learning
        freedom : float
            Parameter controlling influence of prior model (0-1)

        Returns
        -------
        HBR
            New model instance with transferred knowledge
        """
        new_hbr_model = HBR(self.name, hbrconf)
        new_hbr_model.compile_model(transferdata, self.idata, freedom)
        with new_hbr_model.pymc_model:
            new_hbr_model.idata = pm.sample(
                hbrconf.draws,
                tune=hbrconf.tune,
                cores=hbrconf.pymc_cores,
                chains=hbrconf.chains,
                nuts_sampler=hbrconf.nuts_sampler,  # type: ignore
                progressbar=progressbar,
            )
            new_hbr_model.is_fitted = True

        return new_hbr_model

    def generate_synthetic_data(self, hbrdata: HBRData, progressbar: bool = True) -> HBRData:
        if not self.pymc_model:
            self.compile_model(hbrdata)
        else:
            hbrdata.set_data_in_existing_model(self.pymc_model)

        with self.pymc_model:
            pred_idata = pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=False,
                var_names=["y_pred"],
                predictions=True,
                progressbar=progressbar,
            )
        preds = az.extract(pred_idata, "predictions", var_names=["y_pred"])
        datapoints, sample = preds.shape
        replace = datapoints > sample
        selected_idx = np.random.choice(sample, size=datapoints, replace=replace)
        hbrdata.y = np.diag(preds.values[:, selected_idx])
        return hbrdata

    def centiles(self, hbrdata: HBRData, cdf: np.ndarray) -> xr.DataArray:
        """
        Calculate centile values for given cumulative densities.

        Parameters
        ----------
        hbrdata : HBRData
            Data to calculate centiles for
        cdf : np.ndarray
            Array of cumulative density values

        Returns
        -------
        xr.DataArray
            Calculated centile values
        """
        centiles_idata = self.predict(hbrdata, extend_inferencedata=False, progressbar=False)
        var_names = self.get_var_names()
        post_pred = az.extract(
            centiles_idata,
            "predictions",
            var_names=var_names,
        )
        array_of_vars = [self.likelihood] + list(map(lambda x: np.squeeze(post_pred[x]), var_names))
        n_datapoints, n_mcmc_samples = post_pred[var_names[0]].shape
        centiles = np.zeros((cdf.shape[0], n_datapoints, n_mcmc_samples))
        for i, _cdf in enumerate(cdf):
            zs = np.full((n_datapoints, n_mcmc_samples), stats.norm.ppf(_cdf), dtype=float)
            centiles[i] = xr.apply_ufunc(centile, *array_of_vars, kwargs={"zs": zs})
        return xr.DataArray(
            centiles,
            dims=["cdf", "datapoints", "sample"],
            coords={"cdf": cdf},
        ).mean(dim="sample")

    def zscores(self, hbrdata: HBRData) -> xr.DataArray:
        """
        Calculate z-scores for observations.

        Parameters
        ----------
        hbrdata : HBRData
            Data containing observations to calculate z-scores for

        Returns
        -------
        xr.DataArray
            Calculated z-scores
        """
        zscores_idata = self.predict(hbrdata, extend_inferencedata=False, progressbar=False)
        var_names = self.get_var_names()
        post_pred = az.extract(
            zscores_idata,
            "predictions",
            var_names=var_names,
        )
        array_of_vars = [self.likelihood] + list(map(lambda x: np.squeeze(post_pred[x]), var_names))

        zscores = xr.apply_ufunc(zscore, *array_of_vars, kwargs={"y": hbrdata.y[:, None]}).mean(dim="sample")

        return zscores

    def logp(self, hbrdata: HBRData) -> xr.DataArray:
        """
        Compute log-probabilities for each observation in the data.
        """
        if not self.pymc_model:
            self.compile_model(hbrdata)
        with self.pymc_model:
            logp = pm.compute_log_likelihood(
                self.idata,
                var_names=["y_pred"],
                extend_inferencedata=False,
                progressbar=False,
            )
        return az.extract(logp, "log_likelihood", var_names=["y_pred"]).mean("sample")

    def get_var_names(self) -> List[str]:
        """Get the variable names for the current likelihood function.

        Returns
        -------
        List[str]
            List of variable names required for the current likelihood function.
            For Normal likelihood: ['mu_samples', 'sigma_samples']
            For SHASH likelihoods: ['mu_samples', 'sigma_samples', 'epsilon_samples', 'delta_samples']

        Raises
        ------
        RuntimeError
            If likelihood is not supported (must be 'Normal', 'SHASHb', or 'SHASHo')
        """
        likelihood = self.likelihood
        if likelihood == "Normal":
            var_names = ["mu_samples", "sigma_samples"]
        elif likelihood.startswith("SHASH"):
            var_names = [
                "mu_samples",
                "sigma_samples",
                "epsilon_samples",
                "delta_samples",
            ]
        elif likelihood == "beta":
            var_names = ["alpha_samples", "beta_samples"]
        else:
            raise Output.error(Errors.ERROR_UNKNOWN_LIKELIHOOD, likelihood=likelihood)
        return var_names

    def compile_model(
        self,
        data: HBRData,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ) -> None:
        """
        Create the PyMC model computational graph.

        Parameters
        ----------
        data : HBRData
            Data object containing model inputs
        idata : Optional[az.InferenceData], optional
            Previous inference data for transfer learning, by default None
        freedom : float, optional
            Parameter controlling influence of prior model, by default 1

        Returns
        -------
        None
            Creates and stores PyMC model in instance
        """
        self.pymc_model = pm.Model(coords=data.coords)
        data.set_data_in_new_model(self.pymc_model)
        if self.likelihood == "Normal":
            self.compile_normal(data, idata, freedom)
        elif self.likelihood == "SHASHb":
            self.compile_SHASHb(data, idata, freedom)
        elif self.likelihood == "SHASHo":
            self.compile_SHASHo(data, idata, freedom)
        elif self.likelihood == "beta":
            self.compile_beta(data, idata, freedom)
        else:
            raise Output.error(Errors.ERROR_UNKNOWN_LIKELIHOOD, likelihood=self.likelihood)

    def compile_normal(
        self,
        data: HBRData,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ) -> None:
        """
        Create PyMC model with Normal likelihood.

        Parameters
        ----------
        data : HBRData
            Data object containing model inputs
        idata : Optional[az.InferenceData], optional
            Previous inference data for transfer learning, by default None
        freedom : float, optional
            Parameter controlling influence of prior model (0-1), by default 1

        Returns
        -------
        None
            Updates the PyMC model in place with Normal likelihood components
        """
        self.mu.compile(self.pymc_model, idata, freedom)
        self.sigma.compile(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.mu.sample(data),
                dims=self.mu.sample_dims,
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.sigma.sample(data),
                dims=self.sigma.sample_dims,
            )
            pm.Normal(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def compile_SHASHb(
        self,
        data: HBRData,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ) -> None:
        """
        Create PyMC model with SHASHb likelihood.

        Parameters
        ----------
        data : HBRData
            Data object containing model inputs
        idata : Optional[az.InferenceData], optional
            Previous inference data for transfer learning, by default None
        freedom : float, optional
            Parameter controlling influence of prior model (0-1), by default 1

        Returns
        -------
        None
            Updates the PyMC model in place with SHASHb likelihood components
        """
        self.mu.compile(self.pymc_model, idata, freedom)
        self.sigma.compile(self.pymc_model, idata, freedom)
        self.epsilon.compile(self.pymc_model, idata, freedom)
        self.delta.compile(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.mu.sample(data),
                dims=self.mu.sample_dims,
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.sigma.sample(data),
                dims=self.sigma.sample_dims,
            )
            epsilon_samples = pm.Deterministic(
                "epsilon_samples",
                self.epsilon.sample(data),
                dims=self.epsilon.sample_dims,
            )
            delta_samples = pm.Deterministic(
                "delta_samples",
                self.delta.sample(data),
                dims=self.delta.sample_dims,
            )
            SHASHb(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def compile_SHASHo(
        self,
        data: HBRData,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ) -> None:
        """
        Create PyMC model with SHASHo likelihood.

        Parameters
        ----------
        data : HBRData
            Data object containing model inputs
        idata : Optional[az.InferenceData], optional
            Previous inference data for transfer learning, by default None
        freedom : float, optional
            Parameter controlling influence of prior model (0-1), by default 1

        Returns
        -------
        None
            Updates the PyMC model in place with SHASHo likelihood components
        """
        self.mu.compile(self.pymc_model, idata, freedom)
        self.sigma.compile(self.pymc_model, idata, freedom)
        self.epsilon.compile(self.pymc_model, idata, freedom)
        self.delta.compile(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.mu.sample(data),
                dims=self.mu.sample_dims,
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.sigma.sample(data),
                dims=self.sigma.sample_dims,
            )
            epsilon_samples = pm.Deterministic(
                "epsilon_samples",
                self.epsilon.sample(data),
                dims=self.epsilon.sample_dims,
            )
            delta_samples = pm.Deterministic(
                "delta_samples",
                self.delta.sample(data),
                dims=self.delta.sample_dims,
            )
            SHASHo(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def compile_beta(
        self,
        data: HBRData,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ) -> None:
        self.alpha.compile(self.pymc_model, idata, freedom)
        self.beta.compile(self.pymc_model, idata, freedom)
        with self.pymc_model:
            alpha_samples = pm.Deterministic(
                "alpha_samples",
                self.alpha.sample(data),
                dims=self.alpha.sample_dims,
            )
            beta_samples = pm.Deterministic(
                "beta_samples",
                self.beta.sample(data),
                dims=self.beta.sample_dims,
            )
            pm.Beta(
                "y_pred",
                alpha=alpha_samples,
                beta=beta_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def to_dict(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize model to dictionary format.

        Parameters
        ----------
        path : Optional[str], optional
            Path to save inference data, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing serialized model
        """
        my_dict = super().to_dict()
        if self.is_fitted and (path is not None):
            idata_path = os.path.join(path, "idata.nc")
            self.save_idata(idata_path)
        return my_dict

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any], path: Optional[str] = None) -> "HBR":
        """
        Create model instance from serialized dictionary.

        Parameters
        ----------
        dict : Dict[str, Any]
            Dictionary containing serialized model
        path : Optional[str], optional
            Path to load inference data from, by default None

        Returns
        -------
        HBR
            New model instance
        """
        name = my_dict["name"]
        conf = HBRConf.from_dict(my_dict["reg_conf"])
        is_fitted = my_dict["is_fitted"]
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        if is_fitted and (path is not None):
            idata_path = os.path.join(path, "idata.nc")
            self.load_idata(idata_path)
        return self

    @classmethod
    def from_args(cls, name: str, args: Dict[str, Any]) -> "HBR":
        """
        Create model instance from command line arguments.

        Parameters
        ----------
        name : str
            Name for new model instance
        args : Dict[str, Any]
            Dictionary of command line arguments

        Returns
        -------
        HBR
            New model instance
        """
        conf = HBRConf.from_args(args)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        return self

    def save_idata(self, path: str) -> None:
        """
        Save inference data to NetCDF file.

        Parameters
        ----------
        path : str
            Path to save inference data to. Should end in '.nc'

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If model is fitted but does not have inference data
        """
        if self.is_fitted:
            if hasattr(self, "idata"):
                # self.remove_samples_from_idata_posterior()
                self.idata.to_netcdf(path, groups=["posterior"])
            else:
                raise Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_IDATA)

    def load_idata(self, path: str) -> None:
        """
        Load inference data from NetCDF file.

        Parameters
        ----------
        path : str
            Path to load inference data from. Should end in '.nc'

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If model is fitted but inference data cannot be loaded from path
        """
        if self.is_fitted:
            try:
                self.idata = az.from_netcdf(path)
            except Exception as exc:
                raise Output.error(Errors.ERROR_HBR_COULD_NOT_LOAD_IDATA, path=path) from exc
            # self.replace_samples_in_idata_posterior()

    def remove_samples_from_idata_posterior(self) -> None:
        """
        Remove sample variables from the posterior group of inference data.

        This method removes variables ending with '_samples' from the posterior group
        before saving to avoid privacy issues. The variables can be recomputed from the
        model parameters. The names of removed variables are stored in idata.attrs['removed_samples'].

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This is used internally before saving the inference data to disk to reduce
        storage size, since sample variables can be recomputed from the model parameters.
        """
        post: xr.Dataset = self.idata.posterior  # type: ignore
        for name in post.variables.mapping.keys():
            if str(name).endswith("_samples"):
                post.drop_vars(str(name))
                if "removed_samples" not in self.idata.attrs:
                    self.idata.attrs["removed_samples"] = []
                self.idata.attrs["removed_samples"].append(name)

    def replace_samples_in_idata_posterior(self) -> None:
        """
        Replace previously removed sample variables in the posterior group.

        This method adds back placeholder arrays for variables that were removed by
        remove_samples_from_idata_posterior(). The arrays are initialized with zeros
        and will be populated when the model is used.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This is used internally after loading inference data from disk to restore
        the structure needed for model predictions. The actual values will be
        recomputed when needed.
        """
        post: xr.Dataset = self.idata.posterior  # type: ignore
        for name in self.idata.attrs["removed_samples"]:
            post[name] = xr.DataArray(
                np.zeros(post[name].shape),
                dims=post[name].dims,
            )

    # pylint: disable=C0116

    @property
    def mu(self) -> Prior:
        return self.reg_conf.mu  # type: ignore

    @property
    def sigma(self) -> Prior:
        return self.reg_conf.sigma  # type: ignore

    @property
    def epsilon(self) -> Prior:
        return self.reg_conf.epsilon  # type: ignore

    @property
    def delta(self) -> Prior:
        return self.reg_conf.delta  # type: ignore

    @property
    def alpha(self) -> Prior:
        return self.reg_conf.alpha  # type: ignore

    @property
    def beta(self) -> Prior:
        return self.reg_conf.beta  # type: ignore

    @property
    def likelihood(self) -> str:
        return self.reg_conf.likelihood  # type: ignore

    @property
    def draws(self) -> int:
        return self.reg_conf.draws  # type: ignore

    @property
    def tune(self) -> int:
        return self.reg_conf.tune  # type: ignore

    @property
    def pymc_cores(self) -> int:
        return self.reg_conf.pymc_cores  # type: ignore

    @property
    def chains(self) -> int:
        return self.reg_conf.chains  # type: ignore

    @property
    def nuts_sampler(self) -> str:
        return self.reg_conf.nuts_sampler  # type: ignore

    @property
    def init(self) -> str:
        return self.reg_conf.init  # type: ignore

    @property
    def reg_conf(self) -> HBRConf:
        return self._reg_conf  # type: ignore
