#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:23:15 2019

@author: seykia
@author: augub
"""

from __future__ import print_function
from __future__ import division
from collections import OrderedDict

from ast import Param
from tkinter.font import names

import numpy as np
import pymc as pm
import pytensor
import arviz as az

from itertools import product
from functools import reduce

from pymc import Metropolis, NUTS, Slice, HamiltonianMC
from scipy import stats
import bspline
from bspline import splinelab

from util.utils import create_poly_basis
from util.utils import expand_all
from pcntoolkit.util.utils import cartesian_product
from pcntoolkit.model.SHASH import *


def bspline_fit(X, order, nknots):
    feature_num = X.shape[1]
    bsp_basis = []

    for i in range(feature_num):
        minx = np.min(X[:, i])
        maxx = np.max(X[:, i])
        delta = maxx - minx
        # Expand range by 20% (10% on both sides)
        splinemin = minx - 0.1 * delta
        splinemax = maxx + 0.1 * delta
        knots = np.linspace(splinemin, splinemax, nknots)
        k = splinelab.augknt(knots, order)
        bsp_basis.append(bspline.Bspline(k, order))

    return bsp_basis


def bspline_transform(X, bsp_basis):
    if type(bsp_basis) != list:
        temp = []
        temp.append(bsp_basis)
        bsp_basis = temp

    feature_num = len(bsp_basis)
    X_transformed = []
    for f in range(feature_num):
        X_transformed.append(np.array([bsp_basis[f](i) for i in X[:, f]]))
    X_transformed = np.concatenate(X_transformed, axis=1)

    return X_transformed


def create_poly_basis(X, order):
    """compute a polynomial basis expansion of the specified order"""

    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
    Phi = np.zeros((X.shape[0], D * order))
    colid = np.arange(0, D)
    for d in range(1, order + 1):
        Phi[:, colid] = X**d
        colid += D
    return Phi


def from_posterior(param, samples, distribution=None, half=False, freedom=1):
    if len(samples.shape) > 1:
        shape = samples.shape[1:]
    else:
        shape = None

    if distribution is None:
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 1000)
        y = stats.gaussian_kde(np.ravel(samples))(x)
        if half:
            x = np.concatenate([x, [x[-1] + 0.1 * width]])
            y = np.concatenate([y, [0]])
        else:
            x = np.concatenate([[x[0] - 0.1 * width], x, [x[-1] + 0.1 * width]])
            y = np.concatenate([[0], y, [0]])
        if shape is None:
            return pm.distributions.Interpolated(param, x, y)
        else:
            return pm.distributions.Interpolated(param, x, y, shape=shape)
    elif distribution == "normal":
        temp = stats.norm.fit(samples)
        if shape is None:
            return pm.Normal(param, mu=temp[0], sigma=freedom * temp[1])
        else:
            return pm.Normal(param, mu=temp[0], sigma=freedom * temp[1], shape=shape)
    elif distribution == "hnormal":
        temp = stats.halfnorm.fit(samples)
        if shape is None:
            return pm.HalfNormal(param, sigma=freedom * temp[1])
        else:
            return pm.HalfNormal(param, sigma=freedom * temp[1], shape=shape)
    elif distribution == "hcauchy":
        temp = stats.halfcauchy.fit(samples)
        if shape is None:
            return pm.HalfCauchy(param, freedom * temp[1])
        else:
            return pm.HalfCauchy(param, freedom * temp[1], shape=shape)
    elif distribution == "uniform":
        upper_bound = np.percentile(samples, 95)
        lower_bound = np.percentile(samples, 5)
        r = np.abs(upper_bound - lower_bound)
        if shape is None:
            return pm.Uniform(
                param, lower=lower_bound - freedom * r, upper=upper_bound + freedom * r
            )
        else:
            return pm.Uniform(
                param,
                lower=lower_bound - freedom * r,
                upper=upper_bound + freedom * r,
                shape=shape,
            )
    elif distribution == "huniform":
        upper_bound = np.percentile(samples, 95)
        lower_bound = np.percentile(samples, 5)
        r = np.abs(upper_bound - lower_bound)
        if shape is None:
            return pm.Uniform(param, lower=0, upper=upper_bound + freedom * r)
        else:
            return pm.Uniform(
                param, lower=0, upper=upper_bound + freedom * r, shape=shape
            )

    elif distribution == "gamma":
        alpha_fit, loc_fit, invbeta_fit = stats.gamma.fit(samples)
        if shape is None:
            return pm.Gamma(
                param, alpha=freedom * alpha_fit, beta=freedom / invbeta_fit
            )
        else:
            return pm.Gamma(
                param,
                alpha=freedom * alpha_fit,
                beta=freedom / invbeta_fit,
                shape=shape,
            )

    elif distribution == "igamma":
        alpha_fit, loc_fit, beta_fit = stats.gamma.fit(samples)
        if shape is None:
            return pm.InverseGamma(
                param, alpha=freedom * alpha_fit, beta=freedom * beta_fit
            )
        else:
            return pm.InverseGamma(
                param, alpha=freedom * alpha_fit, beta=freedom * beta_fit, shape=shape
            )


def hbr(X, y, batch_effects, configs, idata=None):
    """
    :param X: [N×P] array of clinical covariates
    :param y: [N×1] array of neuroimaging measures
    :param batch_effects: [N×M] array of batch effects
    :param batch_effects_size: [b1, b2,...,bM] List of counts of unique values of batch effects
    :param configs:
    :param idata:
    :param return_shared_variables: If true, returns references to the shared variables. The values of the shared variables can be set manually, allowing running the same model on different data without re-compiling it.
    :return:
    """

    # Make a param builder that will make the correct calls
    pb = ParamBuilder(X, y, batch_effects, idata, configs)

    def get_sample_dims(var):
        if configs[f'random_{var}']:
            return 'datapoints'
        elif configs[f'random_slope_{var}']:
            return 'datapoints'
        elif configs[f'random_intercept_{var}']:
            return 'datapoints'
        elif configs[f'linear_{var}']:
            return 'datapoints'
        return None

    with pm.Model(coords=pb.coords) as model:
        model.add_coord("datapoints", np.arange(X.shape[0]), mutable=True)
        X = pm.MutableData("X", X, dims=("datapoints", "basis_functions"))
        pb.X = X
        y = pm.MutableData("y", np.squeeze(y), dims="datapoints")
        pb.model = model
        pb.batch_effect_indices = tuple(
            [
                pm.Data(
                    pb.batch_effect_dim_names[i],
                    pb.batch_effect_indices[i],
                    mutable=True,
                    dims="datapoints",
                )
                for i in range(len(pb.batch_effect_indices))
            ]
        )

        if configs["likelihood"] == "Normal":
            
            mu = pm.Deterministic(
                "mu_samples",
                pb.make_param(
                    "mu",
                    mu_slope_mu_params=(0.0, 1.0),
                    sigma_slope_mu_params=(1.0,),
                    mu_intercept_mu_params=(0.0, 1.0),
                    sigma_intercept_mu_params=(1.0,),
                ).get_samples(pb),
                dims=get_sample_dims('mu'),
            )
            sigma = pm.Deterministic(
                "sigma_samples",
                pb.make_param(
                    "sigma", mu_sigma_params=(0.0, 2.0), sigma_sigma_params=(5.0,)
                ).get_samples(pb),
                dims=get_sample_dims('sigma'),
            )
            sigma_plus = pm.Deterministic(
                "sigma_plus", pm.math.log(1 + pm.math.exp(sigma)), dims=get_sample_dims('sigma')
            )
            y_like = pm.Normal(
                "y_like", mu, sigma=sigma_plus, observed=y, dims="datapoints"
            )

        elif configs["likelihood"] in ["SHASHb", "SHASHo", "SHASHo2"]:
            """
            Comment 1
            The current parameterizations are tuned towards standardized in- and output data.
            It is possible to adjust the priors through the XXX_dist and XXX_params kwargs, like here we do with epsilon_params.
            Supported distributions are listed in the Prior class.
            Comment 2
            Any mapping that is applied here after sampling should also be applied in util.hbr_utils.forward in order for the functions there to properly work.
            For example, the softplus applied to sigma here is also applied in util.hbr_utils.forward
            """
            SHASH_map = {"SHASHb": SHASHb, "SHASHo": SHASHo, "SHASHo2": SHASHo2}

            mu = pm.Deterministic(
                "mu_samples",
                pb.make_param(
                    "mu",
                    slope_mu_params=(0.0, 2.0),
                    mu_slope_mu_params=(0.0, 2.0),
                    sigma_slope_mu_params=(2.0,),
                    mu_intercept_mu_params=(0.0, 2.0),
                    sigma_intercept_mu_params=(2.0,),
                ).get_samples(pb),
                dims=get_sample_dims('mu'),
            )
            sigma = pm.Deterministic(
                "sigma_samples",
                pb.make_param(
                    "sigma",
                    sigma_params=(1.0, 1.0),
                    sigma_dist="normal",
                    slope_sigma_params=(0.0, 1.0),
                    intercept_sigma_params=(1.0, 1.0),
                ).get_samples(pb),
                dims=get_sample_dims('sigma'),
            )
            sigma_plus = pm.Deterministic(
                "sigma_plus_samples", np.log(1 + np.exp(sigma)), dims=get_sample_dims('sigma')
            )
            epsilon = pm.Deterministic(
                "epsilon_samples",
                pb.make_param(
                    "epsilon",
                    epsilon_params=(0.0, 1.0),
                    slope_epsilon_params=(0.0, 1.0),
                    intercept_epsilon_params=(0.0, 1.0),
                ).get_samples(pb),
                dims=get_sample_dims('epsilon'),
            )
            delta = pm.Deterministic(
                "delta_samples",
                pb.make_param(
                    "delta",
                    delta_params=(1.0, 1.0),
                    delta_dist="normal",
                    slope_epsilon_params=(0.0, 1.0),
                    intercept_epsilon_params=(1.0, 1.0),
                ).get_samples(pb),
                dims=get_sample_dims('delta'),
            )
            delta_plus = pm.Deterministic(
                "delta_plus_samples",
                np.log(1 + np.exp(delta * 10)) / 10 + 0.3,
                dims=get_sample_dims('delta'),
            )
            y_like = SHASH_map[configs["likelihood"]](
                "y_like",
                mu=mu,
                sigma=sigma_plus,
                epsilon=epsilon,
                delta=delta_plus,
                observed=y,
                dims="datapoints",
            )
        return model

  

class HBR:

    """Hierarchical Bayesian Regression for normative modeling

    Basic usage::

        model = HBR(configs)
        idata = model.estimate(X, y, batch_effects)
        ys,s2 = model.predict(X, batch_effects)

    where the variables are

    :param configs: a dictionary of model configurations.
    :param X: N-by-P input matrix of P features for N subjects
    :param y: N-by-1 vector of outputs.
    :param batch_effects: N-by-B matrix of B batch ids for N subjects.

    :returns: * ys - predictive mean
              * s2 - predictive variance

    Written by S.M. Kia
    """

    def __init__(self, configs):
        self.bsp = None
        self.model_type = configs["type"]
        self.configs = configs

    def get_modeler(self):
        return hbr

    def transform_X(self, X):
        if self.model_type == "polynomial":
            Phi = create_poly_basis(X, self.configs["order"])
        elif self.model_type == "bspline":
            if self.bsp is None:
                self.bsp = bspline_fit(X, self.configs["order"], self.configs["nknots"])
            bspline = bspline_transform(X, self.bsp)
            Phi = np.concatenate((X, bspline), axis=1)
        else:
            Phi = X
        return Phi

    def find_map(self, X, y, batch_effects, method="L-BFGS-B"):
        """Function to estimate the model"""
        X, y, batch_effects = expand_all(X, y, batch_effects)
        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs) as m:
            self.MAP = pm.find_MAP(method=method)
        return self.MAP

    def estimate(self, X, y, batch_effects):
        """Function to estimate the model"""
        X, y, batch_effects = expand_all(X, y, batch_effects)
        X = self.transform_X(X)
        modeler = self.get_modeler()
        if self.idata:
            del self.idata
        with modeler(X, y, batch_effects, self.configs) as m:
            self.idata = pm.sample(
                draws=self.configs["n_samples"],
                tune=self.configs["n_tuning"],
                chains=self.configs["n_chains"],
                init=self.configs["init"],
                n_init=500000,
                cores=self.configs["cores"],
            )
        return self.idata

    def predict(
        self, X, batch_effects, batch_effects_maps, pred="single", var_names=["y_like"]
    ):
        """Function to make predictions from the model
        Args:
            X: Covariates
            batch_effects: batch effects corresponding to X
            all_batch_effects: combinations of all batch effects that were present the training data
        """
        X, batch_effects = expand_all(X, batch_effects)

        samples = self.configs["n_samples"]
        y = np.zeros([X.shape[0], 1])
        X = self.transform_X(X)
        modeler = self.get_modeler()

        # Make an array with occurences of all the values in be_train, but with the same size as be_test
        truncated_batch_effects_train = np.stack(
            [
                np.resize(np.array(list(batch_effects_maps[i].keys())), X.shape[0])
                for i in range(batch_effects.shape[1])
            ],
            axis=1,
        )
        n_samples = X.shape[0]
        with modeler(X, y, truncated_batch_effects_train, self.configs) as model:
            # TODO
            # This fails because the batch_effects provided here may not contain the same batch_effects as in the training set. If some are missing, the dimensions of the distributions don't match
            # For each batch effect dim
            for i in range(batch_effects.shape[1]):
                # Make a map that maps batch effect values to their index
                valmap = batch_effects_maps[i]
                # Compute those indices for the test data
                indices = list(map(lambda x: valmap[x], batch_effects[:, i]))
                # Those indices need to be used by the model
                pm.set_data({f"batch_effect_{i}": indices})

            self.idata = pm.sample_posterior_predictive(
                trace=self.idata,
                extend_inferencedata=True,
                progressbar=True,
                var_names=var_names,
            )
        pred_mean = self.idata.posterior_predictive["y_like"].to_numpy().mean(axis=(0, 1))
        pred_var = self.idata.posterior_predictive["y_like"].to_numpy().var(axis=(0, 1))

        return pred_mean, pred_var

    def estimate_on_new_site(self, X, y, batch_effects):
        """Function to adapt the model"""
        X, y, batch_effects = expand_all(X, y, batch_effects)
        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs, idata=self.idata) as m:
            self.idata = pm.sample(
                self.configs["n_samples"],
                tune=self.configs["n_tuning"],
                chains=self.configs["n_chains"],
                target_accept=self.configs["target_accept"],
                init=self.configs["init"],
                n_init=50000,
                cores=self.configs["cores"],
            )
        return self.idata

    def predict_on_new_site(self, X, batch_effects):
        """Function to make predictions from the model"""
        X, batch_effects = expand_all(X, batch_effects)
        samples = self.configs["n_samples"]
        y = np.zeros([X.shape[0], 1])
        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs, idata=self.idata):
            self.idata = pm.sample_posterior_predictive(
                self.idata, extend_inferencedata=True, progressbar=True
            )
        pred_mean = self.idata.posterior_predictive["y_like"].mean(axis=(0, 1))
        pred_var = self.idata.posterior_predictive["y_like"].var(axis=(0, 1))

        return pred_mean, pred_var

    def generate(self, X, batch_effects, samples):
        """Function to generate samples from posterior predictive distribution"""
        X, batch_effects = expand_all(X, batch_effects)

        y = np.zeros([X.shape[0], 1])

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs):
            ppc = pm.sample_posterior_predictive(self.idata, progressbar=True)
        generated_samples = np.reshape(
            ppc.posterior_predictive["y_like"].squeeze().T, [X.shape[0] * samples, 1]
        )
        X = np.repeat(X, samples)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        batch_effects = np.repeat(batch_effects, samples, axis=0)
        if len(batch_effects.shape) == 1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        return X, batch_effects, generated_samples

    def sample_prior_predictive(self, X, batch_effects, samples, idata=None):
        """Function to sample from prior predictive distribution"""
        X, batch_effects = expand_all(X, batch_effects)
        y = np.zeros([X.shape[0], 1])
        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs, idata):
            self.idata = pm.sample_prior_predictive(samples=samples)
        return self.idata

    def get_model(self, X, y, batch_effects):
        X, y, batch_effects = expand_all(X, y, batch_effects)
        modeler = self.get_modeler()
        X = self.transform_X(X)
        idata = self.idata if hasattr(self, "idata") else None
        return modeler(X, y, batch_effects, self.configs, idata=idata)

    def create_dummy_inputs(self, covariate_ranges=[[0.1, 0.9, 0.01]]):
        arrays = []
        for i in range(len(covariate_ranges)):
            arrays.append(
                np.arange(
                    covariate_ranges[i][0],
                    covariate_ranges[i][1],
                    covariate_ranges[i][2],
                )
            )
        X = cartesian_product(arrays)
        X_dummy = np.concatenate([X for i in range(np.prod(self.batch_effects_size))])
        arrays = []
        for i in range(self.batch_effects_num):
            arrays.append(np.arange(0, self.batch_effects_size[i]))
        batch_effects = cartesian_product(arrays)
        batch_effects_dummy = np.repeat(batch_effects, X.shape[0], axis=0)
        return X_dummy, batch_effects_dummy

    def Rhats(self, var_names, thin = 1, resolution = 100):
        """Get Rhat of posterior samples as function of sampling iteration"""
        idata = self.idata
        testvars = az.extract(idata, group='posterior', var_names=var_names, combined=False)
        rhat_dict={}
        for var_name in var_names:
            var = np.stack(testvars[var_name].to_numpy())[:,::thin]     
            var = var.reshape((var.shape[0], var.shape[1], -1))
            vardim = var.shape[2]
            interval_skip=var.shape[1]//resolution
            rhats_var = np.zeros((resolution, vardim))
            for v in range(vardim):
                for j in range(resolution):
                    rhats_var[j,v] = az.rhat(var[:,:j*interval_skip,v])
            rhat_dict[var_name] = rhats_var
        return rhat_dict

class Prior:
    """
    A wrapper class for a PyMC distribution.
    - creates a fitted distribution from the idata, if one is present
    - overloads the __getitem__ function with something that switches between indexing or not, based on the shape
    """

    def __init__(self, name, dist, params, pb, has_random_effect=False) -> None:
        self.dist = None
        self.name = name
        self.has_random_effect = has_random_effect
        self.distmap = {
            "normal": pm.Normal,
            "hnormal": pm.HalfNormal,
            "gamma": pm.Gamma,
            "uniform": pm.Uniform,
            "igamma": pm.InverseGamma,
            "hcauchy": pm.HalfCauchy,
            "hstudt": pm.HalfStudentT,
            "studt": pm.StudentT,
        }
        self.make_dist(dist, params, pb)

    def make_dist(self, dist, params, pb):
        """This creates a pymc distribution. If there is a idata, the distribution is fitted to the idata. If there isn't a idata, the prior is parameterized by the values in (params)"""
        with pb.model as m:
            if pb.idata is not None:
                samples = az.extract(pb.idata, var_names=self.name).to_numpy()
                if not self.has_random_effect:
                    samples = np.reshape(samples, (-1,))
                self.dist = from_posterior(
                    param=self.name,
                    samples=samples,
                    distribution=dist,
                    freedom=pb.configs["freedom"],
                )
            dims = []
            if self.has_random_effect:
                dims = dims + pb.batch_effect_dim_names
            if self.name.startswith("slope") or self.name.startswith("offset_slope"):
                dims = dims + ["basis_functions"]
            if dims == []:
                self.dist = self.distmap[dist](self.name, *params)
            else:
                self.dist = self.distmap[dist](self.name, *params, dims=dims)

    def __getitem__(self, idx):
        """The idx here is the index of the batch-effect. If the prior does not model batch effects, this should return the same value for each index"""
        assert self.dist is not None, "Distribution not initialized"
        if self.has_random_effect:
            return self.dist[idx]
        else:
            return self.dist


class ParamBuilder:
    """
    A class that simplifies the construction of parameterizations.
    It has a lot of attributes necessary for creating the model, including the data, but it is never saved with the model.
    It also contains a lot of decision logic for creating the parameterizations.
    """

    def __init__(self, X, y, batch_effects, idata, configs):
        """

        :param model: model to attach all the distributions to
        :param X: Covariates
        :param y: IDPs
        :param batch_effects: array of batch effects
        :param idata:  idem
        :param configs: idem
        """
        self.model = None  # Needs to be set later, because coords need to be passed at construction of Model
        self.X = X
        self.n_basis_functions = X.shape[1]
        self.y = y
        self.batch_effects = batch_effects.astype(np.int16)
        self.idata: az.InferenceData = idata
        self.configs = configs

        self.y_shape = y.shape
        self.n_ys = y.shape[0]
        self.batch_effects_num = batch_effects.shape[1]

        self.batch_effect_dim_names = []
        self.batch_effect_indices = []
        self.coords = OrderedDict()
        self.coords["basis_functions"] = np.arange(self.n_basis_functions)

        for i in range(self.batch_effects_num):
            batch_effect_dim_name = f"batch_effect_{i}"
            self.batch_effect_dim_names.append(batch_effect_dim_name)
            this_be_values, this_be_indices = np.unique(
                self.batch_effects[:, i], return_inverse=True
            )
            self.coords[batch_effect_dim_name] = this_be_values
            self.batch_effect_indices.append(this_be_indices)

    def make_param(self, name, **kwargs):
        if self.configs.get(f"linear_{name}", False):
            # First make a slope and intercept, and use those to make a linear parameterization
            slope_parameterization = self.make_param(f"slope_{name}", **kwargs)
            intercept_parameterization = self.make_param(f"intercept_{name}", **kwargs)
            return LinearParameterization(
                name=name,
                slope_parameterization=slope_parameterization,
                intercept_parameterization=intercept_parameterization,
                **kwargs,
            )

        elif self.configs.get(f"random_{name}", False):
            if self.configs.get(f"centered_{name}", True):
                return CentralRandomFixedParameterization(name=name, pb=self, **kwargs)
            else:
                return NonCentralRandomFixedParameterization(
                    name=name, pb=self, **kwargs
                )
        else:
            return FixedParameterization(name=name, pb=self, **kwargs)


class Parameterization:
    """
    This is the top-level parameterization class from which all the other parameterizations inherit.
    """

    def __init__(self, name):
        self.name = name
        print(name, type(self))

    def get_samples(self, pb):
        pass


class FixedParameterization(Parameterization):
    """
    A parameterization that takes a single value for all input. It does not depend on anything except its hyperparameters
    """

    def __init__(self, name, pb: ParamBuilder, **kwargs):
        super().__init__(name)
        dist = kwargs.get(f"{name}_dist", "normal")
        params = kwargs.get(f"{name}_params", (0.0, 1.0))
        self.dist = Prior(name, dist, params, pb)

    def get_samples(self, pb):
        with pb.model:
            return self.dist[0]


class CentralRandomFixedParameterization(Parameterization):
    """
    A parameterization that is fixed for each batch effect. This is sampled in a central fashion;
    the values are sampled from normal distribution with a group mean and group variance
    """

    def __init__(self, name, pb: ParamBuilder, **kwargs):
        super().__init__(name)

        # Normal distribution is default for mean
        mu_dist = kwargs.get(f"mu_{name}_dist", "normal")
        mu_params = kwargs.get(f"mu_{name}_params", (0.0, 1.0))
        mu_prior = Prior(f"mu_{name}", mu_dist, mu_params, pb)

        # HalfNormal is default for sigma
        sigma_dist = kwargs.get(f"sigma_{name}_dist", "hnormal")
        sigma_params = kwargs.get(f"sigma_{name}_params", (1.0,))
        sigma_prior = Prior(f"sigma_{name}", sigma_dist, sigma_params, pb)

        dims = (
            [*pb.batch_effect_dim_names, "basis_functions"]
            if self.name.startswith("slope")
            else pb.batch_effect_dim_names
        )
        self.dist = pm.Normal(
            name=name,
            mu=mu_prior.dist,
            sigma=sigma_prior.dist,
            dims=dims,
        )

    def get_samples(self, pb: ParamBuilder):
        with pb.model:
            samples = self.dist[pb.batch_effect_indices]
            return samples


class NonCentralRandomFixedParameterization(Parameterization):
    """
    A parameterization that is fixed for each batch effect. This is sampled in a non-central fashion;
    the values are a sum of a group mean and noise values scaled with a group scaling factor
    """

    def __init__(self, name, pb: ParamBuilder, **kwargs):
        super().__init__(name)

        # Normal distribution is default for mean
        mu_dist = kwargs.get(f"mu_{name}_dist", "normal")
        mu_params = kwargs.get(f"mu_{name}_params", (0.0, 1.0))
        mu_prior = Prior(f"mu_{name}", mu_dist, mu_params, pb)

        # HalfNormal is default for sigma
        sigma_dist = kwargs.get(f"sigma_{name}_dist", "hnormal")
        sigma_params = kwargs.get(f"sigma_{name}_params", (1.0,))
        sigma_prior = Prior(f"sigma_{name}", sigma_dist, sigma_params, pb)

        # Normal is default for offset
        offset_dist = kwargs.get(f"offset_{name}_dist", "normal")
        offset_params = kwargs.get(f"offset_{name}_params", (0.0, 1.0))
        offset_prior = Prior(
            f"offset_{name}", offset_dist, offset_params, pb, has_random_effect=True
        )
        dims = (
            [*pb.batch_effect_dim_names, "basis_functions"]
            if self.name.startswith("slope")
            else pb.batch_effect_dim_names
        )
        self.dist = pm.Deterministic(
            name=name,
            var=mu_prior.dist + sigma_prior.dist * offset_prior.dist,
            dims=dims,
        )

    def get_samples(self, pb: ParamBuilder):
        with pb.model:
            samples = self.dist[pb.batch_effect_indices]
            return samples


class LinearParameterization(Parameterization):
    """
    A parameterization that can model a linear dependence on X.
    """

    def __init__(
        self, name, slope_parameterization, intercept_parameterization, **kwargs
    ):
        super().__init__(name)
        self.slope_parameterization = slope_parameterization
        self.intercept_parameterization = intercept_parameterization

    def get_samples(self, pb):
        with pb.model:
            intc = self.intercept_parameterization.get_samples(pb)
            slope_samples = self.slope_parameterization.get_samples(pb)
            if pb.configs[f"random_slope_{self.name}"]:
                slope = pb.X * self.slope_parameterization.get_samples(pb)
                slope = slope.sum(axis=-1)
            else:
                slope = pb.X @ self.slope_parameterization.get_samples(pb)

            samples = pm.math.flatten(intc) + pm.math.flatten(slope)
            return samples


def get_design_matrix(X, nm, basis="linear"):
    if basis == "bspline":
        Phi = bspline_transform(X, nm.hbr.bsp)
    elif basis == "polynomial":
        Phi = create_poly_basis(X, 3)
    else:
        Phi = X
    return Phi


def nn_hbr(X, y, batch_effects, batch_effects_size, configs, idata=None):
    n_hidden = configs["nn_hidden_neuron_num"]
    n_layers = configs["nn_hidden_layers_num"]
    feature_num = X.shape[1]
    batch_effects_num = batch_effects.shape[1]
    all_idx = []
    for i in range(batch_effects_num):
        all_idx.append(np.int16(np.unique(batch_effects[:, i])))
    be_idx = list(product(*all_idx))

    # Initialize random weights between each layer for the mu:
    init_1 = pm.floatX(
        np.random.randn(feature_num, n_hidden) * np.sqrt(1 / feature_num)
    )
    init_out = pm.floatX(np.random.randn(n_hidden) * np.sqrt(1 / n_hidden))

    std_init_1 = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out = pm.floatX(np.random.rand(n_hidden))

    # And initialize random weights between each layer for sigma_noise:
    init_1_noise = pm.floatX(
        np.random.randn(feature_num, n_hidden) * np.sqrt(1 / feature_num)
    )
    init_out_noise = pm.floatX(np.random.randn(n_hidden) * np.sqrt(1 / n_hidden))

    std_init_1_noise = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out_noise = pm.floatX(np.random.rand(n_hidden))

    # If there are two hidden layers, then initialize weights for the second layer:
    if n_layers == 2:
        init_2 = pm.floatX(np.random.randn(n_hidden, n_hidden) * np.sqrt(1 / n_hidden))
        std_init_2 = pm.floatX(np.random.rand(n_hidden, n_hidden))
        init_2_noise = pm.floatX(
            np.random.randn(n_hidden, n_hidden) * np.sqrt(1 / n_hidden)
        )
        std_init_2_noise = pm.floatX(np.random.rand(n_hidden, n_hidden))

    with pm.Model() as model:
        X = pm.Data("X", X)
        y = pm.Data("y", y)

        if idata is not None:  # Used when estimating/predicting on a new site
            weights_in_1_grp = from_posterior(
                "w_in_1_grp",
                idata["w_in_1_grp"],
                distribution="normal",
                freedom=configs["freedom"],
            )

            weights_in_1_grp_sd = from_posterior(
                "w_in_1_grp_sd",
                idata["w_in_1_grp_sd"],
                distribution="hcauchy",
                freedom=configs["freedom"],
            )

            if n_layers == 2:
                weights_1_2_grp = from_posterior(
                    "w_1_2_grp",
                    idata["w_1_2_grp"],
                    distribution="normal",
                    freedom=configs["freedom"],
                )

                weights_1_2_grp_sd = from_posterior(
                    "w_1_2_grp_sd",
                    idata["w_1_2_grp_sd"],
                    distribution="hcauchy",
                    freedom=configs["freedom"],
                )

            weights_2_out_grp = from_posterior(
                "w_2_out_grp",
                idata["w_2_out_grp"],
                distribution="normal",
                freedom=configs["freedom"],
            )

            weights_2_out_grp_sd = from_posterior(
                "w_2_out_grp_sd",
                idata["w_2_out_grp_sd"],
                distribution="hcauchy",
                freedom=configs["freedom"],
            )

            mu_prior_intercept = from_posterior(
                "mu_prior_intercept",
                idata["mu_prior_intercept"],
                distribution="normal",
                freedom=configs["freedom"],
            )
            sigma_prior_intercept = from_posterior(
                "sigma_prior_intercept",
                idata["sigma_prior_intercept"],
                distribution="hcauchy",
                freedom=configs["freedom"],
            )

        else:
            # Group the mean distribution for input to the hidden layer:
            weights_in_1_grp = pm.Normal(
                "w_in_1_grp", 0, sd=1, shape=(feature_num, n_hidden), testval=init_1
            )

            # Group standard deviation:
            weights_in_1_grp_sd = pm.HalfCauchy(
                "w_in_1_grp_sd", 1.0, shape=(feature_num, n_hidden), testval=std_init_1
            )

            if n_layers == 2:
                # Group the mean distribution for hidden layer 1 to hidden layer 2:
                weights_1_2_grp = pm.Normal(
                    "w_1_2_grp", 0, sd=1, shape=(n_hidden, n_hidden), testval=init_2
                )

                # Group standard deviation:
                weights_1_2_grp_sd = pm.HalfCauchy(
                    "w_1_2_grp_sd", 1.0, shape=(n_hidden, n_hidden), testval=std_init_2
                )

            # Group the mean distribution for hidden to output:
            weights_2_out_grp = pm.Normal(
                "w_2_out_grp", 0, sd=1, shape=(n_hidden,), testval=init_out
            )

            # Group standard deviation:
            weights_2_out_grp_sd = pm.HalfCauchy(
                "w_2_out_grp_sd", 1.0, shape=(n_hidden,), testval=std_init_out
            )

            # mu_prior_intercept = pm.Uniform('mu_prior_intercept', lower=-100, upper=100)
            mu_prior_intercept = pm.Normal("mu_prior_intercept", mu=0.0, sigma=1e3)
            sigma_prior_intercept = pm.HalfCauchy("sigma_prior_intercept", 5)

        # Now create separate weights for each group, by doing
        # weights * group_sd + group_mean, we make sure the new weights are
        # coming from the (group_mean, group_sd) distribution.
        weights_in_1_raw = pm.Normal(
            "w_in_1", 0, sd=1, shape=(batch_effects_size + [feature_num, n_hidden])
        )
        weights_in_1 = weights_in_1_raw * weights_in_1_grp_sd + weights_in_1_grp

        if n_layers == 2:
            weights_1_2_raw = pm.Normal(
                "w_1_2", 0, sd=1, shape=(batch_effects_size + [n_hidden, n_hidden])
            )
            weights_1_2 = weights_1_2_raw * weights_1_2_grp_sd + weights_1_2_grp

        weights_2_out_raw = pm.Normal(
            "w_2_out", 0, sd=1, shape=(batch_effects_size + [n_hidden])
        )
        weights_2_out = weights_2_out_raw * weights_2_out_grp_sd + weights_2_out_grp

        intercepts_offset = pm.Normal(
            "intercepts_offset", mu=0, sd=1, shape=(batch_effects_size)
        )

        intercepts = pm.Deterministic(
            "intercepts", intercepts_offset + mu_prior_intercept * sigma_prior_intercept
        )

        # Build the neural network and estimate y_hat:
        y_hat = pytensor.tensor.zeros(y.shape)
        for be in be_idx:
            # Find the indices corresponding to 'group be':
            a = []
            for i, b in enumerate(be):
                a.append(batch_effects[:, i] == b)
            idx = reduce(np.logical_and, a).nonzero()
            if idx[0].shape[0] != 0:
                act_1 = pm.math.tanh(pytensor.tensor.dot(X[idx, :], weights_in_1[be]))
                if n_layers == 2:
                    act_2 = pm.math.tanh(pytensor.tensor.dot(act_1, weights_1_2[be]))
                    y_hat = pytensor.tensor.set_subtensor(
                        y_hat[idx, 0],
                        intercepts[be] + pytensor.tensor.dot(act_2, weights_2_out[be]),
                    )
                else:
                    y_hat = pytensor.tensor.set_subtensor(
                        y_hat[idx, 0],
                        intercepts[be] + pytensor.tensor.dot(act_1, weights_2_out[be]),
                    )

        # If we want to estimate varying noise terms across groups:
        if configs["random_noise"]:
            if configs["hetero_noise"]:
                if idata is not None:  # # Used when estimating/predicting on a new site
                    weights_in_1_grp_noise = from_posterior(
                        "w_in_1_grp_noise",
                        idata["w_in_1_grp_noise"],
                        distribution="normal",
                        freedom=configs["freedom"],
                    )

                    weights_in_1_grp_sd_noise = from_posterior(
                        "w_in_1_grp_sd_noise",
                        idata["w_in_1_grp_sd_noise"],
                        distribution="hcauchy",
                        freedom=configs["freedom"],
                    )

                    if n_layers == 2:
                        weights_1_2_grp_noise = from_posterior(
                            "w_1_2_grp_noise",
                            idata["w_1_2_grp_noise"],
                            distribution="normal",
                            freedom=configs["freedom"],
                        )

                        weights_1_2_grp_sd_noise = from_posterior(
                            "w_1_2_grp_sd_noise",
                            idata["w_1_2_grp_sd_noise"],
                            distribution="hcauchy",
                            freedom=configs["freedom"],
                        )

                    weights_2_out_grp_noise = from_posterior(
                        "w_2_out_grp_noise",
                        idata["w_2_out_grp_noise"],
                        distribution="normal",
                        freedom=configs["freedom"],
                    )

                    weights_2_out_grp_sd_noise = from_posterior(
                        "w_2_out_grp_sd_noise",
                        idata["w_2_out_grp_sd_noise"],
                        distribution="hcauchy",
                        freedom=configs["freedom"],
                    )

                else:
                    # The input layer to the first hidden layer:
                    weights_in_1_grp_noise = pm.Normal(
                        "w_in_1_grp_noise",
                        0,
                        sd=1,
                        shape=(feature_num, n_hidden),
                        testval=init_1_noise,
                    )
                    weights_in_1_grp_sd_noise = pm.HalfCauchy(
                        "w_in_1_grp_sd_noise",
                        1,
                        shape=(feature_num, n_hidden),
                        testval=std_init_1_noise,
                    )

                    # The first hidden layer to second hidden layer:
                    if n_layers == 2:
                        weights_1_2_grp_noise = pm.Normal(
                            "w_1_2_grp_noise",
                            0,
                            sd=1,
                            shape=(n_hidden, n_hidden),
                            testval=init_2_noise,
                        )
                        weights_1_2_grp_sd_noise = pm.HalfCauchy(
                            "w_1_2_grp_sd_noise",
                            1,
                            shape=(n_hidden, n_hidden),
                            testval=std_init_2_noise,
                        )

                    # The second hidden layer to output layer:
                    weights_2_out_grp_noise = pm.Normal(
                        "w_2_out_grp_noise",
                        0,
                        sd=1,
                        shape=(n_hidden,),
                        testval=init_out_noise,
                    )
                    weights_2_out_grp_sd_noise = pm.HalfCauchy(
                        "w_2_out_grp_sd_noise",
                        1,
                        shape=(n_hidden,),
                        testval=std_init_out_noise,
                    )

                    # mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', sigma=1e3)
                    # sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)

                # Now create separate weights for each group:
                weights_in_1_raw_noise = pm.Normal(
                    "w_in_1_noise",
                    0,
                    sd=1,
                    shape=(batch_effects_size + [feature_num, n_hidden]),
                )
                weights_in_1_noise = (
                    weights_in_1_raw_noise * weights_in_1_grp_sd_noise
                    + weights_in_1_grp_noise
                )

                if n_layers == 2:
                    weights_1_2_raw_noise = pm.Normal(
                        "w_1_2_noise",
                        0,
                        sd=1,
                        shape=(batch_effects_size + [n_hidden, n_hidden]),
                    )
                    weights_1_2_noise = (
                        weights_1_2_raw_noise * weights_1_2_grp_sd_noise
                        + weights_1_2_grp_noise
                    )

                weights_2_out_raw_noise = pm.Normal(
                    "w_2_out_noise", 0, sd=1, shape=(batch_effects_size + [n_hidden])
                )
                weights_2_out_noise = (
                    weights_2_out_raw_noise * weights_2_out_grp_sd_noise
                    + weights_2_out_grp_noise
                )

                # intercepts_offset_noise = pm.Normal('intercepts_offset_noise', mu=0, sd=1,
                #                          shape=(batch_effects_size))

                # intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise +
                #                      intercepts_offset_noise * sigma_prior_intercept_noise)

                # Build the neural network and estimate the sigma_y:
                sigma_y = pytensor.tensor.zeros(y.shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:, i] == b)
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0] != 0:
                        act_1_noise = pm.math.sigmoid(
                            pytensor.tensor.dot(X[idx, :], weights_in_1_noise[be])
                        )
                        if n_layers == 2:
                            act_2_noise = pm.math.sigmoid(
                                pytensor.tensor.dot(act_1_noise, weights_1_2_noise[be])
                            )
                            temp = (
                                pm.math.log1pexp(
                                    pytensor.tensor.dot(
                                        act_2_noise, weights_2_out_noise[be]
                                    )
                                )
                                + 1e-5
                            )
                        else:
                            temp = (
                                pm.math.log1pexp(
                                    pytensor.tensor.dot(
                                        act_1_noise, weights_2_out_noise[be]
                                    )
                                )
                                + 1e-5
                            )
                        sigma_y = pytensor.tensor.set_subtensor(sigma_y[idx, 0], temp)

            else:  # homoscedastic noise:
                if idata is not None:  # Used for transferring the priors
                    upper_bound = np.percentile(idata["sigma_noise"], 95)
                    sigma_noise = pm.Uniform(
                        "sigma_noise",
                        lower=0,
                        upper=2 * upper_bound,
                        shape=(batch_effects_size),
                    )
                else:
                    sigma_noise = pm.Uniform(
                        "sigma_noise", lower=0, upper=100, shape=(batch_effects_size)
                    )

                sigma_y = pytensor.tensor.zeros(y.shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:, i] == b)
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0] != 0:
                        sigma_y = pytensor.tensor.set_subtensor(
                            sigma_y[idx, 0], sigma_noise[be]
                        )

        else:  # do not allow for random noise terms across groups:
            if idata is not None:  # Used for transferring the priors
                upper_bound = np.percentile(idata["sigma_noise"], 95)
                sigma_noise = pm.Uniform("sigma_noise", lower=0, upper=2 * upper_bound)
            else:
                sigma_noise = pm.Uniform("sigma_noise", lower=0, upper=100)
            sigma_y = pytensor.tensor.zeros(y.shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:, i] == b)
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0] != 0:
                    sigma_y = pytensor.tensor.set_subtensor(
                        sigma_y[idx, 0], sigma_noise
                    )

        if configs["skewed_likelihood"]:
            skewness = pm.Uniform(
                "skewness", lower=-10, upper=10, shape=(batch_effects_size)
            )
            alpha = pytensor.tensor.zeros(y.shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:, i] == b)
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0] != 0:
                    alpha = pytensor.tensor.set_subtensor(alpha[idx, 0], skewness[be])
        else:
            alpha = 0  # symmetrical normal distribution

        y_like = pm.SkewNormal(
            "y_like", mu=y_hat, sigma=sigma_y, alpha=alpha, observed=y
        )

    return model
