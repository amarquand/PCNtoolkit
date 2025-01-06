#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:23:15 2019

@author: seykia
@author: augub
"""

from __future__ import division, print_function

from collections import OrderedDict
from functools import reduce
from itertools import product

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import xarray
from scipy import stats
from util.utils import create_poly_basis, expand_all

from pcntoolkit.model.SHASH import *
from pcntoolkit.util.bspline import BSplineBasis
from pcntoolkit.util.utils import cartesian_product


def create_poly_basis(X, order):
    """
    Create a polynomial basis expansion of the specified order
    :param X: [N×P] array of clinical covariates
    :param order: order of the polynomial
    :return: a [N×(P×order)] array of transformed data
    """
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
    Phi = np.zeros((X.shape[0], D * order))
    colid = np.arange(0, D)
    for d in range(1, order + 1):
        Phi[:, colid] = X**d
        colid += D
    return Phi


def from_posterior(param, samples, shape,  distribution=None, dims=None, half=False, freedom=1):
    """
    Create a PyMC distribution from posterior samples

    :param param: name of the parameter
    :param samples: samples from the posterior
    :param shape: shape of the parameter
    :param distribution: distribution to use for the parameter
    :param dims: dims of the parameter
    :param half: if true, the distribution is assumed to be defined on the positive real line 
    :param freedom: freedom parameter for the distribution
    :return: a PyMC distribution
    """
    if dims == []:
        dims = None
    if distribution is None:
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 1000)
        y = stats.gaussian_kde(np.ravel(samples))(x)
        if half:
            x = np.concatenate([x, [x[-1] + 0.1 * width]])
            y = np.concatenate([y, [0]])
        else:
            x = np.concatenate(
                [[x[0] - 0.1 * width], x, [x[-1] + 0.1 * width]])
            y = np.concatenate([[0], y, [0]])
        if shape is None:
            return pm.distributions.Interpolated(param, x, y)
        else:
            return pm.distributions.Interpolated(param, x, y, shape=shape, dims=dims)
    elif distribution == "normal":
        temp = stats.norm.fit(samples)
        if shape is None:
            return pm.Normal(param, mu=temp[0], sigma=freedom * temp[1])
        else:
            return pm.Normal(param, mu=temp[0], sigma=freedom * temp[1], shape=shape, dims=dims)
    elif distribution == "hnormal":
        temp = stats.halfnorm.fit(samples)
        if shape is None:
            return pm.HalfNormal(param, sigma=freedom * temp[1])
        else:
            return pm.HalfNormal(param, sigma=freedom * temp[1], shape=shape, dims=dims)
    elif distribution == "hcauchy":
        temp = stats.halfcauchy.fit(samples)
        if shape is None:
            return pm.HalfCauchy(param, freedom * temp[1])
        else:
            return pm.HalfCauchy(param, freedom * temp[1], shape=shape, dims=dims)
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
                dims=dims,
            )
    elif distribution == "huniform":
        upper_bound = np.percentile(samples, 95)
        lower_bound = np.percentile(samples, 5)
        r = np.abs(upper_bound - lower_bound)
        if shape is None:
            return pm.Uniform(param, lower=0, upper=upper_bound + freedom * r)
        else:
            return pm.Uniform(
                param, lower=0, upper=upper_bound + freedom * r, shape=shape, dims=dims
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
                dims=dims,
            )

    elif distribution == "igamma":
        alpha_fit, loc_fit, beta_fit = stats.gamma.fit(samples)
        if shape is None:
            return pm.InverseGamma(
                param, alpha=freedom * alpha_fit, beta=freedom * beta_fit
            )
        else:
            return pm.InverseGamma(
                param, alpha=freedom * alpha_fit, beta=freedom * beta_fit, shape=shape, dims=dims   
            )


def hbr(X, y, batch_effects, configs, idata=None):
    """
    Create a Hierarchical Bayesian Regression model

    :param X: [N×P] array of clinical covariates
    :param y: [N×1] array of neuroimaging measures
    :param batch_effects: [N×M] array of batch effects
    :param configs:
    :param idata:
    :param return_shared_variables: If true, returns references to the shared variables. The values of the shared variables can be set manually, allowing running the same model on different data without re-compiling it.
    :return:
    """

    # Make a param builder that contains all the data and configs
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
        model.add_coord("datapoints", np.arange(X.shape[0]))
        X = pm.Data("X", X, dims=("datapoints", "basis_functions"))
        pb.X = X
        y = pm.Data("y", np.squeeze(y), dims="datapoints")
        pb.y = y
        pb.model = model
        pb.batch_effect_indices = tuple(
            [
                pm.Data(
                    pb.batch_effect_dim_names[i]+"_data",
                    pb.batch_effect_indices[i],
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
                    intercept_mu_params=(0.0, 10.0),
                    slope_mu_params=(0.0, 10.0),
                    mu_slope_mu_params=(0.0, 10.0),
                    sigma_slope_mu_params=(10.0,),
                    mu_intercept_mu_params=(0.0, 10.0),
                    sigma_intercept_mu_params=(10.0,),
                ).get_samples(pb),
                dims=get_sample_dims('mu'),
            )
            sigma = pm.Deterministic(
                "sigma_samples",
                pb.make_param(
                    "sigma",
                    sigma_params=(10., 10.0),
                    sigma_dist="normal",
                    slope_sigma_params=(0.0, 10.0),
                    intercept_sigma_params=(10.0, 10.0),
                ).get_samples(pb),
                dims=get_sample_dims('sigma'),
            )
            sigma_plus = pm.Deterministic(
                "sigma_plus_samples", np.log(1+np.exp(sigma/10))*10, dims=get_sample_dims('sigma')
            )
            y_like = pm.Normal(
                "y_like",
                mu=mu,
                sigma=sigma_plus,
                observed=y,
                dims="datapoints",
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
            SHASH_map = {"SHASHb": SHASHb,
                         "SHASHo": SHASHo, "SHASHo2": SHASHo2}

            mu = pm.Deterministic(
                "mu_samples",
                pb.make_param(
                    "mu",
                    intercept_mu_params=(0.0, 10.0),
                    slope_mu_params=(0.0, 10.0),
                    mu_slope_mu_params=(0.0, 10.0),
                    sigma_slope_mu_params=(10.0,),
                    mu_intercept_mu_params=(0.0, 10.0),
                    sigma_intercept_mu_params=(10.0,),
                ).get_samples(pb),
                dims=get_sample_dims('mu'),
            )
            sigma = pm.Deterministic(
                "sigma_samples",
                pb.make_param(
                    "sigma",
                    sigma_params=(10., 10.0),
                    sigma_dist="normal",
                    slope_sigma_params=(0.0, 10.0),
                    intercept_sigma_params=(10.0, 10.0),
                ).get_samples(pb),
                dims=get_sample_dims('sigma'),
            )
            sigma_plus = pm.Deterministic(
                "sigma_plus_samples", np.log(1+np.exp(sigma/10))*10, dims=get_sample_dims('sigma')
            )
            epsilon = pm.Deterministic(
                "epsilon_samples",
                pb.make_param(
                    "epsilon",
                    epsilon_params=(0.0, 2.0),
                    slope_epsilon_params=(0.0, 3.0),
                    intercept_epsilon_params=(0.0, 3.0),
                ).get_samples(pb),
                dims=get_sample_dims('epsilon'),
            )
            delta = pm.Deterministic(
                "delta_samples",
                pb.make_param(
                    "delta",
                    delta_params=(0., 2.0),
                    delta_dist="normal",
                    slope_delta_params=(0.0, 1.0),
                    intercept_delta_params=(0.0, 1.0),
                ).get_samples(pb),
                dims=get_sample_dims('delta'),
            )
            delta_plus = pm.Deterministic(
                "delta_plus_samples",
                np.log(1+np.exp(delta/3))*3 + 0.3,
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
        """
        This used to return hbr or nnhbr, but now it returns hbr.
        Can be removed in a future release
        //TODO remove this in a future release
        """
        return hbr

    def transform_X(self, X, adapt=False):
        """
        Transform the covariates according to the model type

        :param X: N-by-P input matrix of P features for N subjects
        :return: transformed covariates
        :adapt: Set to true when range adaptation for bspline is needed (for example in the 
        transfer scenario)
        """
        if self.model_type == "polynomial":
            Phi = create_poly_basis(X, self.configs["order"])
        elif self.model_type == "bspline":
            if self.bsp is None:
                self.bsp = BSplineBasis(order=self.configs["order"], 
                                        nknots=self.configs["nknots"])
                self.bsp.fit(X)
                #self.bsp = bspline_fit(
                #    X, self.configs["order"], self.configs["nknots"])
            elif adapt:
                self.bsp.adapt(X)
                
            bspline = self.bsp.transform(X)
            #bspline = bspline_transform(X, self.bsp)
            Phi = np.concatenate((X, bspline), axis=1)
        else:
            Phi = X
        return Phi

    def find_map(self, X, y, batch_effects, method="L-BFGS-B"):
        """
        Find the maximum a posteriori (MAP) estimate of the model parameters.

        This function transforms the data according to the model type, 
        and then uses the modeler to find the MAP estimate of the model parameters. 
        The results are stored in the instance variable `MAP`.

        :param X: N-by-P input matrix of P features for N subjects. This is the input data for the model.
        :param y: N-by-1 vector of outputs. This is the target data for the model.
        :param batch_effects: N-by-B matrix of B batch ids for N subjects. This represents the batch effects to be considered in the model.
        :param method: String representing the optimization method to use. Default is "L-BFGS-B".
        :return: A dictionary of MAP estimates.
        """
        X, y, batch_effects = expand_all(X, y, batch_effects)
        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs) as m:
            self.MAP = pm.find_MAP(method=method)
        return self.MAP

    def estimate(self, X, y, batch_effects, **kwargs):
        """
        Estimate the model parameters using the provided data.

        This function transforms the data according to the model type, 
        and then samples from the posterior using pymc. The results are stored 
        in the instance variable `idata`.

        :param X: N-by-P input matrix of P features for N subjects. This is the input data for the model.
        :param y: N-by-1 vector of outputs. This is the target data for the model.
        :param batch_effects: N-by-B matrix of B batch ids for N subjects. This represents the batch effects to be considered in the model.
        :param kwargs: Additional keyword arguments to be passed to the modeler.
        :return: idata. The results are also stored in the instance variable `self.idata`.
        """
        X, y, batch_effects = expand_all(X, y, batch_effects)
        
        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = [len(np.unique(batch_effects[:,i])) for i in range(self.batch_effects_num)] 
        
        X = self.transform_X(X)
        modeler = self.get_modeler()
        if hasattr(self, 'idata'):
            del self.idata
        with modeler(X, y, batch_effects, self.configs) as m:
            self.idata = pm.sample(
                draws=self.configs["n_samples"],
                tune=self.configs["n_tuning"],
                chains=self.configs["n_chains"],
                init=self.configs["init"],
                n_init=500000,
                cores=self.configs["cores"],
                nuts_sampler=self.configs["nuts_sampler"],
            )
        self.vars_to_sample = ['y_like']
        if self.configs['remove_datapoints_from_posterior']:
            chain = self.idata.posterior.coords['chain'].data
            draw = self.idata.posterior.coords['draw'].data
            for j in self.idata.posterior.variables.mapping.keys():
                if j.endswith('_samples'):
                    dummy_array = xarray.DataArray(data=np.zeros((len(chain), len(draw), 1)), coords={
                                                   'chain': chain, 'draw': draw, 'empty': np.array([0])}, name=j)
                    self.idata.posterior[j] = dummy_array
                    self.vars_to_sample.append(j)

            # zero-out all data
            for i in self.idata.constant_data.data_vars:
                self.idata.constant_data[i] *= 0
            for i in self.idata.observed_data.data_vars:
                self.idata.observed_data[i] *= 0

        return self.idata

    def predict(
        self, X, batch_effects, batch_effects_maps, pred="single", var_names=None, **kwargs
    ):
        """
        Make predictions from the model.

        This function expands the input data, transforms it according to the model type, 
        and then uses the modeler to make predictions. The results are stored in the instance variable `idata`.

        :param X: Covariates. This is the input data for the model.
        :param batch_effects: Batch effects corresponding to X. This represents the batch effects to be considered in the model.
        :param batch_effects_maps: A map from batch_effect values to indices. This is used to map the batch effects to the indices used by the model.
        :param pred: String representing the prediction method to use. Default is "single".
        :param var_names: List of variable names to consider in the prediction. If None or ['y_like'], self.vars_to_sample is used.
        :param kwargs: Additional keyword arguments to be passed to the modeler.
        :return: A 2-tuple of xarray datasets with the mean and variance of the posterior predictive distribution. The results are also stored in the instance variable `self.idata`.
        """
        X, batch_effects = expand_all(X, batch_effects)

        samples = self.configs["n_samples"]
        y = np.zeros([X.shape[0], 1])
        X = self.transform_X(X)
        modeler = self.get_modeler()

        # Make an array with occurences of all the values in be_train, but with the same size as be_test
        truncated_batch_effects_train = np.stack(
            [
                np.resize(
                    np.array(list(batch_effects_maps[i].keys())), X.shape[0])
                for i in range(batch_effects.shape[1])
            ],
            axis=1,
        )

        # See if a list of var_names is provided, set to self.vars_to_sample otherwise
        if (var_names is None) or (var_names == ['y_like']):
            var_names = self.vars_to_sample

        n_samples = X.shape[0]

        # Need to delete self.idata.posterior_predictive, otherwise, if it exists, it will not be overwritten
        if hasattr(self.idata, 'posterior_predictive'):
            del self.idata.posterior_predictive

        with modeler(X, y, truncated_batch_effects_train, self.configs) as model:
            # For each batch effect dim
            for i in range(batch_effects.shape[1]):
                # Make a map that maps batch effect values to their index
                valmap = batch_effects_maps[i]
                # Compute those indices for the test data
                indices = list(map(lambda x: valmap[x], batch_effects[:, i]))
                # Those indices need to be used by the model
                pm.set_data({f"batch_effect_{i}_data": indices})

            self.idata = pm.sample_posterior_predictive(
                trace=self.idata,
                extend_inferencedata=True,
                progressbar=True,
                var_names=var_names
            )
        pred_mean = self.idata.posterior_predictive["y_like"].to_numpy().mean(
            axis=(0, 1))
        pred_var = self.idata.posterior_predictive["y_like"].to_numpy().var(
            axis=(0, 1))

        return pred_mean, pred_var

    def transfer(self, X, y, batch_effects):
        
        """
        This function is used to transfer a reference model (i.e. the source model that is estimated on source big datasets) 
        to data from new sites (i.e. target data). It uses the posterior
        of the reference model as a prior for the target model.

        :param X: Covariates. This is the input data for the model.
        :param y: Outputs. This is the target data for the model.
        :param batch_effects: Batch effects corresponding to X. This represents the batch effects to be considered in the model.
        :return: An inferencedata object containing samples from the posterior distribution.
        """
        X, y, batch_effects = expand_all(X, y, batch_effects)
        
        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = [len(np.unique(batch_effects[:,i])) for i in range(self.batch_effects_num)] 
    
        
        X = self.transform_X(X, adapt=True)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs, idata=self.idata) as m:
            self.idata = pm.sample(
                self.configs["n_samples"],
                tune=self.configs["n_tuning"],
                chains=self.configs["n_chains"],
                target_accept=self.configs["target_accept"],
                init=self.configs["init"],
                n_init=500000,
                cores=self.configs["cores"],
                nuts_sampler=self.configs["nuts_sampler"],
            )
          
        self.vars_to_sample = ['y_like']
        
        # This part is for data privacy
        if self.configs['remove_datapoints_from_posterior']:
            chain = self.idata.posterior.coords['chain'].data
            draw = self.idata.posterior.coords['draw'].data
            for j in self.idata.posterior.variables.mapping.keys():
                if j.endswith('_samples'):
                    dummy_array = xarray.DataArray(data=np.zeros((len(chain), len(draw), 1)), coords={
                                                'chain': chain, 'draw': draw, 'empty': np.array([0])}, name=j)
                    self.idata.posterior[j] = dummy_array
                    self.vars_to_sample.append(j)

            # zero-out all data
            for i in self.idata.constant_data.data_vars:
                self.idata.constant_data[i] *= 0
            for i in self.idata.observed_data.data_vars:
                self.idata.observed_data[i] *= 0
                
        return self.idata


    def generate(self, X, batch_effects, samples, batch_effects_maps, var_names=None):
        """
        Generate samples from the posterior predictive distribution.

        This function expands and transforms the input data, then uses the modeler to generate samples from the posterior predictive distribution. 

        :param X: Covariates. This is the input data for the model.
        :param batch_effects: Batch effects corresponding to X. This represents the batch effects to be considered in the model.
        :param samples: Number of samples to generate. This number of samples is generated for each input sample.
        :return: A tuple containing the expanded and repeated X, batch_effects, and the generated samples.
        """
        X, batch_effects = expand_all(X, batch_effects)
    
        y = np.zeros([X.shape[0], 1])

        X_transformed = self.transform_X(X)
        modeler = self.get_modeler()
        
        # See if a list of var_names is provided, set to self.vars_to_sample otherwise
        if (var_names is None) or (var_names == ['y_like']):
            var_names = self.vars_to_sample

        # Need to delete self.idata.posterior_predictive, otherwise, if it exists, it will not be overwritten
        if hasattr(self.idata, 'posterior_predictive'):
            del self.idata.posterior_predictive

        with modeler(X_transformed, y, batch_effects, self.configs):
            # For each batch effect dim
            for i in range(batch_effects.shape[1]):
                # Make a map that maps batch effect values to their index
                valmap = batch_effects_maps[i]
                # Compute those indices for the test data
                indices = list(map(lambda x: valmap[x], batch_effects[:, i]))
                # Those indices need to be used by the model
                pm.set_data({f"batch_effect_{i}_data": indices})

            self.idata = pm.sample_posterior_predictive(
                trace=self.idata,
                extend_inferencedata=True,
                progressbar=True,
                var_names=var_names
            )
            
        generated_samples = np.reshape(self.idata.posterior_predictive["y_like"].to_numpy()[0,0:samples,:].T, 
                                       [X.shape[0] * samples, 1])
        
        X = np.repeat(X, samples, axis=0)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        batch_effects = np.repeat(batch_effects, samples, axis=0)
        if len(batch_effects.shape) == 1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        return X, batch_effects, generated_samples


    def sample_prior_predictive(self, X, batch_effects, samples, y=None, idata=None):
        """
        Sample from the prior predictive distribution.

        This function transforms the input data, then uses the modeler to sample from the prior predictive distribution. 

        :param X: Covariates. This is the input data for the model.
        :param batch_effects: Batch effects corresponding to X. This represents the batch effects to be considered in the model.
        :param samples: Number of samples to generate. This number of samples is generated for each input sample.
        :param y: Outputs. If None, a zero array of appropriate shape is created.
        :param idata: An xarray dataset with the posterior distribution. If None, self.idata is used if it exists.
        :return: An xarray dataset with the prior predictive distribution. The results are also stored in the instance variable `self.idata`.
        """
        if y is None:
            y = np.zeros([X.shape[0], 1])
        X, y, batch_effects = expand_all(X, y, batch_effects)

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.configs, idata):
            self.idata = pm.sample_prior_predictive(samples=samples)
        return self.idata

    def get_model(self, X, y, batch_effects):
        """
        Get the model for the given data.

        This function expands and transforms the input data, then creates a pymc model using the hbr method

        :param X: Covariates. This is the input data for the model.
        :param y: Outputs. This is the target data for the model.
        :param batch_effects: Batch effects corresponding to X. This represents the batch effects to be considered in the model.
        :return: The model for the given data.
        """
        X, y, batch_effects = expand_all(X, y, batch_effects)
        modeler = self.get_modeler()
        X = self.transform_X(X)
        idata = self.idata if hasattr(self, "idata") else None
        return modeler(X, y, batch_effects, self.configs, idata=idata)

    def create_dummy_inputs(self, X, step_size=0.05):
        """
        Create dummy inputs for the model based on the input covariates.

        This function generates a Cartesian product of the covariate ranges determined from the input X 
        (min and max values of each covariate). It repeats this for each batch effect. 
        It also generates a Cartesian product of the batch effect indices and repeats it for each input sample.

        :param X: 2D numpy array, where rows are samples and columns are covariates.
        :param step_size: Step size for generating ranges for each covariate. Default is 0.05.
        :return: A tuple containing the dummy input data and the dummy batch effects.
        """
        arrays = []
        for i in range(X.shape[1]):
            cov_min = np.min(X[:, i])
            cov_max = np.max(X[:, i])
            arrays.append(np.arange(cov_min, cov_max + step_size, step_size))
        
        X_dummy = cartesian_product(arrays)
        X_dummy = np.concatenate(
            [X_dummy for _ in range(np.prod(self.batch_effects_size))]
        )
        
        arrays = []
        for i in range(self.batch_effects_num):
            arrays.append(np.arange(0, self.batch_effects_size[i]))
        
        batch_effects = cartesian_product(arrays)
        batch_effects_dummy = np.repeat(batch_effects, X_dummy.shape[0] // np.prod(self.batch_effects_size), axis=0)
        
        return X_dummy, batch_effects_dummy


    def Rhats(self, var_names=None, thin=1, resolution=100):
        """
        Get Rhat of posterior samples as function of sampling iteration.

        This function extracts the posterior samples from the instance variable `idata`, computes the Rhat statistic for each variable and sampling iteration, and returns a dictionary of Rhat values.

        :param var_names: List of variable names to consider. If None, all variables in `idata` are used.
        :param thin: Integer representing the thinning factor for the samples. Default is 1.
        :param resolution: Integer representing the number of points at which to compute the Rhat statistic. Default is 100.
        :return: A dictionary where the keys are variable names and the values are arrays of Rhat values.
        """
        idata = self.idata
        testvars = az.extract(idata, group='posterior',
                              var_names=var_names, combined=False)
        testvar_names = [var for var in list(
            testvars.data_vars.keys()) if '_samples' not in var]
        rhat_dict = {}
        for var_name in testvar_names:
            var = np.stack(testvars[var_name].to_numpy())[:, ::thin]
            var = var.reshape((var.shape[0], var.shape[1], -1))
            vardim = var.shape[2]
            interval_skip = var.shape[1]//resolution
            rhats_var = np.zeros((resolution, vardim))
            for v in range(vardim):
                for j in range(resolution):
                    rhats_var[j, v] = az.rhat(var[:, :j*interval_skip, v])
            rhat_dict[var_name] = rhats_var
        return rhat_dict


class Prior:
    """
    A wrapper class for a PyMC distribution.
    - creates a fitted distribution from the idata, if one is present
    - overloads the __getitem__ function with something that switches between indexing or not, based on the shape
    """

    def __init__(self, name, dist, params, pb, has_random_effect=False) -> None:
        """
        Initialize the Prior object.

        This function initializes the Prior object with the given name, distribution, parameters, and model. 
        It also sets a flag indicating whether the prior has a random effect.

        :param name: String representing the name of the prior.
        :param dist: String representing the type of the distribution.
        :param params: Dictionary of parameters for the distribution.
        :param pb: The model object.
        :param has_random_effect: Boolean indicating whether the prior has a random effect. Default is False.
        """
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
            "lognormal": pm.LogNormal,
        }
        self.make_dist(dist, params, pb)

    def make_dist(self, dist, params, pb):
        """
        Create a PyMC distribution.

        This function creates a PyMC distribution. If there is an `idata` present, the distribution is fitted to the `idata`. 
        If there isn't an `idata`, the prior is parameterized by the values in `params`.

        :param dist: String representing the type of the distribution.
        :param params: List of parameters for the distribution.
        :param pb: The model object.
        """
        with pb.model as m:
            if pb.idata is not None:
                # Get samples
                samples = az.extract(pb.idata, var_names=self.name)
                # Define mapping to new shape

                def get_new_dim_size(tup):
                    oldsize, name = tup
                    if name.startswith('batch_effect_'):
                        ind = pb.batch_effect_dim_names.index(name)
                        return len(np.unique(pb.batch_effect_indices[ind].container.data))
                    return oldsize

                new_shape = list(
                    map(get_new_dim_size, zip(samples.shape, samples.dims)))
                if len(new_shape) == 1:
                    new_shape = None
                else:
                    new_shape = new_shape[:-1]

                dims = []
                if self.has_random_effect:
                    dims = dims + pb.batch_effect_dim_names
                if self.name.startswith("slope") or self.name.startswith("offset_slope"):
                    dims = dims + ["basis_functions"]
                if dims == []:
                    self.dist = from_posterior(
                        param=self.name,
                        samples=samples.to_numpy(),
                        shape=new_shape,
                        distribution=dist,
                        freedom=pb.configs["freedom"],
                    )
                else:
                    self.dist = from_posterior(
                        param=self.name,
                        samples=samples.to_numpy(),
                        shape=new_shape,
                        distribution=dist,
                        dims=dims,
                    freedom=pb.configs["freedom"],
                )

            else:
                dims = []
                if self.has_random_effect:
                    dims = dims + pb.batch_effect_dim_names
                if self.name.startswith("slope") or self.name.startswith("offset_slope"):
                    dims = dims + ["basis_functions"]
                if dims == []:
                    self.dist = self.distmap[dist](self.name, *params)
                else:
                    self.dist = self.distmap[dist](
                        self.name, *params, dims=dims)

    def __getitem__(self, idx):
        """
        Retrieve the distribution for a specific batch effect.

        This function retrieves the distribution for a specific batch effect. 
        If the prior does not model batch effects, this should return the same value for each index.

        :param idx: Index of the batch effect.
        :return: The distribution for the specified batch effect.
        """
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
        """
        Create a parameterization based on the configuration.

        This function creates a parameterization based on the configuration. 
        If the configuration specifies a linear parameterization, it creates a slope and intercept and uses those to make a linear parameterization. 
        If the configuration specifies a random parameterization, it creates a random parameterization, either centered or non-centered. 
        Otherwise, it creates a fixed parameterization.

        :param name: String representing the name of the parameter.
        :param kwargs: Additional keyword arguments to be passed to the parameterization.
        :return: The created parameterization.
        """
        if self.configs.get(f"linear_{name}", False):
            # First make a slope and intercept, and use those to make a linear parameterization
            slope_parameterization = self.make_param(f"slope_{name}", **kwargs)
            intercept_parameterization = self.make_param(
                f"intercept_{name}", **kwargs)
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
        """
        Initialize the Parameterization object.

        This function initializes the Parameterization object with the given name.

        :param name: String representing the name of the parameterization.
        """
        self.name = name
        # print(name, type(self))

    def get_samples(self, pb):
        """
        Get samples from the parameterization.

        This function should be overridden by subclasses to provide specific sampling methods.

        :param pb: The ParamBuilder object.
        :return: None. This method should be overridden by subclasses.
        """
        pass


class FixedParameterization(Parameterization):
    """
    A parameterization that takes a single value for all input. 

    It does not depend on anything except its hyperparameters. This class inherits from the Parameterization class.
    """

    def __init__(self, name, pb: ParamBuilder, **kwargs):
        """
        Initialize the FixedParameterization object.

        This function initializes the FixedParameterization object with the given name, ParamBuilder object, and additional arguments.

        :param name: String representing the name of the parameterization.
        :param pb: The ParamBuilder object.
        :param kwargs: Additional keyword arguments to be passed to the parameterization.
        """
        super().__init__(name)
        dist = kwargs.get(f"{name}_dist", "normal")
        params = kwargs.get(f"{name}_params", (0.0, 1.0))
        self.dist = Prior(name, dist, params, pb)

    def get_samples(self, pb):
        """
        Get samples from the parameterization.

        This function gets samples from the parameterization using the ParamBuilder object.

        :param pb: The ParamBuilder object.
        :return: The samples from the parameterization.
        """
        with pb.model:
            return self.dist[0]


class CentralRandomFixedParameterization(Parameterization):
    """
    A parameterization that is fixed for each batch effect. 

    This is sampled in a central fashion; the values are sampled from normal distribution with a group mean and group variance
    """

    def __init__(self, name, pb: ParamBuilder, **kwargs):
        """
        Initialize the CentralRandomFixedParameterization object.

        This function initializes the CentralRandomFixedParameterization object with the given name, ParamBuilder object, and additional arguments.

        :param name: String representing the name of the parameterization.
        :param pb: The ParamBuilder object.
        :param kwargs: Additional keyword arguments to be passed to the parameterization.
        """
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
        """
        Get samples from the parameterization.

        This function gets samples from the parameterization using the ParamBuilder object.

        :param pb: The ParamBuilder object.
        :return: The samples from the parameterization.
        """
        with pb.model:
            return self.dist[pb.batch_effect_indices]



class NonCentralRandomFixedParameterization(Parameterization):
    """
    A parameterization that is fixed for each batch effect. This is sampled in a non-central fashion;
    the values are a sum of a group mean and noise values scaled with a group scaling factor
    """

    def __init__(self, name, pb: ParamBuilder, **kwargs):
        """
        Initialize the NonCentralRandomFixedParameterization object.

        This function initializes the NonCentralRandomFixedParameterization object with the given name, ParamBuilder object, and additional arguments.

        :param name: String representing the name of the parameterization.
        :param pb: The ParamBuilder object.
        :param kwargs: Additional keyword arguments to be passed to the parameterization.
        """
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
        """
        Get samples from the parameterization.

        This function gets samples from the parameterization using the ParamBuilder object.

        :param pb: The ParamBuilder object.
        :return: The samples from the parameterization.
        """
        with pb.model:
            return self.dist[pb.batch_effect_indices]


class LinearParameterization(Parameterization):
    """
    This class inherits from the Parameterization class and represents a parameterization that can model a linear dependence on X.

    """

    def __init__(
        self, name, slope_parameterization, intercept_parameterization, **kwargs
    ):
        """
        Initialize the LinearParameterization object.

        This function initializes the LinearParameterization object with the given name, slope parameterization, intercept parameterization, and additional arguments.

        :param name: String representing the name of the parameterization.
        :param slope_parameterization: An instance of a Parameterization subclass representing the slope.
        :param intercept_parameterization: An instance of a Parameterization subclass representing the intercept.
        :param kwargs: Additional keyword arguments to be passed to the parameterization.
        """
        super().__init__(name)
        self.slope_parameterization = slope_parameterization
        self.intercept_parameterization = intercept_parameterization

    def get_samples(self, pb):
        """
        Get samples from the parameterization.

        This function gets samples from the parameterization using the ParamBuilder object. It computes the samples as the sum of the intercept and the product of X and the slope.

        :param pb: The ParamBuilder object.
        :return: The samples from the parameterization.
        """
        with pb.model:
            intercept_samples = self.intercept_parameterization.get_samples(pb)
            slope_samples = self.slope_parameterization.get_samples(pb)

            if pb.configs[f"random_slope_{self.name}"]:
                if slope_samples.shape.eval()[1] > 1:
                    slope = pm.math.sum(
                        pb.X * slope_samples, axis=1)
                else:
                    slope = pb.X *slope_samples
            else:
                slope = pb.X @ slope_samples

            samples = pm.math.flatten(intercept_samples) + pm.math.flatten(slope)
            return samples


def get_design_matrix(X, nm, basis="linear"):
    """
    Get the design matrix for the given data.

    This function gets the design matrix for the given data.

    :param X: Covariates. This is the input data for the model.
    :param nm: A normative model.
    :param basis: String representing the basis to use. Default is "linear".
    """
    if basis == "bspline":
        Phi = nm.hbr.bsp.transform(X)
        #Phi = bspline_transform(X, nm.hbr.bsp)
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
    init_out_noise = pm.floatX(np.random.randn(
        n_hidden) * np.sqrt(1 / n_hidden))

    std_init_1_noise = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out_noise = pm.floatX(np.random.rand(n_hidden))

    # If there are two hidden layers, then initialize weights for the second layer:
    if n_layers == 2:
        init_2 = pm.floatX(np.random.randn(
            n_hidden, n_hidden) * np.sqrt(1 / n_hidden))
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
            mu_prior_intercept = pm.Normal(
                "mu_prior_intercept", mu=0.0, sigma=1e3)
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
                act_1 = pm.math.tanh(pytensor.tensor.dot(
                    X[idx, :], weights_in_1[be]))
                if n_layers == 2:
                    act_2 = pm.math.tanh(
                        pytensor.tensor.dot(act_1, weights_1_2[be]))
                    y_hat = pytensor.tensor.set_subtensor(
                        y_hat[idx, 0],
                        intercepts[be] +
                        pytensor.tensor.dot(act_2, weights_2_out[be]),
                    )
                else:
                    y_hat = pytensor.tensor.set_subtensor(
                        y_hat[idx, 0],
                        intercepts[be] +
                        pytensor.tensor.dot(act_1, weights_2_out[be]),
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
                            pytensor.tensor.dot(
                                X[idx, :], weights_in_1_noise[be])
                        )
                        if n_layers == 2:
                            act_2_noise = pm.math.sigmoid(
                                pytensor.tensor.dot(
                                    act_1_noise, weights_1_2_noise[be])
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
                        sigma_y = pytensor.tensor.set_subtensor(
                            sigma_y[idx, 0], temp)

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
                sigma_noise = pm.Uniform(
                    "sigma_noise", lower=0, upper=2 * upper_bound)
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
                    alpha = pytensor.tensor.set_subtensor(
                        alpha[idx, 0], skewness[be])
        else:
            alpha = 0  # symmetrical normal distribution

        y_like = pm.SkewNormal(
            "y_like", mu=y_hat, sigma=sigma_y, alpha=alpha, observed=y
        )

    return model
