#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:23:15 2019

@author: seykia
@author: augub
"""

from __future__ import print_function
from __future__ import division
from ast import Param
from tkinter.font import names

import numpy as np
import pymc3 as pm
import theano
from itertools import product
from functools import reduce

import theano
from pymc3 import Metropolis, NUTS, Slice, HamiltonianMC
from scipy import stats
import bspline
from bspline import splinelab

from model.SHASH import SHASHo2, SHASHb, SHASHo
from util.utils import create_poly_basis
from util.utils import expand_all
from pcntoolkit.util.utils import cartesian_product

from theano import printing, function

def bspline_fit(X, order, nknots):
    feature_num = X.shape[1]
    bsp_basis = []

    for i in range(feature_num):
        minx = np.min(X[:,i])
        maxx = np.max(X[:,i])
        delta = maxx-minx
        # Expand range by 20% (10% on both sides)
        splinemin = minx-0.1*delta
        splinemax = maxx+0.1*delta
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
    """ compute a polynomial basis expansion of the specified order"""

    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
    Phi = np.zeros((X.shape[0], D * order))
    colid = np.arange(0, D)
    for d in range(1, order + 1):
        Phi[:, colid] = X ** d
        colid += D

    return Phi


def from_posterior(param, samples, distribution=None, half=False, freedom=1):
    if len(samples.shape) > 1:
        shape = samples.shape[1:]
    else:
        shape = None

    if (distribution is None):
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
    elif (distribution == 'normal'):
        temp = stats.norm.fit(samples)
        if shape is None:
            return pm.Normal(param, mu=temp[0], sigma=freedom * temp[1])
        else:
            return pm.Normal(param, mu=temp[0], sigma=freedom * temp[1], shape=shape)
    elif (distribution == 'hnormal'):
        temp = stats.halfnorm.fit(samples)
        if shape is None:
            return pm.HalfNormal(param, sigma=freedom * temp[1])
        else:
            return pm.HalfNormal(param, sigma=freedom * temp[1], shape=shape)
    elif (distribution == 'hcauchy'):
        temp = stats.halfcauchy.fit(samples)
        if shape is None:
            return pm.HalfCauchy(param, freedom * temp[1])
        else:
            return pm.HalfCauchy(param, freedom * temp[1], shape=shape)
    elif (distribution == 'uniform'):
        upper_bound = np.percentile(samples, 95)
        lower_bound = np.percentile(samples, 5)
        r = np.abs(upper_bound - lower_bound)
        if shape is None:
            return pm.Uniform(param, lower=lower_bound - freedom * r,
                              upper=upper_bound + freedom * r)
        else:
            return pm.Uniform(param, lower=lower_bound - freedom * r,
                              upper=upper_bound + freedom * r, shape=shape)
    elif (distribution == 'huniform'):
        upper_bound = np.percentile(samples, 95)
        lower_bound = np.percentile(samples, 5)
        r = np.abs(upper_bound - lower_bound)
        if shape is None:
            return pm.Uniform(param, lower=0, upper=upper_bound + freedom * r)
        else:
            return pm.Uniform(param, lower=0, upper=upper_bound + freedom * r, shape=shape)

    elif (distribution == 'gamma'):
        alpha_fit, loc_fit, invbeta_fit = stats.gamma.fit(samples)
        if shape is None:
            return pm.Gamma(param, alpha=freedom * alpha_fit, beta=freedom / invbeta_fit)
        else:
            return pm.Gamma(param, alpha=freedom * alpha_fit, beta=freedom / invbeta_fit, shape=shape)

    elif (distribution == 'igamma'):
        alpha_fit, loc_fit, beta_fit = stats.gamma.fit(samples)
        if shape is None:
            return pm.InverseGamma(param, alpha=freedom * alpha_fit, beta=freedom * beta_fit)
        else:
            return pm.InverseGamma(param, alpha=freedom * alpha_fit, beta=freedom * beta_fit, shape=shape)


def hbr(X, y, batch_effects, batch_effects_size, configs, trace=None):
    """
    :param X: [N×P] array of clinical covariates
    :param y: [N×1] array of neuroimaging measures
    :param batch_effects: [N×M] array of batch effects
    :param batch_effects_size: [b1, b2,...,bM] List of counts of unique values of batch effects
    :param configs:
    :param trace:
    :param return_shared_variables: If true, returns references to the shared variables. The values of the shared variables can be set manually, allowing running the same model on different data without re-compiling it. 
    :return:
    """
    X = theano.shared(X)
    X = theano.tensor.cast(X,'floatX')
    y = theano.shared(y)
    y = theano.tensor.cast(y,'floatX')


    with pm.Model() as model:

        # Make a param builder that will make the correct calls
        pb = ParamBuilder(model, X, y, batch_effects, trace, configs)

        if configs['likelihood'] == 'Normal':
            mu = pb.make_param("mu").get_samples(pb)
            sigma = pb.make_param("sigma").get_samples(pb)
            sigma_plus = pm.math.log(1+pm.math.exp(sigma))
            y_like = pm.Normal('y_like',mu=mu, sigma=sigma_plus, observed=y)

        elif configs['likelihood'] in ['SHASHb','SHASHo','SHASHo2']:
            """
            Comment 1
            The current parameterizations are tuned towards standardized in- and output data. 
            It is possible to adjust the priors through the XXX_dist and XXX_params kwargs, like here we do with epsilon_params.
            Supported distributions are listed in the Prior class. 

            Comment 2
            Any mapping that is applied here after sampling should also be applied in util.hbr_utils.forward in order for the functions there to properly work. 
            For example, the softplus applied to sigma here is also applied in util.hbr_utils.forward
            """
            SHASH_map = {'SHASHb':SHASHb,'SHASHo':SHASHo,'SHASHo2':SHASHo2}

            mu =            pb.make_param("mu",         slope_mu_params = (0.,3.), mu_intercept_mu_params=(0.,1.), sigma_intercept_mu_params = (1.,)).get_samples(pb)
            sigma =         pb.make_param("sigma",      sigma_params = (1.,2.),    slope_sigma_params=(0.,1.),     intercept_sigma_params = (1., 1.)).get_samples(pb)
            sigma_plus =    pm.math.log(1+pm.math.exp(sigma))
            epsilon =       pb.make_param("epsilon",    epsilon_params = (0.,1.),  slope_epsilon_params=(0.,1.), intercept_epsilon_params=(0.,1)).get_samples(pb)
            delta =         pb.make_param("delta",      delta_params=(1.5,2.),     slope_delta_params=(0.,1),   intercept_delta_params=(2., 1)).get_samples(pb)
            delta_plus =    pm.math.log(1+pm.math.exp(delta)) + 0.3
            y_like = SHASH_map[configs['likelihood']]('y_like', mu=mu, sigma=sigma_plus, epsilon=epsilon, delta=delta_plus, observed = y)

    return model



class HBR:

    """Hierarchical Bayesian Regression for normative modeling

    Basic usage::

        model = HBR(configs)
        trace = model.estimate(X, y, batch_effects)
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

    def get_step_methods(self, m):
        """
        This can be used to assign default step functions. However, the nuts initialization keyword doesnt work together with this... so better not use it. 

        STEP_METHODS = (
            NUTS,
            HamiltonianMC,
            Metropolis,
            BinaryMetropolis,
            BinaryGibbsMetropolis,
            Slice,
            CategoricalGibbsMetropolis,
        )
        :param m: a PyMC3 model
        :return:
        """
        samplermap = {'NUTS': NUTS, 'MH': Metropolis, 'Slice': Slice, 'HMC': HamiltonianMC}
        fallbacks = [Metropolis]         # We are using MH as a fallback method here
        if self.configs['sampler'] == 'NUTS':
            step_kwargs = {'nuts': {'target_accept': self.configs['target_accept']}}
        else:
            step_kwargs = None
        return pm.sampling.assign_step_methods(m, methods=[samplermap[self.configs['sampler']]] + fallbacks,
                                               step_kwargs=step_kwargs)

    def __init__(self, configs):
        self.bsp = None
        self.model_type = configs['type']
        self.configs = configs

    def get_modeler(self):
        return {'nn': nn_hbr}.get(self.model_type, hbr)
        
    def transform_X(self, X):
        if self.model_type == 'polynomial':
            Phi = create_poly_basis(X, self.configs['order'])
        elif self.model_type == 'bspline':
            if self.bsp is None:
                self.bsp = bspline_fit(X, self.configs['order'], self.configs['nknots'])
            bspline = bspline_transform(X, self.bsp)
            Phi = np.concatenate((X, bspline), axis = 1)
        else:
            Phi = X
        return Phi


    def find_map(self, X, y, batch_effects,method='L-BFGS-B'):
        """ Function to estimate the model """
        X, y, batch_effects = expand_all(X, y, batch_effects)

        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:, i])))

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.batch_effects_size, self.configs) as m:
            self.MAP = pm.find_MAP(method=method)
        return self.MAP

    def estimate(self, X, y, batch_effects):

        """ Function to estimate the model """
        X, y, batch_effects = expand_all(X, y, batch_effects)

        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:, i])))

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.batch_effects_size, self.configs) as m:
            self.trace = pm.sample(draws=self.configs['n_samples'],
                                   tune=self.configs['n_tuning'],
                                   chains=self.configs['n_chains'],
                                   init=self.configs['init'], n_init=500000,
                                   cores=self.configs['cores'])
        return self.trace

    def predict(self, X, batch_effects, pred='single'):
        """ Function to make predictions from the model """
        X, batch_effects = expand_all(X, batch_effects)

        samples = self.configs['n_samples']
        y = np.zeros([X.shape[0], 1])

        if pred == 'single':
            X = self.transform_X(X)
            modeler = self.get_modeler()
            with modeler(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True)
            pred_mean = ppc['y_like'].mean(axis=0)
            pred_var = ppc['y_like'].var(axis=0)

        return pred_mean, pred_var

    def estimate_on_new_site(self, X, y, batch_effects):
        """ Function to adapt the model """
        X, y, batch_effects = expand_all(X, y, batch_effects)

        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:, i])))

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.batch_effects_size,
                     self.configs, trace=self.trace) as m:
            self.trace = pm.sample(self.configs['n_samples'],
                                   tune=self.configs['n_tuning'],
                                   chains=self.configs['n_chains'],
                                   target_accept=self.configs['target_accept'],
                                   init=self.configs['init'], n_init=50000,
                                   cores=self.configs['cores'])
        return self.trace

    def predict_on_new_site(self, X, batch_effects):
        """ Function to make predictions from the model """
        X, batch_effects = expand_all(X, batch_effects)

        samples = self.configs['n_samples']
        y = np.zeros([X.shape[0], 1])

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.batch_effects_size, self.configs, trace=self.trace):
            ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True)
        pred_mean = ppc['y_like'].mean(axis=0)
        pred_var = ppc['y_like'].var(axis=0)

        return pred_mean, pred_var

    def generate(self, X, batch_effects, samples):
        """ Function to generate samples from posterior predictive distribution """
        X, batch_effects = expand_all(X, batch_effects)

        y = np.zeros([X.shape[0], 1])

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.batch_effects_size, self.configs):
            ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True)

        generated_samples = np.reshape(ppc['y_like'].squeeze().T, [X.shape[0] * samples, 1])
        X = np.repeat(X, samples)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        batch_effects = np.repeat(batch_effects, samples, axis=0)
        if len(batch_effects.shape) == 1:
            batch_effects = np.expand_dims(batch_effects, axis=1)

        return X, batch_effects, generated_samples

    def sample_prior_predictive(self, X, batch_effects, samples, trace=None):
        """ Function to sample from prior predictive distribution """

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape) == 1:
            batch_effects = np.expand_dims(batch_effects, axis=1)

        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:, i])))

        y = np.zeros([X.shape[0], 1])

        if self.model_type == 'linear':
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs,
                     trace):
                ppc = pm.sample_prior_predictive(samples=samples)
        return ppc

    def get_model(self, X, y, batch_effects):
        X, y, batch_effects = expand_all(X, y, batch_effects)

        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:, i])))
        modeler = self.get_modeler()
        X = self.transform_X(X)
        return modeler(X, y, batch_effects, self.batch_effects_size, self.configs, self.trace)


class Prior:
    """
    A wrapper class for a PyMC3 distribution. 
    - creates a fitted distribution from the trace, if one is present
    - overloads the __getitem__ function with something that switches between indexing or not, based on the shape
    """
    def __init__(self, name, dist, params, pb, shape=(1,)) -> None:
        self.dist = None
        self.name = name
        self.shape = shape
        self.has_random_effect = True if len(shape)>1 else False
        self.distmap = {'normal': pm.Normal,
                   'hnormal': pm.HalfNormal,
                   'gamma': pm.Gamma,
                   'uniform': pm.Uniform,
                   'igamma': pm.InverseGamma,
                   'hcauchy': pm.HalfCauchy}
        self.make_dist(dist, params, pb)
 
    def make_dist(self, dist, params, pb):
        """This creates a pymc3 distribution. If there is a trace, the distribution is fitted to the trace. If there isn't a trace, the prior is parameterized by the values in (params)"""
        with pb.model as m:
            if (pb.trace is not None) and (not self.has_random_effect):
                int_dist = from_posterior(param=self.name,
                                            samples=pb.trace[self.name],
                                            distribution=dist,
                                            freedom=pb.configs['freedom'])
                self.dist = int_dist.reshape(self.shape)
            else:
                shape_prod = np.product(np.array(self.shape))
                print(self.name)
                print(f"dist={dist}")
                print(f"params={params}")
                int_dist = self.distmap[dist](self.name, *params, shape=shape_prod)
                self.dist = int_dist.reshape(self.shape)

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

    def __init__(self, model, X, y, batch_effects, trace, configs):
        """

        :param model: model to attach all the distributions to
        :param X: Covariates
        :param y: IDPs
        :param batch_effects: I guess this speaks for itself
        :param trace:  idem
        :param configs: idem
        """
        self.model = model
        self.X = X
        self.y = y
        self.batch_effects = batch_effects
        self.trace = trace
        self.configs = configs

        self.feature_num = X.shape[1].eval().item()
        self.y_shape = y.shape.eval()
        self.n_ys = y.shape[0].eval().item()
        self.batch_effects_num = batch_effects.shape[1]

        self.batch_effects_size = []
        self.all_idx = []
        for i in range(self.batch_effects_num):
            # Count the unique values for each batch effect
            self.batch_effects_size.append(len(np.unique(self.batch_effects[:, i])))
            # Store the unique values for each batch effect
            self.all_idx.append(np.int16(np.unique(self.batch_effects[:, i])))
        # Make a cartesian product of all the unique values of each batch effect
        self.be_idx = list(product(*self.all_idx))

        # Make tuples of batch effects ID's and indices of datapoints with that specific combination of batch effects
        self.be_idx_tups = []
        for be in self.be_idx:
            a = []
            for i, b in enumerate(be):
                a.append(self.batch_effects[:, i] == b)
            idx = reduce(np.logical_and, a).nonzero()
            if idx[0].shape[0] != 0:
                self.be_idx_tups.append((be, idx))

    def make_param(self, name, dim = (1,), **kwargs):
        if self.configs.get(f'linear_{name}', False):
            # First make a slope and intercept, and use those to make a linear parameterization 
            slope_parameterization = self.make_param(f'slope_{name}', dim=[self.feature_num], **kwargs)
            intercept_parameterization = self.make_param(f'intercept_{name}', **kwargs)
            return LinearParameterization(name=name, dim=dim, 
                                    slope_parameterization=slope_parameterization, intercept_parameterization=intercept_parameterization,
                                    pb=self, 
                                    **kwargs)
        
        elif self.configs.get(f'random_{name}', False):
            if self.configs.get(f'centered_{name}', True):
                return CentralRandomFixedParameterization(name=name, pb=self, dim=dim, **kwargs)
            else:
                return NonCentralRandomFixedParameterization(name=name, pb=self, dim=dim, **kwargs)
        else:
            return FixedParameterization(name=name, dim=dim, pb=self,**kwargs)


class Parameterization:
    """
    This is the top-level parameterization class from which all the other parameterizations inherit.
    """
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim
        print(name, type(self))

    def get_samples(self, pb):

        with pb.model:
            samples = theano.tensor.zeros([pb.n_ys, *self.dim])
            for be, idx in pb.be_idx_tups:
                samples = theano.tensor.set_subtensor(samples[idx], self.dist[be])
        return samples


class FixedParameterization(Parameterization):
    """
    A parameterization that takes a single value for all input. It does not depend on anything except its hyperparameters
    """
    def __init__(self, name, dim, pb:ParamBuilder, **kwargs):
        super().__init__(name, dim)
        dist = kwargs.get(f'{name}_dist','normal')
        params = kwargs.get(f'{name}_params',(0.,1.))
        self.dist = Prior(name, dist, params, pb, shape = dim)


class CentralRandomFixedParameterization(Parameterization):
    """
    A parameterization that is fixed for each batch effect. This is sampled in a central fashion;
    the values are sampled from normal distribution with a group mean and group variance 
    """
    def __init__(self, name, dim, pb:ParamBuilder, **kwargs):
        super().__init__(name, dim)

        # Normal distribution is default for mean
        mu_dist = kwargs.get(f'mu_{name}_dist','normal')
        mu_params = kwargs.get(f'mu_{name}_params',(0.,1.))
        mu_prior = Prior(f'mu_{name}', mu_dist, mu_params, pb, shape = dim)

        # HalfCauchy is default for sigma
        sigma_dist = kwargs.get(f'sigma_{name}_dist','hcauchy')
        sigma_params = kwargs.get(f'sigma_{name}_params',(1.,))
        sigma_prior = Prior(f'sigma_{name}',sigma_dist, sigma_params, pb, shape = [*pb.batch_effects_size, *dim])

        self.dist = pm.Normal(name=name, mu=mu_prior.dist, sigma=sigma_prior.dist, shape = [*pb.batch_effects_size, *dim])
    

class NonCentralRandomFixedParameterization(Parameterization):
    """
    A parameterization that is fixed for each batch effect. This is sampled in a non-central fashion;
    the values are a sum of a group mean and noise values scaled with a group scaling factor 
    """
    def __init__(self, name,dim,  pb:ParamBuilder, **kwargs):
        super().__init__(name, dim)

        # Normal distribution is default for mean
        mu_dist = kwargs.get(f'mu_{name}_dist','normal')
        mu_params = kwargs.get(f'mu_{name}_params',(0.,1.))
        mu_prior = Prior(f'mu_{name}', mu_dist, mu_params, pb, shape = dim)

        # HalfCauchy is default for sigma
        sigma_dist = kwargs.get(f'sigma_{name}_dist','hcauchy')
        sigma_params = kwargs.get(f'sigma_{name}_params',(1.,))
        sigma_prior = Prior(f'sigma_{name}',sigma_dist, sigma_params, pb, shape = dim)

        # Normal is default for offset
        offset_dist = kwargs.get(f'offset_{name}_dist','normal')
        offset_params = kwargs.get(f'offset_{name}_params',(0.,1.))
        offset_prior = Prior(f'offset_{name}',offset_dist, offset_params, pb, shape = [*pb.batch_effects_size, *dim])

        self.dist = pm.Deterministic(name=name, var=mu_prior.dist+sigma_prior.dist*offset_prior.dist)


class LinearParameterization(Parameterization):
    """
    A parameterization that can model a linear dependence on X. 
    """
    def __init__(self, name, dim, slope_parameterization, intercept_parameterization, pb, **kwargs):
        super().__init__( name, dim)
        self.slope_parameterization = slope_parameterization
        self.intercept_parameterization = intercept_parameterization

    def get_samples(self, pb:ParamBuilder):
        with pb.model:
            samples = theano.tensor.zeros([pb.n_ys, *self.dim])
            for be, idx in pb.be_idx_tups:
                dot = theano.tensor.dot(pb.X[idx,:], self.slope_parameterization.dist[be]).T
                intercept = self.intercept_parameterization.dist[be]
                samples = theano.tensor.set_subtensor(samples[idx,:],dot+intercept)
        return samples


def get_design_matrix(X, nm, basis="linear"):
    if basis == "bspline":
        Phi = bspline_transform(X, nm.hbr.bsp)
    elif basis == "polynomial":
        Phi = create_poly_basis(X, 3)
    else:
        Phi = X
    return Phi



def nn_hbr(X, y, batch_effects, batch_effects_size, configs, trace=None):
    n_hidden = configs['nn_hidden_neuron_num']
    n_layers = configs['nn_hidden_layers_num']
    feature_num = X.shape[1]
    batch_effects_num = batch_effects.shape[1]
    all_idx = []
    for i in range(batch_effects_num):
        all_idx.append(np.int16(np.unique(batch_effects[:, i])))
    be_idx = list(product(*all_idx))

    # Initialize random weights between each layer for the mu:
    init_1 = pm.floatX(np.random.randn(feature_num, n_hidden) * np.sqrt(1 / feature_num))
    init_out = pm.floatX(np.random.randn(n_hidden) * np.sqrt(1 / n_hidden))

    std_init_1 = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out = pm.floatX(np.random.rand(n_hidden))

    # And initialize random weights between each layer for sigma_noise:
    init_1_noise = pm.floatX(np.random.randn(feature_num, n_hidden) * np.sqrt(1 / feature_num))
    init_out_noise = pm.floatX(np.random.randn(n_hidden) * np.sqrt(1 / n_hidden))

    std_init_1_noise = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out_noise = pm.floatX(np.random.rand(n_hidden))

    # If there are two hidden layers, then initialize weights for the second layer:
    if n_layers == 2:
        init_2 = pm.floatX(np.random.randn(n_hidden, n_hidden) * np.sqrt(1 / n_hidden))
        std_init_2 = pm.floatX(np.random.rand(n_hidden, n_hidden))
        init_2_noise = pm.floatX(np.random.randn(n_hidden, n_hidden) * np.sqrt(1 / n_hidden))
        std_init_2_noise = pm.floatX(np.random.rand(n_hidden, n_hidden))

    with pm.Model() as model:

        X = pm.Data('X', X)
        y = pm.Data('y', y)

        if trace is not None:  # Used when estimating/predicting on a new site
            weights_in_1_grp = from_posterior('w_in_1_grp', trace['w_in_1_grp'],
                                              distribution='normal', freedom=configs['freedom'])

            weights_in_1_grp_sd = from_posterior('w_in_1_grp_sd', trace['w_in_1_grp_sd'],
                                                 distribution='hcauchy', freedom=configs['freedom'])

            if n_layers == 2:
                weights_1_2_grp = from_posterior('w_1_2_grp', trace['w_1_2_grp'],
                                                 distribution='normal', freedom=configs['freedom'])

                weights_1_2_grp_sd = from_posterior('w_1_2_grp_sd', trace['w_1_2_grp_sd'],
                                                    distribution='hcauchy', freedom=configs['freedom'])

            weights_2_out_grp = from_posterior('w_2_out_grp', trace['w_2_out_grp'],
                                               distribution='normal', freedom=configs['freedom'])

            weights_2_out_grp_sd = from_posterior('w_2_out_grp_sd', trace['w_2_out_grp_sd'],
                                                  distribution='hcauchy', freedom=configs['freedom'])

            mu_prior_intercept = from_posterior('mu_prior_intercept', trace['mu_prior_intercept'],
                                                distribution='normal', freedom=configs['freedom'])
            sigma_prior_intercept = from_posterior('sigma_prior_intercept', trace['sigma_prior_intercept'],
                                                   distribution='hcauchy', freedom=configs['freedom'])

        else:
            # Group the mean distribution for input to the hidden layer:
            weights_in_1_grp = pm.Normal('w_in_1_grp', 0, sd=1,
                                         shape=(feature_num, n_hidden), testval=init_1)

            # Group standard deviation:
            weights_in_1_grp_sd = pm.HalfCauchy('w_in_1_grp_sd', 1.,
                                                shape=(feature_num, n_hidden), testval=std_init_1)

            if n_layers == 2:
                # Group the mean distribution for hidden layer 1 to hidden layer 2:
                weights_1_2_grp = pm.Normal('w_1_2_grp', 0, sd=1,
                                            shape=(n_hidden, n_hidden), testval=init_2)

                # Group standard deviation:
                weights_1_2_grp_sd = pm.HalfCauchy('w_1_2_grp_sd', 1.,
                                                   shape=(n_hidden, n_hidden), testval=std_init_2)

            # Group the mean distribution for hidden to output:
            weights_2_out_grp = pm.Normal('w_2_out_grp', 0, sd=1,
                                          shape=(n_hidden,), testval=init_out)

            # Group standard deviation:
            weights_2_out_grp_sd = pm.HalfCauchy('w_2_out_grp_sd', 1.,
                                                 shape=(n_hidden,), testval=std_init_out)

            # mu_prior_intercept = pm.Uniform('mu_prior_intercept', lower=-100, upper=100)
            mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e3)
            sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)

        # Now create separate weights for each group, by doing
        # weights * group_sd + group_mean, we make sure the new weights are
        # coming from the (group_mean, group_sd) distribution.
        weights_in_1_raw = pm.Normal('w_in_1', 0, sd=1,
                                     shape=(batch_effects_size + [feature_num, n_hidden]))
        weights_in_1 = weights_in_1_raw * weights_in_1_grp_sd + weights_in_1_grp

        if n_layers == 2:
            weights_1_2_raw = pm.Normal('w_1_2', 0, sd=1,
                                        shape=(batch_effects_size + [n_hidden, n_hidden]))
            weights_1_2 = weights_1_2_raw * weights_1_2_grp_sd + weights_1_2_grp

        weights_2_out_raw = pm.Normal('w_2_out', 0, sd=1,
                                      shape=(batch_effects_size + [n_hidden]))
        weights_2_out = weights_2_out_raw * weights_2_out_grp_sd + weights_2_out_grp

        intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1,
                                      shape=(batch_effects_size))

        intercepts = pm.Deterministic('intercepts', intercepts_offset +
                                      mu_prior_intercept * sigma_prior_intercept)

        # Build the neural network and estimate y_hat:
        y_hat = theano.tensor.zeros(y.shape)
        for be in be_idx:
            # Find the indices corresponding to 'group be':
            a = []
            for i, b in enumerate(be):
                a.append(batch_effects[:, i] == b)
            idx = reduce(np.logical_and, a).nonzero()
            if idx[0].shape[0] != 0:
                act_1 = pm.math.tanh(theano.tensor.dot(X[idx, :], weights_in_1[be]))
                if n_layers == 2:
                    act_2 = pm.math.tanh(theano.tensor.dot(act_1, weights_1_2[be]))
                    y_hat = theano.tensor.set_subtensor(y_hat[idx, 0],
                                                        intercepts[be] + theano.tensor.dot(act_2, weights_2_out[be]))
                else:
                    y_hat = theano.tensor.set_subtensor(y_hat[idx, 0],
                                                        intercepts[be] + theano.tensor.dot(act_1, weights_2_out[be]))

        # If we want to estimate varying noise terms across groups:
        if configs['random_noise']:
            if configs['hetero_noise']:
                if trace is not None:  # # Used when estimating/predicting on a new site
                    weights_in_1_grp_noise = from_posterior('w_in_1_grp_noise',
                                                            trace['w_in_1_grp_noise'],
                                                            distribution='normal', freedom=configs['freedom'])

                    weights_in_1_grp_sd_noise = from_posterior('w_in_1_grp_sd_noise',
                                                               trace['w_in_1_grp_sd_noise'],
                                                               distribution='hcauchy', freedom=configs['freedom'])

                    if n_layers == 2:
                        weights_1_2_grp_noise = from_posterior('w_1_2_grp_noise',
                                                               trace['w_1_2_grp_noise'],
                                                               distribution='normal', freedom=configs['freedom'])

                        weights_1_2_grp_sd_noise = from_posterior('w_1_2_grp_sd_noise',
                                                                  trace['w_1_2_grp_sd_noise'],
                                                                  distribution='hcauchy', freedom=configs['freedom'])

                    weights_2_out_grp_noise = from_posterior('w_2_out_grp_noise',
                                                             trace['w_2_out_grp_noise'],
                                                             distribution='normal', freedom=configs['freedom'])

                    weights_2_out_grp_sd_noise = from_posterior('w_2_out_grp_sd_noise',
                                                                trace['w_2_out_grp_sd_noise'],
                                                                distribution='hcauchy', freedom=configs['freedom'])

                else:
                    # The input layer to the first hidden layer:
                    weights_in_1_grp_noise = pm.Normal('w_in_1_grp_noise', 0, sd=1,
                                                       shape=(feature_num, n_hidden),
                                                       testval=init_1_noise)
                    weights_in_1_grp_sd_noise = pm.HalfCauchy('w_in_1_grp_sd_noise', 1,
                                                              shape=(feature_num, n_hidden),
                                                              testval=std_init_1_noise)

                    # The first hidden layer to second hidden layer:
                    if n_layers == 2:
                        weights_1_2_grp_noise = pm.Normal('w_1_2_grp_noise', 0, sd=1,
                                                          shape=(n_hidden, n_hidden),
                                                          testval=init_2_noise)
                        weights_1_2_grp_sd_noise = pm.HalfCauchy('w_1_2_grp_sd_noise', 1,
                                                                 shape=(n_hidden, n_hidden),
                                                                 testval=std_init_2_noise)

                    # The second hidden layer to output layer:
                    weights_2_out_grp_noise = pm.Normal('w_2_out_grp_noise', 0, sd=1,
                                                        shape=(n_hidden,),
                                                        testval=init_out_noise)
                    weights_2_out_grp_sd_noise = pm.HalfCauchy('w_2_out_grp_sd_noise', 1,
                                                               shape=(n_hidden,),
                                                               testval=std_init_out_noise)

                    # mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', sigma=1e3)
                    # sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)

                # Now create separate weights for each group:
                weights_in_1_raw_noise = pm.Normal('w_in_1_noise', 0, sd=1,
                                                   shape=(batch_effects_size + [feature_num, n_hidden]))
                weights_in_1_noise = weights_in_1_raw_noise * weights_in_1_grp_sd_noise + weights_in_1_grp_noise

                if n_layers == 2:
                    weights_1_2_raw_noise = pm.Normal('w_1_2_noise', 0, sd=1,
                                                      shape=(batch_effects_size + [n_hidden, n_hidden]))
                    weights_1_2_noise = weights_1_2_raw_noise * weights_1_2_grp_sd_noise + weights_1_2_grp_noise

                weights_2_out_raw_noise = pm.Normal('w_2_out_noise', 0, sd=1,
                                                    shape=(batch_effects_size + [n_hidden]))
                weights_2_out_noise = weights_2_out_raw_noise * weights_2_out_grp_sd_noise + weights_2_out_grp_noise

                # intercepts_offset_noise = pm.Normal('intercepts_offset_noise', mu=0, sd=1,
                #                          shape=(batch_effects_size))

                # intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise +
                #                      intercepts_offset_noise * sigma_prior_intercept_noise)

                # Build the neural network and estimate the sigma_y:
                sigma_y = theano.tensor.zeros(y.shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:, i] == b)
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0] != 0:
                        act_1_noise = pm.math.sigmoid(theano.tensor.dot(X[idx, :], weights_in_1_noise[be]))
                        if n_layers == 2:
                            act_2_noise = pm.math.sigmoid(theano.tensor.dot(act_1_noise, weights_1_2_noise[be]))
                            temp = pm.math.log1pexp(theano.tensor.dot(act_2_noise, weights_2_out_noise[be])) + 1e-5
                        else:
                            temp = pm.math.log1pexp(theano.tensor.dot(act_1_noise, weights_2_out_noise[be])) + 1e-5
                        sigma_y = theano.tensor.set_subtensor(sigma_y[idx, 0], temp)

            else:  # homoscedastic noise:
                if trace is not None:  # Used for transferring the priors
                    upper_bound = np.percentile(trace['sigma_noise'], 95)
                    sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=2 * upper_bound, shape=(batch_effects_size))
                else:
                    sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100, shape=(batch_effects_size))

                sigma_y = theano.tensor.zeros(y.shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:, i] == b)
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0] != 0:
                        sigma_y = theano.tensor.set_subtensor(sigma_y[idx, 0], sigma_noise[be])

        else:  # do not allow for random noise terms across groups:
            if trace is not None:  # Used for transferring the priors
                upper_bound = np.percentile(trace['sigma_noise'], 95)
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=2 * upper_bound)
            else:
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100)
            sigma_y = theano.tensor.zeros(y.shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:, i] == b)
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0] != 0:
                    sigma_y = theano.tensor.set_subtensor(sigma_y[idx, 0], sigma_noise)

        if configs['skewed_likelihood']:
            skewness = pm.Uniform('skewness', lower=-10, upper=10, shape=(batch_effects_size))
            alpha = theano.tensor.zeros(y.shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:, i] == b)
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0] != 0:
                    alpha = theano.tensor.set_subtensor(alpha[idx, 0], skewness[be])
        else:
            alpha = 0  # symmetrical normal distribution

        y_like = pm.SkewNormal('y_like', mu=y_hat, sigma=sigma_y, alpha=alpha, observed=y)

    return model
