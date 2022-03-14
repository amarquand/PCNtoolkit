#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:23:15 2019

@author: seykia
"""

from __future__ import print_function
from __future__ import division

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


def bspline_fit(X, order, nknots):
    feature_num = X.shape[1]
    bsp_basis = []
    for i in range(feature_num):
        knots = np.linspace(X[:, i].min(), X[:, i].max(), nknots)
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
    :return:
    """
    X = theano.shared(X)
    y = theano.shared(y)

    with pm.Model() as model:

        # Make a param builder that will make the correct calls
        parb = ParamBuilder(model, X, y, batch_effects, trace, configs)

        # MU =========================================================================================================
        if configs['mu_linear']:
            # Make mu with an intercept and a slope
            mu = parb.make_linear_param('mu', non_default_params={'sigma_intercept_dist': 'igamma',
                                                                  'sigma_intercept_pars': (5, 6),
                                                                  'offset_intercept_dist': 'uniform',
                                                                  'offset_intercept_pars': (-0.1, 0.1)})
            # mu = parb.make_linear_param('mu', non_default_params = {'offset_intercept_pars':(-1,1),
            #                                                         'sigma_intercept_dist':'normal','sigma_intercept_pars':(0,1)})

        else:
            # This will probably never happen
            mu = parb.make_fixed_param('mu', dist='normal', params=(0.0, 1.), exponentiate=False)

        # sigma ========================================================================================================
        if configs['sigma_linear']:
            sigma = parb.make_linear_param('sigma', non_default_params={'mapfunc': 'softplus_epsilon'})
        else:
            sigma = parb.make_fixed_param('sigma', dist='igamma', params=(2, 3), exponentiate=False)

        if configs['likelihood'] == 'Normal':
            y_like = pm.Normal('y_like',
                               mu=mu.values,
                               sigma=sigma.values,
                               observed=y)
        else:
            # epsilon ==================================================================================================
            if configs['epsilon_linear']:
                epsilon = parb.make_linear_param('epsilon', non_default_params={'mu_intercept_pars': (0, 1),
                                                                                'mu_slope_pars': (0, 1)})
            else:
                epsilon = parb.make_fixed_param('epsilon', dist='normal', params=(0., 1.), exponentiate=False)

            # delta ==================================================================================================
            if configs['delta_linear']:
                delta = parb.make_linear_param('delta', non_default_params={'mu_intercept_pars': (0, 1),
                                                                            'mu_slope_pars': (0, 1)})
            else:
                delta = parb.make_fixed_param('delta', dist='igamma', params=(10, 11), exponentiate=False)

            if configs['likelihood'] == 'SHASHo':
                y_like = SHASHo('y_like',
                                mu=mu.values,
                                sigma=sigma.values,
                                epsilon=epsilon.values,
                                delta=delta.values,
                                observed=y)
            elif configs['likelihood'] == 'SHASHo2':
                y_like = SHASHo2('y_like',
                                 mu=mu.values,
                                 sigma=sigma.values,
                                 epsilon=epsilon.values,
                                 delta=delta.values,
                                 observed=y)
            elif configs['likelihood'] == 'SHASHb':
                y_like = SHASHb('y_like',
                                mu=mu.values,
                                sigma=sigma.values,
                                epsilon=epsilon.values,
                                delta=delta.values,
                                observed=y)
            else:
                print(f"Selected likelihood {configs['likelihood']} is invalid")

    return model


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
        STEP_METHODS = (
            NUTS,
            HamiltonianMC,
            Metropolis,
            BinaryMetropolis,
            BinaryGibbsMetropolis,
            Slice,
            CategoricalGibbsMetropolis,
        )
        :param m:
        :return:
        """
        samplermap = {'NUTS': NUTS, 'MH': Metropolis, 'Slice': Slice, 'HMC': HamiltonianMC}
        if self.configs['sampler'] == 'NUTS':
            step_kwargs = {'nuts': {'target_accept': self.configs['target_accept']}}
        else:
            step_kwargs = None
        # We are using MH as a fallback method here
        return pm.sampling.assign_step_methods(m, methods=[samplermap[self.configs['sampler']]] + [Metropolis],
                                               step_kwargs=step_kwargs)

    def __init__(self, configs):

        self.model_type = configs['type']
        self.configs = configs

    def get_modeler(self):
        return {'nn': nn_hbr}.get(self.model_type, hbr)

    def transform_X(self, X):
        if self.model_type == 'polynomial':
            X = create_poly_basis(X, self.configs['order'])
        elif self.model_type == 'bspline':
            self.bsp = bspline_fit(X, self.configs['order'], self.configs['nknots'])
            X = bspline_transform(X, self.bsp)
        return X

    def find_map(self, X, y, batch_effects):
        """ Function to estimate the model """
        X, y, batch_effects = expand_all(X, y, batch_effects)

        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:, i])))

        X = self.transform_X(X)
        modeler = self.get_modeler()
        with modeler(X, y, batch_effects, self.batch_effects_size, self.configs) as m:
            self.MAP = pm.find_MAP()
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
            step = self.get_step_methods(m)
            self.trace = pm.sample(draws=self.configs['n_samples'],
                                   tune=self.configs['n_tuning'],
                                   step=step,
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
            # step = self.get_step_methods(m)

            self.trace = pm.sample(self.configs['n_samples'],
                                   tune=self.configs['n_tuning'],
                                   # step=step,
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
        return modeler(X, y, batch_effects, self.batch_effects_size, self.configs)


class ParamBuilder:
    """
    This is just a relay class. It simplifies the construction of Parameterization classes.
    Makes the correctly parameterized calls to all the Parameter classes.
    """

    # TODO verify that this class is not part of the model
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
        self.y_shape = y.shape
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

    def make_fixed_param(self, name, dist, params, exponentiate):
        return FixedParam(name, self.trace, self.model, self.configs, self.y_shape, self.be_idx_tups,
                          self.batch_effects_size, dist, params, exponentiate)

    def make_linear_param(self, name, non_default_params):
        return LinearParam(name, self.trace, self.model, self.configs, non_default_params, self.batch_effects_size,
                           self.feature_num, self.y_shape, self.be_idx_tups, self.X)


class Random_Indexing_Switcher:
    """
    This enables indexing distributions, even if they are 1-dimensional. This allows us to treat the 1-dimensional
     distribution the same as multi-dimensional distribution. You could call it syntactic sugar?
    """

    def __init__(self, name, dist, config):
        self.name = name
        self.dist = dist
        self.config = config
        self.is_random = self.config[f'random_{self.name}']

    def __getitem__(self, idx):
        if self.is_random:
            return self.dist[idx]
        else:
            return self.dist


class Trace_Dist:
    """
    Build a distribution from the posterior if there is a trace, otherwise construct something with a simple prior.
    """

    def __init__(self, model, name, dist, params, trace, configs, shape=None):
        # this distmap needs to be put elsewhere
        distmap = {'normal': pm.Normal,
                   'hnormal': pm.HalfNormal,
                   'gamma': pm.Gamma,
                   'uniform': pm.Uniform,
                   'igamma': pm.InverseGamma,
                   'hcauchy': pm.HalfCauchy}
        with model as model:
            if trace is not None:
                if shape is not None:
                    int_dist = from_posterior(param=name,
                                              samples=trace[name],
                                              distribution=dist,
                                              freedom=configs['freedom'])
                    self.dist = int_dist.reshape(shape)
                else:
                    self.dist = from_posterior(param=name,
                                               samples=trace[name],
                                               distribution=dist,
                                               freedom=configs['freedom'])
            else:
                if shape is not None:
                    # Trying this
                    shape_prod = np.product(np.array(shape))
                    int_dist = distmap[dist](name, *params, shape=shape_prod)
                    self.dist = int_dist.reshape(shape)
                else:
                    self.dist = distmap[dist](name, *params)


class Random_Dist:
    """
    Create a multi-dimensional distribution when the 'random_X' keyword is true, and a 1-d dist when it is false.
    """

    def __init__(self, model, name, par_name, dist, params, shape, configs):
        # this distmap needs to be put elsewhere
        distmap = {'normal': pm.Normal,
                   'hnormal': pm.HalfNormal,
                   'gamma': pm.Gamma,
                   'uniform': pm.Uniform,
                   'igamma': pm.InverseGamma,
                   'hcauchy': pm.HalfCauchy}
        with model as model:
            if configs[f'random_{par_name}']:
                # This reshape was necessary because the pm MH sampler didn't like non-flat dists.
                shape_prod = np.product(np.array(shape))
                int_dist = distmap[dist](name, *params, shape=shape_prod)
                self.shape = shape
                self.dist = int_dist.reshape(shape)

            else:
                self.dist = distmap[dist](name, *params)
                self.shape = (1,)


class Parameterization:

    def __init__(self, name, trace, model, configs):
        self.name = name
        self.trace = trace
        self.model = model
        self.configs = configs
        self.values = None

    def get_values(self):
        return self.values


class FixedParam(Parameterization):

    def __init__(self, name, trace, model, configs, y_shape, be_idx_tups, batch_effects_size, dist='normal',
                 params=(0, 2.5), exponentiate=False):
        """

        :param name: Name for the PyMC3 dist
        :param shape: Output shape
        :param trace:
        :param model: PyMC3 model
        :param configs:
        :param y_shape:
        :param be_idx_tups:
        :param batch_effects_size:
        :param dist:
        :param params: parameters of 'dist'
        :param exponentiate:
        """
        super().__init__(name, trace, model, configs)
        self.dist = dist
        self.params = params

        prefix = 'log_' if exponentiate else ''
        trace_dist_name = prefix + name

        if configs[f'random_{name}_intercept']:
            trace_dist_shape = batch_effects_size
        else:
            trace_dist_shape = None

        trace_dist = Trace_Dist(model, trace_dist_name, dist, params, trace, configs, trace_dist_shape)
        trace_dist = Random_Indexing_Switcher(f"{name}_intercept", trace_dist.dist, configs)
        with model as model:
            self.values = theano.tensor.zeros(y_shape)
            mapfunc = np.exp if exponentiate else (lambda x: x)
            for be, idx in be_idx_tups:
                self.values = theano.tensor.set_subtensor(self.values[idx, 0], mapfunc(trace_dist[be]))


class LinearParam(Parameterization):
    def __init__(self, name, trace, model, configs, non_default_params, batch_effects_size=None,
                 feature_num=None, y_shape=None, be_idx_tups=None, X=None):
        super().__init__(name, trace, model, configs)
        default_pars = {'mu_slope_dist': 'normal',
                        'mu_slope_pars': (0., 1.),
                        'sigma_slope_dist': 'igamma',
                        'sigma_slope_pars': (3., 4.),
                        'offset_slope_dist': 'uniform',
                        'offset_slope_pars': (-0.1, 0.1),
                        'mu_intercept_dist': 'normal',
                        'mu_intercept_pars': (0., 1.),
                        'sigma_intercept_dist': 'igamma',
                        'sigma_intercept_pars': (3., 4.),
                        'offset_intercept_dist': 'uniform',
                        'offset_intercept_pars': (-0.1, 0.1),
                        'mapfunc': 'id'
                        }
        mapfuncmap = {'softplus_epsilon': lambda x: pm.math.log1pexp(x) + 1e-5,
                      'id': (lambda x: x)}
        pars = default_pars
        for i in non_default_params.keys():
            pars[i] = non_default_params[i]

        ############################################## SLOPE ###################################################

        mu_prior_slope = Trace_Dist(name=f"mu_prior_slope_{name}",
                                    dist=pars['mu_slope_dist'],
                                    params=pars['mu_slope_pars'],
                                    shape=(feature_num,), model=model, trace=trace, configs=configs)

        if not self.configs[f'random_{name}_slope']:
            with model as model:
                slope = pm.Deterministic(f"slope_{name}", mu_prior_slope.dist)

        else:
            if not self.configs[f'centered_{name}_slope']:
                sigma_prior_slope = Trace_Dist(name=f"sigma_prior_slope_{name}",
                                               dist=pars['sigma_slope_dist'],
                                               params=pars['sigma_slope_pars'],
                                               shape=(feature_num,), model=model, trace=trace, configs=configs)
                offset_prior_slope = Random_Dist(name=f"offset_prior_slope_{name}",
                                                 par_name=f'{name}_slope',
                                                 dist=pars['offset_slope_dist'],
                                                 params=pars['offset_slope_pars'],
                                                 shape=batch_effects_size + [feature_num], model=model, configs=configs)
                with model as model:
                    slope = pm.Deterministic(f"slope_{name}", mu_prior_slope.dist + offset_prior_slope.dist *
                                             sigma_prior_slope.dist)
            else:
                sigma_prior_slope = Random_Dist(name=f"sigma_prior_slope_{name}",
                                                par_name=f'{name}_slope',
                                                dist=pars['sigma_slope_dist'],
                                                params=pars['sigma_slope_pars'],
                                                shape=batch_effects_size + [feature_num], model=model, configs=configs)
                with model as model:
                    slope = pm.Normal(name=f"slope_{name}", mu=mu_prior_slope.dist, sigma=sigma_prior_slope.dist,
                                      shape=sigma_prior_slope.shape)

        slope = Random_Indexing_Switcher(f"{name}_slope", slope, configs)

        ############################################ INTERCEPT #################################################

        mu_prior_intercept = Trace_Dist(name=f"mu_prior_intercept_{name}",
                                        dist=pars['mu_intercept_dist'],
                                        params=pars['mu_intercept_pars'],
                                        shape=(1,),
                                        model=model, trace=trace, configs=configs)

        if not self.configs[f'random_{name}_intercept']:
            with model as model:
                intercept = pm.Deterministic(f"intercept_{name}", mu_prior_intercept.dist)

        else:
            if not self.configs[f'centered_{name}_intercept']:
                sigma_prior_intercept = Trace_Dist(name=f"sigma_prior_intercept_{name}",
                                                   dist=pars['sigma_intercept_dist'],
                                                   params=pars['sigma_intercept_pars'],
                                                   shape=(1,),
                                                   model=model, trace=trace, configs=configs)
                offset_prior_intercept = Random_Dist(name=f"offset_prior_intercept_{name}",
                                                     par_name=f'{name}_intercept',
                                                     dist=pars['offset_intercept_dist'],
                                                     params=pars['offset_intercept_pars'],
                                                     shape=batch_effects_size, model=model, configs=configs)

                with model as model:
                    intercept = pm.Deterministic(f"intercept_{name}",
                                                 mu_prior_intercept.dist + offset_prior_intercept.dist *
                                                 sigma_prior_intercept.dist)
            else:
                sigma_prior_intercept = Random_Dist(name=f"sigma_prior_intercept_{name}",
                                                    par_name=f'{name}_intercept',
                                                    dist=pars['sigma_intercept_dist'],
                                                    params=pars['sigma_intercept_pars'],
                                                    shape=batch_effects_size, model=model, configs=configs)
                with model as model:
                    intercept = pm.Normal(name=f"intercept_{name}",
                                          mu=mu_prior_intercept.dist,
                                          sigma=sigma_prior_intercept.dist,
                                          shape=sigma_prior_intercept.shape)
        intercept = Random_Indexing_Switcher(f"{name}_intercept", intercept, configs)

        ########################################### REGRESSION #################################################

        with model as model:
            self.values = theano.tensor.zeros(y_shape)
            mapfunc = mapfuncmap[pars['mapfunc']]
            for be, idx in be_idx_tups:
                self.values = theano.tensor.set_subtensor(self.values[idx, 0],
                                                          mapfunc(intercept[be] + theano.tensor.dot(X[idx, :],
                                                                                                    slope[be])))


def get_design_matrix(X, nm, basis="linear"):
    if basis == "bspline":
        Phi = bspline_transform(X, nm.hbr.bsp)
    elif basis == "polynomial":
        Phi = create_poly_basis(X, 3)
    else:
        Phi = X
    return Phi