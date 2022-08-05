from __future__ import print_function
import os
import sys
import numpy as np
from scipy import stats
import scipy.special as spp
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
from pcntoolkit.model.SHASH import *
from pcntoolkit.model.hbr import bspline_transform

"""
@author: augub
"""

def MCMC_estimate(f, trace):
    """Get an MCMC estimate of f given a trace"""
    out = np.zeros_like(f(trace.point(0)))
    n=0
    for p in trace.points():
        out += f(p)
        n+=1
    return out/n


def get_MCMC_zscores(X, Y, Z, model):
    """Get an MCMC estimate of the z-scores of Y"""
    def f(sample):
         return get_single_zscores(X, Y, Z, model,sample)
    return MCMC_estimate(f, model.hbr.trace)


def get_single_zscores(X, Y, Z, model, sample):
    """Get the z-scores of y, given clinical covariates and a model"""
    likelihood = model.configs['likelihood']
    params = forward(X,Z,model,sample)
    return z_score(Y, params, likelihood = likelihood)
    

def z_score(Y, params, likelihood = "Normal"):
    """Get the z-scores of Y, given likelihood parameters"""
    if likelihood.startswith('SHASH'):
        mu = params['mu']
        sigma = params['sigma']
        epsilon = params['epsilon']
        delta = params['delta']
        if likelihood == "SHASHo":
            SHASH = (Y-mu)/sigma
            Z = np.sinh(np.arcsinh(SHASH)*delta - epsilon)
        elif likelihood == "SHASHo2":
            sigma_d = sigma/delta
            SHASH = (Y-mu)/sigma_d
            Z = np.sinh(np.arcsinh(SHASH)*delta - epsilon)
        elif likelihood == "SHASHb":
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu ** 2))
            SHASH_c = ((Y-mu)/sigma)
            SHASH = SHASH_c * true_sigma + true_mu
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
    elif likelihood == 'Normal':
        Z = (Y-params['mu'])/params['sigma']
    else:
        exit("Unsupported likelihood")
    return Z


def get_MCMC_quantiles(synthetic_X, z_scores, model, be):
    """Get an MCMC estimate of the quantiles"""
    """This does not use the get_single_quantiles function, for memory efficiency"""
    resolution = synthetic_X.shape[0]
    synthetic_X_transformed = model.hbr.transform_X(synthetic_X)
    be = np.reshape(np.array(be),(1,-1))
    synthetic_Z = np.repeat(be, resolution, axis = 0)
    z_scores = np.reshape(np.array(z_scores),(1,-1))
    zs = np.repeat(z_scores, resolution, axis=0)
    def f(sample):
        params = forward(synthetic_X_transformed,synthetic_Z, model,sample)
        q = quantile(zs, params, likelihood = model.configs['likelihood'])
        return q
    out = MCMC_estimate(f, model.hbr.trace)
    return out


def get_single_quantiles(synthetic_X, z_scores, model, be, sample):
    """Get the quantiles within a given range of covariates, given a model"""
    resolution = synthetic_X.shape[0]
    synthetic_X_transformed = model.hbr.transform_X(synthetic_X)
    be = np.reshape(np.array(be),(1,-1))
    synthetic_Z = np.repeat(be, resolution, axis = 0)
    z_scores = np.reshape(np.array(z_scores),(1,-1))
    zs = np.repeat(z_scores, resolution, axis=0)
    params = forward(synthetic_X_transformed,synthetic_Z, model,sample)
    q = quantile(zs, params, likelihood = model.configs['likelihood'])
    return q


def quantile(zs, params, likelihood = "Normal"):
    """Get the zs'th quantiles given likelihood parameters"""
    if likelihood.startswith('SHASH'):
        mu = params['mu']
        sigma = params['sigma']
        epsilon = params['epsilon']
        delta = params['delta']
        if likelihood == "SHASHo":
            quantiles = S_inv(zs,epsilon,delta)*sigma + mu
        elif likelihood == "SHASHo2":
            sigma_d = sigma/delta
            quantiles = S_inv(zs,epsilon,delta)*sigma_d + mu
        elif likelihood == "SHASHb":
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu ** 2))
            SHASH_c = ((S_inv(zs,epsilon,delta)-true_mu)/true_sigma)
            quantiles = SHASH_c *sigma + mu
    elif likelihood == 'Normal':
        quantiles = zs*params['sigma'] + params['mu']
    else:
        exit("Unsupported likelihood")
    return quantiles


def single_parameter_forward(X, Z, model, sample, p_name):
    """Get a likelihood paramameter given covariates, batch-effects and model parameters"""
    outs = np.zeros(X.shape[0])[:,None]
    all_bes = np.unique(Z,axis=0)
    for be in all_bes:
        bet = tuple(be)
        idx = (Z==be).all(1)
        if model.configs[f"linear_{p_name}"]:
            if model.configs[f'random_slope_{p_name}']:
                slope_be = sample[f"slope_{p_name}"][bet]
            else:
                slope_be = sample[f"slope_{p_name}"]
            if model.configs[f'random_intercept_{p_name}']:
                intercept_be = sample[f"intercept_{p_name}"][bet]
            else:
                intercept_be = sample[f"intercept_{p_name}"]

            out = (X[np.squeeze(idx),:]@slope_be)[:,None] + intercept_be
            outs[np.squeeze(idx),:] = out
        else:
            if model.configs[f'random_{p_name}']:
                outs[np.squeeze(idx),:] = sample[p_name][bet]
            else:
                outs[np.squeeze(idx),:] = sample[p_name]

    return outs


def forward(X, Z, model, sample):
    """Get all likelihood paramameters given covariates and batch-effects and model parameters"""
    # TODO think if this is the correct spot for this
    mapfuncs={'sigma': lambda x: np.log(1+np.exp(x)), 'delta':lambda x :np.log(1+np.exp(x)) + 0.3}

    likelihood = model.configs['likelihood']

    if likelihood == 'Normal':
        parameter_list = ['mu','sigma']
    elif likelihood in ['SHASHb','SHASHo','SHASHo2']:
        parameter_list = ['mu','sigma','epsilon','delta']
    else:
        exit("Unsupported likelihood")

    for i in parameter_list:
        if not (i in mapfuncs.keys()):
            mapfuncs[i] = lambda x: x

    output_dict = {p_name:np.zeros(X.shape) for p_name in parameter_list}

    for p_name in parameter_list:
        output_dict[p_name] = mapfuncs[p_name](single_parameter_forward(X,Z,model,sample,p_name))

    return output_dict


def Rhats(model, thin = 1, resolution = 100, varnames = None):
    """Get Rhat as function of sampling iteration"""
    trace = model.hbr.trace

    if varnames == None:
        varnames = trace.varnames
    chain_length = trace.get_values(varnames[0],chains=trace.chains[0], thin=thin).shape[0]
    
    interval_skip=chain_length//resolution

    rhat_dict = {}

    for varname in varnames:
        testvar = np.stack(trace.get_values(varname,combine=False))
        vardim = testvar.reshape((testvar.shape[0], testvar.shape[1], -1)).shape[2]
        rhats_var = np.zeros((resolution, vardim))

        var = np.stack(trace.get_values(varname,combine=False))
        var = var.reshape((var.shape[0], var.shape[1], -1))    
        for v in range(var.shape[2]):
            for j in range(resolution):
                rhats_var[j,v] = pm.rhat(var[:,:j*interval_skip,v])
        rhat_dict[varname] = rhats_var
    return rhat_dict


def S_inv(x, e, d):
    return np.sinh((np.arcsinh(x) + e) / d)
    
def K(p, x):
    return np.array(spp.kv(p, x))

def P(q):
    """
    The P function as given in Jones et al.
    :param q:
    :return:
    """
    frac = np.exp(1 / 4) / np.sqrt(8 * np.pi)
    K1 = K((q + 1) / 2, 1 / 4)
    K2 = K((q - 1) / 2, 1 / 4)
    a = (K1 + K2) * frac
    return a

def m(epsilon, delta, r):
    """
    The r'th uncentered moment. Given by Jones et al.
    """
    frac1 = 1 / np.power(2, r)
    acc = 0
    for i in range(r + 1):
        combs = spp.comb(r, i)
        flip = np.power(-1, i)
        ex = np.exp((r - 2 * i) * epsilon / delta)
        p = P((r - 2 * i) / delta)
        acc += combs * flip * ex * p
    return frac1 * acc



