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

class HLR:
    """Hierarchical Linear Regression for normative modeling

    Basic usage::

        model = HLR(age, site_id, gender, mri_voxel)
        trace = model.estimate()
        ys,s2 = model.predict(age, site_id, gender)

    where the variables are

    :param age: N-vector of ages for N subjects
    :param site_id: N-vector of site IDs for N subjects
    :param gender: N-vector of genders for N subjects
    :param mri_voxel: N-vector of one voxel values for N subjects

    :returns: * ys - predictive mean
              * s2 - predictive variance

    Written by S.M. Kia
    """

    def __init__(self, age, site_id, gender, y):
        self.site_num = len(np.unique(site_id))
        a = theano.shared(age)
        s = theano.shared(site_id)
        g = theano.shared(gender)
        with pm.Model() as model:
            # Priors
            mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e5)
            sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
            mu_prior_slope = pm.Normal('mu_prior_slope', mu=0., sigma=1e5)
            sigma_prior_slope = pm.HalfCauchy('sigma_prior_slope', 5)
            
            # Random intercepts
            intercepts = pm.Normal('intercepts', mu=mu_prior_intercept, sigma=sigma_prior_intercept, shape=self.site_num)
            
            # Random slopes
            slopes = pm.Normal('slopes', mu=mu_prior_slope, sigma=sigma_prior_slope, shape=2)
            
            # Model error
            sigma_y = pm.Uniform('sigma_y', lower=0, upper=100)
            
            # Expected value
            y_hat = intercepts[s] + a * slopes[g]
            
            # Data likelihood
            y_like = pm.Normal('y_like', mu=y_hat, sigma=sigma_y, observed=y)
            
        self.model = model

    def estimate(self):
        """ Function to estimate the model """
        with self.model:
            self.trace = pm.sample(1000, tune=1000)
        
        return self.trace

    def predict(self, age, site_id, gender):
        """ Function to make predictions from the model """
        a = theano.shared(age)
        s = theano.shared(site_id)
        g = theano.shared(gender)
        with self.model:
            a.set_value(age)
            s.set_value(site_id)
            g.set_value(gender)
            ppc = pm.sample_posterior_predictive(self.trace, samples=500, progressbar=True) 
        
        # Predicting on training set
        print('Done!')
        pred_mean = ppc['y_like'].mean(axis=0)
        pred_var = ppc['y_like'].var(axis=0)
        
        return pred_mean, pred_var
