#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:01:24 2019

@author: seykia
@author: augub
"""

from __future__ import division, print_function

import os
import sys
from sys import exit

import arviz as az
import numpy as np
import xarray
from scipy import special as spp

try:
    from pcntoolkit.dataio import fileio
    from pcntoolkit.model.hbr import HBR
    from pcntoolkit.normative_model.norm_base import NormBase
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path
    import dataio.fileio as fileio
    from model.hbr import HBR
    from norm_base import NormBase


class NormHBR(NormBase):
    """HBR multi-batch normative modelling class. By default, this function
    estimates a linear model with random intercept, random slope, and random
    homoscedastic noise.

    :param X: [N×P] array of clinical covariates
    :param y: [N×1] array of neuroimaging measures
    :param trbefile: the address to the batch effects file for the training set.
        the batch effect array should be a [N×M] array where M is the number of
        the type of batch effects. For example when the site and gender is modeled
        as batch effects M=2. Each column in the batch effect array contains the
        batch ID (starting from 0) for each sample. If not specified (default=None)
        then all samples assumed to be from the same batch (i.e., the batch effect
                                                            is not modelled).
    :param tsbefile: Similar to trbefile for the test set.
    :param model_type: Specifies the type of the model from 'linear', 'plynomial',
        and 'bspline' (defauls is 'linear').
    :param likelihood: specifies the type of likelihood among 'Normal' 'SHASHb','SHASHo',
        and 'SHASHo2' (defauls is normal).
    :param linear_mu: Boolean (default='True') to decide whether the mean (mu) is
        parametrized on a linear function (thus changes with covariates) or it is fixed.
    :param linear_sigma: Boolean (default='False') to decide whether the variance (sigma) is
        parametrized on a linear function (heteroscedastic noise) or it is fixed for
        each batch (homoscedastic noise).
    :param linear_epsilon: Boolean (default='False') to decide the parametrization
        of epsilon for the SHASH likelihood that controls its skewness.
        If True, epsilon is  parametrized on a linear function
        (thus changes with covariates) otherwise it is fixed for each batch.
    :param linear_delta: Boolean (default='False') to decide the parametrization
        of delta for the SHASH likelihood that controls its kurtosis.
        If True, delta is  parametrized on a linear function
        (thus changes with covariates) otherwise it is fixed for each batch.
    :param random_intercept_{parameter}: if parameters mu (default='True'),
        sigma (default='False'), epsilon (default='False'), and delta (default='False')
        are parametrized on a linear function, then this boolean decides
        whether the intercept can vary across batches.
    :param random_slope_{parameter}: if parameters mu (default='True'),
        sigma (default='False'), epsilon (default='False'), and delta (default='False')
        are parametrized on a linear function, then this boolean decides
        whether the slope can vary across batches.
    :param centered_intercept_{parameter}: if parameters mu (default='False'),
        sigma (default='False'), epsilon (default='False'), and delta (default='False')
        are parametrized on a linear function, then this boolean decides
        whether the parameters of intercept are estimated in a centered or
        non-centered manner (default). While centered estimation runs faster
        it may cause some problems for the sampler (the funnel of hell).
    :param centered_slope_{parameter}: if parameters mu (default='False'),
        sigma (default='False'), epsilon (default='False'), and delta (default='False')
        are parametrized on a linear function, then this boolean decides
        whether the parameters of slope are estimated in a centered or
        non-centered manner (default). While centered estimation runs faster
        it may cause some problems for the sampler (the funnel of hell).
    :param sampler: specifies the type of PyMC sampler (Defauls is 'NUTS').
    :param n_samples: The number of samples to draw (Default is '1000'). Please
        note that this parameter must be specified in a string fromat ('1000' and
                                                                       not 1000).
    :param n_tuning: String that specifies the number of iterations to adjust
        the samplers's step sizes, scalings or similar (defauls is '500').
    :param n_chains: String that specifies the number of chains to sample. Defauls
        is '1' for faster estimation, but note that sampling independent chains
        is important for some convergence checks.
    :param cores: String that specifies the number of chains to run in parallel.
        (defauls is '1').
    :param init: Initialization method to use for auto-assigned NUTS samplers. The
        defauls is 'jitter+adapt_diag' that starts with a identity mass matrix
        and then adapt a diagonal based on the variance of the tuning samples
        while adding a uniform jitter in [-1, 1] to the starting point in each chain.
    :param target_accept: String that of a float in [0, 1] that regulates the
        step size such that we approximate this acceptance rate. The defauls is '0.8'
        but higher values like 0.9 or 0.95 often work better for problematic posteriors.
    :param order: String that defines the order of bspline or polynomial model.
        The defauls is '3'.
    :param nknots: String that defines the numbers of interior knots for the bspline model.
        The defauls is '3'. Two knots will be added to this number for boundries. So final 
        number of knots will be nknots+2. Higher values increase the model complexity with negative
        effect on the spped of estimations.
    :param nn_hidden_layers_num: String the specifies the number of hidden layers
        in neural network model. It can be either '1' or '2'. The default is set to '2'.
    :param nn_hidden_neuron_num: String that specifies the number of neurons in
        the hidden layers. The defauls is set to  '2'.

    Written by S.de Boer and S.M. Kia

    """

    def __init__(self, **kwargs):
        self.configs = dict()
        # inputs
        self.configs["trbefile"] = kwargs.get("trbefile", None)
        self.configs["tsbefile"] = kwargs.get("tsbefile", None)
        # Model settings
        self.configs["type"] = kwargs.get("model_type", "linear")
        self.configs["random_noise"] = kwargs.get(
            "random_noise", "True") == "True"
        self.configs["likelihood"] = kwargs.get("likelihood", "Normal")
        # sampler settings
        self.configs["nuts_sampler"] = kwargs.get("nuts_sampler", "pymc")
        self.configs["n_samples"] = int(kwargs.get("n_samples", "1000"))
        self.configs["n_tuning"] = int(kwargs.get("n_tuning", "500"))
        self.configs["n_chains"] = int(kwargs.get("n_chains", "1"))
        self.configs["sampler"] = kwargs.get("sampler", "NUTS")
        self.configs["target_accept"] = float(
            kwargs.get("target_accept", "0.8"))
        self.configs["init"] = kwargs.get("init", "jitter+adapt_diag_grad")
        self.configs["cores"] = int(kwargs.get("cores", "1"))
        self.configs["remove_datapoints_from_posterior"] = (
            kwargs.get("remove_datapoints_from_posterior", "True") == "True"
        )
        # model transfer setting
        self.configs["freedom"] = int(kwargs.get("freedom", "1"))
        self.configs["transferred"] = False
        # deprecated settings
        self.configs["skewed_likelihood"] = (
            kwargs.get("skewed_likelihood", "False") == "True"
        )
        # misc
        self.configs["pred_type"] = kwargs.get("pred_type", "single")

        if self.configs["type"] == "bspline":
            self.configs["order"] = int(kwargs.get("order", "3"))
            self.configs["nknots"] = int(kwargs.get("nknots", "3"))
        elif self.configs["type"] == "polynomial":
            self.configs["order"] = int(kwargs.get("order", "3"))
        elif self.configs["type"] == "nn":
            self.configs["nn_hidden_neuron_num"] = int(
                kwargs.get("nn_hidden_neuron_num", "2")
            )
            self.configs["nn_hidden_layers_num"] = int(
                kwargs.get("nn_hidden_layers_num", "2")
            )
            if self.configs["nn_hidden_layers_num"] > 2:
                raise ValueError(
                    "Using "
                    + str(self.configs["nn_hidden_layers_num"])
                    + " layers was not implemented. The number of "
                    + " layers has to be less than 3."
                )
        elif self.configs["type"] == "linear":
            pass
        else:
            raise ValueError(
                "Unknown model type, please specify from 'linear', \
                             'polynomial', 'bspline', or 'nn'."
            )

        if self.configs["type"] in ["bspline", "polynomial", "linear"]:
            for p in ["mu", "sigma", "epsilon", "delta"]:
                self.configs[f"linear_{p}"] = (
                    kwargs.get(f"linear_{p}", "False") == "True"
                )

                # Deprecations (remove in later version)
                if f"{p}_linear" in kwargs.keys():
                    print(
                        f"The keyword '{p}_linear' is deprecated. It is now automatically replaced with 'linear_{p}'"
                    )
                    self.configs[f"linear_{p}"] = (
                        kwargs.get(f"{p}_linear", "False") == "True"
                    )
                # End Deprecations

                for c in ["centered", "random"]:
                    self.configs[f"{c}_{p}"] = kwargs.get(
                        f"{c}_{p}", "False") == "True"
                    for sp in ["slope", "intercept"]:
                        self.configs[f"{c}_{sp}_{p}"] = (
                            kwargs.get(f"{c}_{sp}_{p}", "False") == "True"
                        )

            # Deprecations (remove in later version)
            if self.configs["linear_sigma"]:
                if "random_noise" in kwargs.keys():
                    print(
                        "The keyword 'random_noise' is deprecated. It is now automatically replaced with 'random_intercept_sigma', because sigma is linear"
                    )
                    self.configs["random_intercept_sigma"] = (
                        kwargs.get("random_noise", "True") == "True"
                    )
            elif "random_noise" in kwargs.keys():
                print(
                    "The keyword 'random_noise' is deprecated. It is now automatically replaced with 'random_sigma', because sigma is fixed"
                )
                self.configs["random_sigma"] = (
                    kwargs.get("random_noise", "True") == "True"
                )
            if "random_slope" in kwargs.keys():
                print(
                    "The keyword 'random_slope' is deprecated. It is now automatically replaced with 'random_intercept_mu'"
                )
                self.configs["random_slope_mu"] = (
                    kwargs.get("random_slope", "True") == "True"
                )
            # End Deprecations

        # Default parameters
        self.configs["linear_mu"] = kwargs.get("linear_mu", "True") == "True"
        self.configs["random_mu"] = kwargs.get("random_mu", "True") == "True"
        self.configs["random_intercept_mu"] = (
            kwargs.get("random_intercept_mu", "True") == "True"
        )
        self.configs["random_slope_mu"] = (
            kwargs.get("random_slope_mu", "True") == "True"
        )
        self.configs["random_sigma"] = kwargs.get(
            "random_sigma", "True") == "True"
        self.configs["centered_sigma"] = kwargs.get(
            "centered_sigma", "True") == "True"
        # End default parameters

        self.hbr = HBR(self.configs)

    @property
    def n_params(self):
        return 1

    @property
    def neg_log_lik(self):
        return -1

    def estimate(self, X, y, **kwargs):
        """
        Sample from the posterior of the Hierarchical Bayesian Regression model.

        This function samples from the posterior distribution of the Hierarchical Bayesian Regression (HBR) model given the data matrix 'X' and target 'y'.
        If 'trbefile' is provided in kwargs, it is used as batch effects for the training data.
        Otherwise, the batch effects are initialized as zeros.

        :param X: Data matrix.
        :param y: Target values.
        :param kwargs: Keyword arguments which may include:
            - 'trbefile': File containing the batch effects for the training data. Optional.
        :return: The instance of the NormHBR object.
        """
        trbefile = kwargs.get("trbefile", None)
        if trbefile is not None:
            batch_effects_train = fileio.load(trbefile)
        else:
            print("Could not find batch-effects file! Initilizing all as zeros ...")
            batch_effects_train = np.zeros([X.shape[0], 1])

        self.batch_effects_maps = [
            {v: i for i, v in enumerate(np.unique(batch_effects_train[:, j]))}
            for j in range(batch_effects_train.shape[1])
        ]

        self.hbr.estimate(X, y, batch_effects_train)

        return self
    
    def predict(self, Xs, X=None, Y=None, **kwargs):
        """
        Predict the target values for the given test data.

        This function predicts the target values for the given test data 'Xs' using the Hierarchical Bayesian Regression (HBR) model.
        If 'X' and 'Y' are provided, they are used to update the model before prediction.
        If 'tsbefile' is provided in kwargs, it is used to as batch effects for the test data.
        Otherwise, the batch effects are initialized as zeros.

        :param Xs: Test data matrix.
        :param X: Training data matrix. Optional.
        :param Y: Training target values. Optional.
        :param kwargs: Keyword arguments which may include:
            - 'tsbefile': File containing the batch effects for the test data. Optional.
        :return: A tuple containing the predicted target values and the marginal variances for the test data.
        :raises ValueError: If the model is a transferred model. In this case, use the predict_on_new_sites function.
        """
        tsbefile = kwargs.get("tsbefile", None)
        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:
            print("Could not find batch-effects file! Initilizing all as zeros ...")
            batch_effects_test = np.zeros([Xs.shape[0], 1])

        pred_type = self.configs["pred_type"]

        # if self.configs["transferred"] == False:
        yhat, s2 = self.hbr.predict(
            X=Xs,
            batch_effects=batch_effects_test,
            batch_effects_maps=self.batch_effects_maps,
            pred=pred_type,
            **kwargs,
        )

        return yhat.squeeze(), s2.squeeze()

    def transfer(self, X, y, batch_effects):
        """
        Samples from the posterior of the Hierarchical Bayesian Regression model.

        This function samples from the posterior of the Hierarchical Bayesian Regression (HBR) model given the data matrix 'X' and target 'y'. The posterior samples from the previous iteration are used to construct the priors for this one.
        If 'trbefile' is provided in kwargs, it is used as batch effects for the training data.
        Otherwise, the batch effects are initialized as zeros.

        :param X: Data matrix.
        :param y: Target values.
        :param kwargs: Keyword arguments which may include:
            - 'trbefile': File containing the batch effects for the training data. Optional.
        :return: The instance of the NormHBR object.
        """
        self.hbr.transfer(X, y, batch_effects)
        self.batch_effects_maps = [{v: i for i, v in enumerate(np.unique(batch_effects[:, j]))}
                                        for j in range(batch_effects.shape[1])]
        self.configs["transferred"] = True
        return self

    def predict_on_new_sites(self, X, batch_effects):
        """
        Predict the target values for the given test data on new sites.

        This function predicts the target values for the given test data 'X' on new sites using the Hierarchical Bayesian Regression (HBR) model.
        The batch effects for the new sites must be provided.

        :param X: Test data matrix for the new sites.
        :param batch_effects: Batch effects for the new sites.
        :return: A tuple containing the predicted target values and the marginal variances for the test data on the new sites.
        """
        
        yhat, s2 = self.hbr.predict(
                X,
                batch_effects=batch_effects,
                batch_effects_maps=self.batch_effects_maps
            )
        
        return yhat, s2

    
    def extend(
        self,
        X,
        y,
        batch_effects,
        X_dummy_ranges=[[0.1, 0.9, 0.01]],
        merge_batch_dim=0,
        samples=10,
        informative_prior=False
    ):
        """
        Extend the Hierarchical Bayesian Regression model using data sampled from the posterior predictive distribution.

        This function extends the Hierarchical Bayesian Regression (HBR) model, given the data matrix 'X' and target 'y'.
        It also generates data from the posterior predictive distribution and merges it with the new data before estimation.
        If 'informative_prior' is True, it uses the adapt method for estimation. Otherwise, it uses the estimate method.

        :param X: Data matrix for the new sites.
        :param y: Target values for the new sites.
        :param batch_effects: Batch effects for the new sites.
        :param X_dummy_ranges: Ranges for generating the dummy data. Default is [[0.1, 0.9, 0.01]].
        :param merge_batch_dim: Dimension for merging the batch effects. Default is 0.
        :param samples: Number of samples to generate for the dummy data. Default is 10.
        :param informative_prior: Whether to use the adapt method for estimation. Default is False.
        :return: The instance of the NormHBR object.
        """
        
        X_dummy, batch_effects_dummy = self.hbr.create_dummy_inputs(X)
        
        X_dummy, batch_effects_dummy, Y_dummy = self.hbr.generate(
            X_dummy, batch_effects_dummy, samples, batch_effects_maps=self.batch_effects_maps
        )

        batch_effects[:, merge_batch_dim] = (
            batch_effects[:, merge_batch_dim]
            + np.max(batch_effects_dummy[:, merge_batch_dim])
            + 1
        )

        X = np.concatenate((X_dummy, X))
        y = np.concatenate((Y_dummy, y))
        batch_effects = np.concatenate((batch_effects_dummy, batch_effects))
        
        self.batch_effects_maps = [ {v: i for i, v in enumerate(np.unique(batch_effects[:, j]))}
                                            for j in range(batch_effects.shape[1])
                                    ]
            
        if informative_prior:
            #raise NotImplementedError("The extension with informaitve prior is not implemented yet.")
            self.hbr.transfer(X, y, batch_effects)
        else:
            
            self.hbr.estimate(X, y, batch_effects)

        return self

    def tune(
        self,
        X,
        y,
        batch_effects,
        X_dummy_ranges=[[0.1, 0.9, 0.01]],
        merge_batch_dim=0,
        samples=10,
        informative_prior=False,
    ):
        """
        This function tunes the Hierarchical Bayesian Regression model using data sampled from the posterior predictive distribution. Its behavior is not tested, and it is unclear if the desired behavior is achieved.
        """

        # TODO need to check if this is correct

        print(
            "The 'tune' function is being called, but it is currently in development and its behavior is not tested. It is unclear if the desired behavior is achieved. Any output following this should be treated as unreliable."
        )

        tune_ids = list(np.unique(batch_effects[:, merge_batch_dim]))

        X_dummy, batch_effects_dummy = self.hbr.create_dummy_inputs(
            X_dummy_ranges)

        for idx in tune_ids:
            X_dummy = X_dummy[batch_effects_dummy[:,
                                                  merge_batch_dim] != idx, :]
            batch_effects_dummy = batch_effects_dummy[
                batch_effects_dummy[:, merge_batch_dim] != idx, :
            ]

        X_dummy, batch_effects_dummy, Y_dummy = self.hbr.generate(
            X_dummy, batch_effects_dummy, samples, self.batch_effects_maps
        )

        if informative_prior:
            self.hbr.adapt(
                np.concatenate((X_dummy, X)),
                np.concatenate((Y_dummy, y)),
                np.concatenate((batch_effects_dummy, batch_effects)),
            )
        else:
            self.hbr.estimate(
                np.concatenate((X_dummy, X)),
                np.concatenate((Y_dummy, y)),
                np.concatenate((batch_effects_dummy, batch_effects)),
            )

        return self

    def merge(
        self, nm, X_dummy_ranges=[[0.1, 0.9, 0.01]], merge_batch_dim=0, samples=10
    ):
        """
        Samples from the posterior predictive distribitions of two models, merges them, and estimates a model on the merged data.

        This function samples from the posterior predictive distribitions of two models, merges them, and estimates a model on the merged data.

        :param nm: The other NormHBR object.
        :param X_dummy_ranges: Ranges for generating the dummy data. Default is [[0.1, 0.9, 0.01]].
        :param merge_batch_dim: Dimension for merging the batch effects. Default is 0.
        :param samples: Number of samples to generate for the dummy data. Default is 10.
        """

        X_dummy1, batch_effects_dummy1 = self.hbr.create_dummy_inputs(
            X_dummy_ranges)
        X_dummy2, batch_effects_dummy2 = nm.hbr.create_dummy_inputs(
            X_dummy_ranges)

        X_dummy1, batch_effects_dummy1, Y_dummy1 = self.hbr.generate(
            X_dummy1, batch_effects_dummy1, samples, self.batch_effects_maps
        )
        X_dummy2, batch_effects_dummy2, Y_dummy2 = nm.hbr.generate(
            X_dummy2, batch_effects_dummy2, samples
        )

        batch_effects_dummy2[:, merge_batch_dim] = (
            batch_effects_dummy2[:, merge_batch_dim]
            + np.max(batch_effects_dummy1[:, merge_batch_dim])
            + 1
        )

        self.hbr.estimate(
            np.concatenate((X_dummy1, X_dummy2)),
            np.concatenate((Y_dummy1, Y_dummy2)),
            np.concatenate((batch_effects_dummy1, batch_effects_dummy2)),
        )

        return self

    def generate(self, X, batch_effects, samples=10):
        X, batch_effects, generated_samples = self.hbr.generate(
            X, batch_effects, samples, self.batch_effects_maps
        )
        return X, batch_effects, generated_samples

    def get_mcmc_quantiles(self, X, batch_effects=None, z_scores=None):
        """
        Computes quantiles of an estimated normative model.

        Args:
            X ([N*p]ndarray): covariates for which the quantiles are computed (must be scaled if scaler is set)
            batch_effects (ndarray): the batch effects corresponding to X
            z_scores (ndarray): Use this to determine which quantiles will be computed. The resulting quantiles will have the z-scores given in this list.
        """
        # Set batch effects to zero if none are provided
        if batch_effects is None:
            batch_effects = np.zeros([X.shape[0], 1])

        # Set the z_scores for which the quantiles are computed
        if z_scores is None:
            z_scores = np.arange(-3, 4)
        elif len(z_scores.shape) == 2:
            if not z_scores.shape[0] == X.shape[0]:
                raise ValueError("The number of columns in z_scores must match the number of columns in X")
            z_scores = z_scores.T

        # Determine the variables to predict
        match self.configs["likelihood"]:   
            case "Normal":
                var_names = ["mu_samples", "sigma_samples", "sigma_plus_samples"]
            case  "SHASHo" | "SHASHo2" | "SHASHb":
                var_names = [
                    "mu_samples",
                    "sigma_samples",
                    "sigma_plus_samples",
                    "epsilon_samples",
                    "delta_samples",
                    "delta_plus_samples",
                ]
            case _:
                exit("Unknown likelihood: " + self.configs["likelihood"])

        # Delete the posterior predictive if it already exists
        if "posterior_predictive" in self.hbr.idata.groups():
            del self.hbr.idata.posterior_predictive

        self.hbr.predict(
            X=X,
            batch_effects=batch_effects,
            batch_effects_maps=self.batch_effects_maps,
            pred="single",
            var_names=var_names + ["y_like"],
        )

        # Extract the relevant samples from the idata
        post_pred = az.extract(
            self.hbr.idata, "posterior_predictive", var_names=var_names
        )

        # Remove superfluous var_nammes
        var_names.remove("sigma_samples")
        if "delta_samples" in var_names:
            var_names.remove("delta_samples")

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = list(map(lambda x: post_pred[x], var_names))

        # Create an array to hold the quantiles
        len_synth_data, n_mcmc_samples = post_pred["mu_samples"].shape
        quantiles = np.zeros(
            (z_scores.shape[0], len_synth_data, n_mcmc_samples))

        # Compute the quantile iteratively for each z-score

        for i, j in enumerate(z_scores):
            if len(z_scores.shape) == 1:
                zs = np.full((len_synth_data, n_mcmc_samples), j, dtype=float)
            else:
                zs = np.repeat(j[:,None], n_mcmc_samples, axis=1)
            quantiles[i] = xarray.apply_ufunc(
                quantile,
                *array_of_vars,
                kwargs={"zs": zs, "likelihood": self.configs["likelihood"]},
            )
        return quantiles.mean(axis=-1)

    def get_mcmc_zscores(self, X, y, **kwargs):
        """
        Computes zscores of data given an estimated model

        Args:
            X ([N*p]ndarray): covariates
            y ([N*1]ndarray): response variables
        """

        print(self.configs["likelihood"])

        tsbefile = kwargs.get("tsbefile", None)
        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:  # Set batch effects to zero if none are provided
            print("Could not find batch-effects file! Initializing all as zeros ...")
            batch_effects_test = np.zeros([X.shape[0], 1])

        # Determine the variables to predict
        if self.configs["likelihood"] == "Normal":
            var_names = ["mu_samples", "sigma_samples", "sigma_plus_samples"]
        elif self.configs["likelihood"].startswith("SHASH"):
            var_names = [
                "mu_samples",
                "sigma_samples",
                "sigma_plus_samples",
                "epsilon_samples",
                "delta_samples",
                "delta_plus_samples",
            ]
        else:
            exit("Unknown likelihood: " + self.configs["likelihood"])

        # Delete the posterior predictive if it already exists
        if "posterior_predictive" in self.hbr.idata.groups():
            del self.hbr.idata.posterior_predictive

        # Do a forward to get the posterior predictive in the idata
        self.hbr.predict(
            X=X,
            batch_effects=batch_effects_test,
            batch_effects_maps=self.batch_effects_maps,
            pred="single",
            var_names=var_names + ["y_like"],
        )

        # Extract the relevant samples from the idata
        post_pred = az.extract(
            self.hbr.idata, "posterior_predictive", var_names=var_names
        )

        # Remove superfluous var_names
        var_names.remove("sigma_samples")
        if "delta_samples" in var_names:
            var_names.remove("delta_samples")

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = list(map(lambda x: post_pred[x], var_names))

        # Create an array to hold the quantiles
        len_data, n_mcmc_samples = post_pred["mu_samples"].shape

        # Compute the quantile iteratively for each z-score
        z_scores = xarray.apply_ufunc(
            z_score,
            *array_of_vars,
            kwargs={"y": y, "likelihood": self.configs["likelihood"]},
        )
        return z_scores.mean(axis=-1).values


def S_inv(x, e, d):
    return np.sinh((np.arcsinh(x) + e) / d)


def K(p, x):
    """
    Computes the values of spp.kv(p,x) for only the unique values of p
    """

    ps, idxs = np.unique(p, return_inverse=True)
    return spp.kv(ps, x)[idxs].reshape(p.shape)


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


def quantile(mu, sigma, epsilon=None, delta=None, zs=0, likelihood="Normal"):
    """Get the zs'th quantiles given likelihood parameters"""
    if likelihood.startswith("SHASH"):
        if likelihood == "SHASHo":
            quantiles = S_inv(zs, epsilon, delta) * sigma + mu
        elif likelihood == "SHASHo2":
            sigma_d = sigma / delta
            quantiles = S_inv(zs, epsilon, delta) * sigma_d + mu
        elif likelihood == "SHASHb":
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            SHASH_c = (S_inv(zs, epsilon, delta) - true_mu) / true_sigma
            quantiles = SHASH_c * sigma + mu
    elif likelihood == "Normal":
        quantiles = zs * sigma + mu
    else:
        exit("Unsupported likelihood")
    return quantiles


def z_score(mu, sigma, epsilon=None, delta=None, y=None, likelihood="Normal"):
    """Get the z-scores of Y, given likelihood parameters"""
    if likelihood.startswith("SHASH"):
        if likelihood == "SHASHo":
            SHASH = (y - mu) / sigma
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        elif likelihood == "SHASHo2":
            sigma_d = sigma / delta
            SHASH = (y - mu) / sigma_d
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        elif likelihood == "SHASHb":
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            SHASH_c = (y - mu) / sigma
            SHASH = SHASH_c * true_sigma + true_mu
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
    elif likelihood == "Normal":
        Z = (y - mu) / sigma
    else:
        exit("Unsupported likelihood")
    return Z
