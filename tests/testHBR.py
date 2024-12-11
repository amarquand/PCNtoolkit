#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
"""
Created on Mon Jul 29 13:26:35 2019

@author: seykia

This script tests HBR models with default configs on toy data.

"""

# %%
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np

from pcntoolkit.normative_model.norm_utils import norm_init
from pcntoolkit.util.utils import simulate_data

filterwarnings("ignore")


########################### Experiment Settings ###############################


random_state = 40
working_dir = "/Users/stijndeboer/tmp/"  # Specify a working directory to save data and results.

simulation_method = "linear"
n_features = 1  # The number of input features of X
n_grps = 3  # Number of batches in data
n_samples = 500  # Number of samples in each group (use a list for different
# sample numbers across different batches)

model_type = "linear"  # modelto try 'linear, ''polynomial', 'bspline'


############################## Data Simulation ################################


X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test, coef = simulate_data(
    simulation_method,
    n_samples,
    n_features,
    n_grps,
    working_dir=working_dir,
    plot=True,
    noise="heteroscedastic_nongaussian",
    random_state=random_state,
)

################################# Fittig and Predicting ###############################

nm = norm_init(
    X_train,
    Y_train,
    alg="hbr",
    model_type=model_type,
    likelihood="SHASHb",
    linear_sigma="True",
    random_slope_mu="True",
    linear_epsilon="False",
    linear_delta="False",
    nuts_sampler="nutpie",
)

nm.estimate(X_train, Y_train, trbefile=working_dir + "trbefile.pkl")
yhat, ys2 = nm.predict(X_test, tsbefile=working_dir + "tsbefile.pkl")


################################# Plotting Quantiles ###############################
for i in range(n_features):
    sorted_idx = X_test[:, i].argsort(axis=0).squeeze()
    temp_X = X_test[sorted_idx, i]
    temp_Y = Y_test[sorted_idx,]
    temp_be = grp_id_test[sorted_idx, :].squeeze()
    temp_yhat = yhat[sorted_idx,]
    temp_s2 = ys2[sorted_idx,]

    plt.figure()
    for j in range(n_grps):
        scat1 = plt.scatter(
            temp_X[temp_be == j,], temp_Y[temp_be == j,], label="Group" + str(j)
        )
        # Showing the quantiles
        resolution = 200
        synth_X = np.linspace(np.min(X_train), np.max(X_train), resolution)
        q = nm.get_mcmc_quantiles(synth_X, batch_effects=j * np.ones(resolution))
        col = scat1.get_facecolors()[0]
        plt.plot(synth_X, q.T, linewidth=1, color=col, zorder=0)

    plt.title("Model %s, Feature %d" % (model_type, i))
    plt.legend()
    plt.show(block=False)
    plt.savefig(working_dir + "quantiles_" + model_type + "_feature_" + str(i) + ".png")

    for j in range(n_grps):
        plt.figure()
        plt.scatter(temp_X[temp_be == j,], temp_Y[temp_be == j,])
        plt.plot(temp_X[temp_be == j,], temp_yhat[temp_be == j,], color="red")
        plt.fill_between(
            temp_X[temp_be == j,].squeeze(),
            (temp_yhat[temp_be == j,] - 2 * np.sqrt(temp_s2[temp_be == j,])).squeeze(),
            (temp_yhat[temp_be == j,] + 2 * np.sqrt(temp_s2[temp_be == j,])).squeeze(),
            color="red",
            alpha=0.2,
        )
        plt.title("Model %s, Group %d, Feature %d" % (model_type, j, i))
        plt.show(block=False)
        plt.savefig(
            working_dir
            + "pred_"
            + model_type
            + "_group_"
            + str(j)
            + "_feature_"
            + str(i)
            + ".png"
        )


############################## Normative Modelling Test #######################

# covfile = working_dir + 'X_train.pkl'
# respfile = working_dir + 'Y_train.pkl'
# testcov = working_dir + 'X_test.pkl'
# testresp = working_dir + 'Y_test.pkl'
# trbefile = working_dir + 'trbefile.pkl'
# tsbefile = working_dir + 'tsbefile.pkl'

# os.chdir(working_dir)

# estimate(covfile, respfile, testcovfile_path=testcov, testrespfile_path=testresp, trbefile=trbefile,
#          tsbefile=tsbefile, alg='hbr', outputsuffix='_' + model_type,
#          inscaler='None', outscaler='None', model_type=model_type,
#          savemodel='True', saveoutput='True')


###############################################################################

 # %%

for j in range(n_grps):
    # Showing the quantiles
    resolution = 200
    synth_X = np.linspace(np.min(X_train), np.max(X_train), resolution)
    q = nm.get_mcmc_quantiles(synth_X, batch_effects=j * np.ones(resolution))
    plt.figure()
    plt.scatter(temp_X[temp_be == j,], temp_Y[temp_be == j,])
    plt.plot(synth_X, q.T, color="black")
    plt.title("Model %s, Group %d, Feature %d" % (model_type, j, i))
    plt.show(block=False)
    plt.savefig(
        working_dir
        + "pred_"
        + model_type
        + "_group_"
        + str(j)
        + "_feature_"
        + str(i)
        + ".png"
    )
# %%
