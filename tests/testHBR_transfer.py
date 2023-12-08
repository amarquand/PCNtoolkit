#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:26:35 2019

@author: seykia

This script tests HBR models with default configs on toy data.

"""

import os
import numpy as np
from pcntoolkit.normative_model.norm_utils import norm_init
from pcntoolkit.util.utils import simulate_data
import matplotlib.pyplot as plt
from pcntoolkit.normative import estimate
from warnings import filterwarnings
from pcntoolkit.util.utils import scaler
import xarray

filterwarnings('ignore')

np.random.seed(10)

########################### Experiment Settings ###############################


working_dir = '/home/stijn/temp/'  # Specifyexit() a working directory
# to save data and results.

simulation_method = 'linear'
n_features = 1      # The number of input features of X
n_grps = 5          # Number of batches in data
n_transfer_groups = 2  # number of batches in transfer data
n_samples = 500     # Number of samples in each group (use a list for different
# sample numbers across different batches)
n_transfer_samples = 100

model_types = ['linear']  # models to try

############################## Data Simulation ################################


X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test, coef = \
    simulate_data(simulation_method, n_samples, n_features, n_grps,
                  working_dir=working_dir, plot=True)

X_train_transfer, Y_train_transfer, grp_id_train_transfer, X_test_transfer, Y_test_transfer, grp_id_test_transfer, coef = simulate_data(
    simulation_method, n_transfer_samples, n_features=n_features, n_grps=n_transfer_groups, plot=True)

################################# Methods Tests ###############################


for model_type in model_types:
    nm = norm_init(X_train, Y_train, alg='hbr', likelihood='Normal', model_type=model_type,
                   n_chains=4, cores=4, n_samples=100, n_tuning=50, freedom=5, nknots=8, target_accept="0.99")

    print("Now Estimating on original train data ==============================================")
    nm.estimate(X_train, Y_train, trbefile=working_dir+'trbefile.pkl')
    print("Now Predicting on original test data ==============================================")
    yhat, ys2 = nm.predict(X_test, tsbefile=working_dir+'tsbefile.pkl')

    for i in range(n_features):
        sorted_idx = X_test[:, i].argsort(axis=0).squeeze()
        temp_X = X_test[sorted_idx, i]
        temp_Y = Y_test[sorted_idx,]
        temp_be = grp_id_test[sorted_idx, :].squeeze()
        temp_yhat = yhat[sorted_idx,]
        temp_s2 = ys2[sorted_idx,]

        plt.figure()
        for j in range(n_grps):
            plt.scatter(temp_X[temp_be == j,], temp_Y[temp_be == j,],
                        label='Group' + str(j))
            plt.plot(temp_X[temp_be == j,], temp_yhat[temp_be == j,])
            plt.fill_between(temp_X[temp_be == j,], temp_yhat[temp_be == j,] -
                             1.96 * np.sqrt(temp_s2[temp_be == j,]),
                             temp_yhat[temp_be == j,] +
                             1.96 * np.sqrt(temp_s2[temp_be == j,]),
                             color='gray', alpha=0.2)
        plt.title('Model %s, Feature %d' % (model_type, i))
        plt.legend()
        plt.show()

    print("Now Estimating on transfer train data ==============================================")
    nm.estimate_on_new_sites(
        X_train_transfer, Y_train_transfer, grp_id_train_transfer)
    print("Now Estimating on transfer test data ==============================================")
    yhat, s2 = nm.predict_on_new_sites(X_test_transfer, grp_id_test_transfer)

    for i in range(n_features):
        sorted_idx = X_test_transfer[:, i].argsort(axis=0).squeeze()
        temp_X = X_test_transfer[sorted_idx, i]
        temp_Y = Y_test_transfer[sorted_idx,]
        temp_be = grp_id_test_transfer[sorted_idx, :].squeeze()
        temp_yhat = yhat[sorted_idx,]
        temp_s2 = ys2[sorted_idx,]

        for j in range(n_transfer_groups):
            plt.scatter(temp_X[temp_be == j,], temp_Y[temp_be == j,],
                        label='Group' + str(j))
            plt.plot(temp_X[temp_be == j,], temp_yhat[temp_be == j,])
            plt.fill_between(temp_X[temp_be == j,], temp_yhat[temp_be == j,] -
                             1.96 * np.sqrt(temp_s2[temp_be == j,]),
                             temp_yhat[temp_be == j,] +
                             1.96 * np.sqrt(temp_s2[temp_be == j,]),
                             color='gray', alpha=0.2)
        plt.title('Transfer model %s, Feature %d' % (model_type, i))
        plt.legend()
        plt.show()


############################## Normative Modelling Test #######################


model_type = model_types[0]

covfile = working_dir + 'X_train.pkl'
respfile = working_dir + 'Y_train.pkl'
testcov = working_dir + 'X_test.pkl'
testresp = working_dir + 'Y_test.pkl'
trbefile = working_dir + 'trbefile.pkl'
tsbefile = working_dir + 'tsbefile.pkl'

os.chdir(working_dir)

estimate(covfile, respfile, testcov=testcov, testresp=testresp, trbefile=trbefile,
         tsbefile=tsbefile, alg='hbr', outputsuffix='_' + model_type,
         inscaler='None', outscaler='None', model_type=model_type,
         savemodel='True', saveoutput='True')


###############################################################################
