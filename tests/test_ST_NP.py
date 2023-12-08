#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:30:11 2019

@author: seykia
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from pcntoolkit.normative_model.norm_utils import norm_init

sample_num = 1114
X_train = np.random.rand(sample_num, 1) * 5 - 2
Y_train = - 2 * X_train**2 + 2 * X_train + 1 + \
    X_train * np.random.randn(sample_num, 1)
X_test = np.random.rand(sample_num, 1) * 5 - 2
Y_test = - 2 * X_test**2 + 2 * X_test + 1 + \
    X_test * np.random.randn(sample_num, 1)

configparam = dict()
configparam['batch_size'] = 10
configparam['epochs'] = 100
configparam['m'] = 200
configparam['hidden_neuron_num'] = 10
configparam['r_dim'] = 5
configparam['z_dim'] = 3
configparam['nv'] = 0.01
configparam['device'] = torch.device('cpu')
with open('NP_configs.pkl', 'wb') as file:
    pickle.dump(configparam, file)

nm = norm_init(X_train, Y_train, alg='np', configparam='NP_configs.pkl')
nm.estimate(X_train, Y_train)
y_hat, ys2 = nm.predict(X_test)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_test, Y_test, label='Test Data')
ax1.errorbar(X_test, y_hat, yerr=1.96 * np.sqrt(ys2).squeeze(),
             fmt='.', c='y', alpha=0.2, label='95% Prediction Intervals')
ax1.scatter(X_test, y_hat, c='r', label='Prediction')
ax1.set_title('Estimated Function')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
