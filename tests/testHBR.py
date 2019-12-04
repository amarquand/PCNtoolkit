#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:26:35 2019

@author: seykia
"""

import pickle
import numpy as np
from nispat.normative_model.norm_utils import norm_init
import matplotlib.pyplot as plt

import pandas as pd
import pymc3 as pm

def trace_quantiles(x):
    return pd.DataFrame(pm.quantiles(x, [5, 25, 50, 75, 95]))

########################### TESTING HBR ################################

# Simulating the data
training_samples_num = 200
test_samples_num = 81
configparam = dict()

configparam['confounds'] = dict()
configparam['batch_effects_train'] = np.random.randint(0,2,[training_samples_num,2]) 
configparam['batch_effects_test'] = np.random.randint(0,2,[test_samples_num,2])
configparam['batch_effects_train'][:,1] = 0
configparam['batch_effects_test'][:,1] = 0


X = np.random.randint(10,91,training_samples_num)
Y = np.zeros([training_samples_num,])
Y[configparam['batch_effects_train'][:,0]==0,] = X[configparam['batch_effects_train'][:,0]==0,] * 0.2 + X[configparam['batch_effects_train'][:,0]==0,] * 0.25 * np.random.randn(np.sum(configparam['batch_effects_train'][:,0]==0))
Y[configparam['batch_effects_train'][:,0]==1,] = X[configparam['batch_effects_train'][:,0]==1,] * 0.85 + 2 + 5 * np.random.randn(np.sum(configparam['batch_effects_train'][:,0]==1))
Y = Y + np.random.randn(training_samples_num)

Xs = np.arange(10,91)
Ys = np.zeros([test_samples_num,])
Ys[configparam['batch_effects_test'][:,0]==0,] = Xs[configparam['batch_effects_test'][:,0]==0,] * 0.2 + Xs[configparam['batch_effects_test'][:,0]==0,] * 0.25 * np.random.randn(np.sum(configparam['batch_effects_test'][:,0]==0))
Ys[configparam['batch_effects_test'][:,0]==1,] = Xs[configparam['batch_effects_test'][:,0]==1,] * 0.85 + 2 + 5 * np.random.randn(np.sum(configparam['batch_effects_test'][:,0]==1))

# Trivial Model
configparam['model_type'] = 'linear'
configparam['random_intercept'] = False
configparam['random_slope'] = False
configparam['random_noise'] = False
configparam['hetero_noise'] = False
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
    
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_trivial, s2_trivial = nm.predict(Xs)

# Random Intercept and Slope
configparam['model_type'] = 'linear'
configparam['random_intercept'] = True
configparam['random_slope'] = True
configparam['random_noise'] = False
configparam['hetero_noise'] = False
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
    
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_rand_int_slp , s2_rand_int_slp = nm.predict(Xs)

# Random Intercept and Slope and Noise
configparam['model_type'] = 'linear'
configparam['random_intercept'] = True
configparam['random_slope'] = True
configparam['random_noise'] = True
configparam['hetero_noise'] = False
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
    
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_rand_int_slp_nse , s2_rand_int_slp_nse = nm.predict(Xs)


# Heteroskedastic Noise Model
configparam['model_type'] = 'linear'
configparam['random_intercept'] = True
configparam['random_slope'] = True
configparam['random_noise'] = True
configparam['hetero_noise'] = True
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
    
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_hetero , s2_hetero = nm.predict(Xs)


plt.subplot(2, 2, 1)
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==0,],Ys[configparam['batch_effects_test'][:,0]==0,])
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==1,],Ys[configparam['batch_effects_test'][:,0]==1,])
plt.plot(Xs[configparam['batch_effects_test'][:,0]==0,],yhat_trivial[[configparam['batch_effects_test'][:,0]==0,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==0,], 
                 yhat_trivial[configparam['batch_effects_test'][:,0]==0,] - 1.96 * np.sqrt(s2_trivial[configparam['batch_effects_test'][:,0]==0,]), 
                 yhat_trivial[configparam['batch_effects_test'][:,0]==0,] + 1.96 * np.sqrt(s2_trivial[configparam['batch_effects_test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['batch_effects_test'][:,0]==1,],yhat_trivial[[configparam['batch_effects_test'][:,0]==1,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==1,], 
                 yhat_trivial[configparam['batch_effects_test'][:,0]==1,] - 1.96 * np.sqrt(s2_trivial[configparam['batch_effects_test'][:,0]==1,]), 
                 yhat_trivial[configparam['batch_effects_test'][:,0]==1,] + 1.96 * np.sqrt(s2_trivial[configparam['batch_effects_test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Trivial Model')

plt.subplot(2, 2, 2)
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==0,],Ys[configparam['batch_effects_test'][:,0]==0,])
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==1,],Ys[configparam['batch_effects_test'][:,0]==1,])
plt.plot(Xs[configparam['batch_effects_test'][:,0]==0,],yhat_rand_int_slp[[configparam['batch_effects_test'][:,0]==0,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==0,], 
                 yhat_rand_int_slp[configparam['batch_effects_test'][:,0]==0,] - np.sqrt(s2_rand_int_slp[configparam['batch_effects_test'][:,0]==0,]), 
                 yhat_rand_int_slp[configparam['batch_effects_test'][:,0]==0,] + np.sqrt(s2_rand_int_slp[configparam['batch_effects_test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['batch_effects_test'][:,0]==1,],yhat_rand_int_slp[[configparam['batch_effects_test'][:,0]==1,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==1,], 
                 yhat_rand_int_slp[configparam['batch_effects_test'][:,0]==1,] - np.sqrt(s2_rand_int_slp[configparam['batch_effects_test'][:,0]==1,]), 
                 yhat_rand_int_slp[configparam['batch_effects_test'][:,0]==1,] + np.sqrt(s2_rand_int_slp[configparam['batch_effects_test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Lin: Random Intercept and Slope')

plt.subplot(2, 2, 3)
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==0,],Ys[configparam['batch_effects_test'][:,0]==0,])
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==1,],Ys[configparam['batch_effects_test'][:,0]==1,])
plt.plot(Xs[configparam['batch_effects_test'][:,0]==0,],yhat_rand_int_slp_nse[[configparam['batch_effects_test'][:,0]==0,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==0,], 
                 yhat_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==0,] - np.sqrt(s2_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==0,]), 
                 yhat_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==0,] + np.sqrt(s2_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['batch_effects_test'][:,0]==1,],yhat_rand_int_slp_nse[[configparam['batch_effects_test'][:,0]==1,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==1,], 
                 yhat_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==1,] - np.sqrt(s2_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==1,]), 
                 yhat_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==1,] + np.sqrt(s2_rand_int_slp_nse[configparam['batch_effects_test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Lin: Random Intercept and Slope and Noise')

plt.subplot(2, 2, 4)
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==0,],Ys[configparam['batch_effects_test'][:,0]==0,])
plt.scatter(Xs[configparam['batch_effects_test'][:,0]==1,],Ys[configparam['batch_effects_test'][:,0]==1,])
plt.plot(Xs[configparam['batch_effects_test'][:,0]==0,],yhat_hetero[[configparam['batch_effects_test'][:,0]==0,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==0,], 
                 yhat_hetero[configparam['batch_effects_test'][:,0]==0,] - np.sqrt(s2_hetero[configparam['batch_effects_test'][:,0]==0,]), 
                 yhat_hetero[configparam['batch_effects_test'][:,0]==0,] + np.sqrt(s2_hetero[configparam['batch_effects_test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['batch_effects_test'][:,0]==1,],yhat_hetero[[configparam['batch_effects_test'][:,0]==1,]])
plt.fill_between(Xs[configparam['batch_effects_test'][:,0]==1,], 
                 yhat_hetero[configparam['batch_effects_test'][:,0]==1,] - np.sqrt(s2_hetero[configparam['batch_effects_test'][:,0]==1,]), 
                 yhat_hetero[configparam['batch_effects_test'][:,0]==1,] + np.sqrt(s2_hetero[configparam['batch_effects_test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Heteroskedastic Noise Model')

