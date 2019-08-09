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

########################### TESTING HBR ################################

# Simulating the data
training_samples_num = 200
test_samples_num = 81
configparam = dict()

configparam['confounds'] = dict()
configparam['confounds']['train'] = np.random.randint(0,2,[training_samples_num,2]) 
configparam['confounds']['test'] = np.random.randint(0,2,[test_samples_num,2])

X = np.random.randint(10,91,training_samples_num)
Y = np.zeros([training_samples_num,])
Y[configparam['confounds']['train'][:,0]==0,] = X[configparam['confounds']['train'][:,0]==0,] * 0.2 + np.random.randn(np.sum(configparam['confounds']['train'][:,0]==0))
Y[configparam['confounds']['train'][:,0]==1,] = X[configparam['confounds']['train'][:,0]==1,] * 0.5 + 2 + 5 * np.random.randn(np.sum(configparam['confounds']['train'][:,0]==1))
Y = Y + np.random.randn(training_samples_num)

Xs = np.arange(10,91)
Ys = np.zeros([test_samples_num,])
Ys[configparam['confounds']['test'][:,0]==0,] = Xs[configparam['confounds']['test'][:,0]==0,] * 0.2 + np.random.randn(np.sum(configparam['confounds']['test'][:,0]==0))
Ys[configparam['confounds']['test'][:,0]==1,] = Xs[configparam['confounds']['test'][:,0]==1,] * 0.5 + 2 + 5 * np.random.randn(np.sum(configparam['confounds']['test'][:,0]==1))

# Running the model
configparam['model_type'] = 'lin_rand_int'
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_rand_int, s2_rand_int = nm.predict(X, Y, Xs)

configparam['model_type'] = 'lin_rand_int_slp'
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_rand_int_slp, s2_rand_int_slp = nm.predict(X, Y, Xs)

configparam['model_type'] = 'lin_rand_int_slp_nse'
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_rand_int_slp_nse, s2_rand_int_slp_nse = nm.predict(X, Y, Xs)

configparam['model_type'] = 'poly2'
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat_poly2, s2_poly2 = nm.predict(X, Y, Xs)

plt.subplot(2, 2, 1)
plt.scatter(Xs[configparam['confounds']['test'][:,0]==0,],Ys[configparam['confounds']['test'][:,0]==0,])
plt.scatter(Xs[configparam['confounds']['test'][:,0]==1,],Ys[configparam['confounds']['test'][:,0]==1,])
plt.plot(Xs[configparam['confounds']['test'][:,0]==0,],yhat_rand_int[[configparam['confounds']['test'][:,0]==0,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==0,], 
                 yhat_rand_int[configparam['confounds']['test'][:,0]==0,] - np.sqrt(s2_rand_int[configparam['confounds']['test'][:,0]==0,]), 
                 yhat_rand_int[configparam['confounds']['test'][:,0]==0,] + np.sqrt(s2_rand_int[configparam['confounds']['test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['confounds']['test'][:,0]==1,],yhat_rand_int[[configparam['confounds']['test'][:,0]==1,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==1,], 
                 yhat_rand_int[configparam['confounds']['test'][:,0]==1,] - np.sqrt(s2_rand_int[configparam['confounds']['test'][:,0]==1,]), 
                 yhat_rand_int[configparam['confounds']['test'][:,0]==1,] + np.sqrt(s2_rand_int[configparam['confounds']['test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Lin: Random Intercept')

plt.subplot(2, 2, 2)
plt.scatter(Xs[configparam['confounds']['test'][:,0]==0,],Ys[configparam['confounds']['test'][:,0]==0,])
plt.scatter(Xs[configparam['confounds']['test'][:,0]==1,],Ys[configparam['confounds']['test'][:,0]==1,])
plt.plot(Xs[configparam['confounds']['test'][:,0]==0,],yhat_rand_int_slp[[configparam['confounds']['test'][:,0]==0,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==0,], 
                 yhat_rand_int_slp[configparam['confounds']['test'][:,0]==0,] - np.sqrt(s2_rand_int_slp[configparam['confounds']['test'][:,0]==0,]), 
                 yhat_rand_int_slp[configparam['confounds']['test'][:,0]==0,] + np.sqrt(s2_rand_int_slp[configparam['confounds']['test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['confounds']['test'][:,0]==1,],yhat_rand_int_slp[[configparam['confounds']['test'][:,0]==1,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==1,], 
                 yhat_rand_int_slp[configparam['confounds']['test'][:,0]==1,] - np.sqrt(s2_rand_int_slp[configparam['confounds']['test'][:,0]==1,]), 
                 yhat_rand_int_slp[configparam['confounds']['test'][:,0]==1,] + np.sqrt(s2_rand_int_slp[configparam['confounds']['test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Lin: Random Intercept and Slope')

plt.subplot(2, 2, 3)
plt.scatter(Xs[configparam['confounds']['test'][:,0]==0,],Ys[configparam['confounds']['test'][:,0]==0,])
plt.scatter(Xs[configparam['confounds']['test'][:,0]==1,],Ys[configparam['confounds']['test'][:,0]==1,])
plt.plot(Xs[configparam['confounds']['test'][:,0]==0,],yhat_rand_int_slp_nse[[configparam['confounds']['test'][:,0]==0,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==0,], 
                 yhat_rand_int_slp_nse[configparam['confounds']['test'][:,0]==0,] - np.sqrt(s2_rand_int_slp_nse[configparam['confounds']['test'][:,0]==0,]), 
                 yhat_rand_int_slp_nse[configparam['confounds']['test'][:,0]==0,] + np.sqrt(s2_rand_int_slp_nse[configparam['confounds']['test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['confounds']['test'][:,0]==1,],yhat_rand_int_slp_nse[[configparam['confounds']['test'][:,0]==1,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==1,], 
                 yhat_rand_int_slp_nse[configparam['confounds']['test'][:,0]==1,] - np.sqrt(s2_rand_int_slp_nse[configparam['confounds']['test'][:,0]==1,]), 
                 yhat_rand_int_slp_nse[configparam['confounds']['test'][:,0]==1,] + np.sqrt(s2_rand_int_slp_nse[configparam['confounds']['test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Lin: Random Intercept and Slope and Noise')

plt.subplot(2, 2, 4)
plt.scatter(Xs[configparam['confounds']['test'][:,0]==0,],Ys[configparam['confounds']['test'][:,0]==0,])
plt.scatter(Xs[configparam['confounds']['test'][:,0]==1,],Ys[configparam['confounds']['test'][:,0]==1,])
plt.plot(Xs[configparam['confounds']['test'][:,0]==0,],yhat_poly2[[configparam['confounds']['test'][:,0]==0,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==0,], 
                 yhat_poly2[configparam['confounds']['test'][:,0]==0,] - np.sqrt(s2_poly2[configparam['confounds']['test'][:,0]==0,]), 
                 yhat_poly2[configparam['confounds']['test'][:,0]==0,] + np.sqrt(s2_poly2[configparam['confounds']['test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['confounds']['test'][:,0]==1,],yhat_poly2[[configparam['confounds']['test'][:,0]==1,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==1,], 
                 yhat_poly2[configparam['confounds']['test'][:,0]==1,] - np.sqrt(s2_poly2[configparam['confounds']['test'][:,0]==1,]), 
                 yhat_poly2[configparam['confounds']['test'][:,0]==1,] + np.sqrt(s2_poly2[configparam['confounds']['test'][:,0]==1,]),
                 color='gray', alpha=0.2)
plt.title('Poly2: Random Intercept and Slope and noise')

