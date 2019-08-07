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

########################### TESTING LINEAR HBR ################################

# Simulating the data
training_samples_num = 200
test_samples_num = 81
configparam = dict()
configparam['model_type'] = 'lin'
configparam['confounds'] = dict()
configparam['confounds']['train'] = np.random.randint(0,2,[training_samples_num,2]) 
configparam['confounds']['test'] = np.random.randint(0,2,[test_samples_num,2])
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)


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
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)

yhat, s2 = nm.predict(X, Y, Xs)

plt.scatter(Xs[configparam['confounds']['test'][:,0]==0,],Ys[configparam['confounds']['test'][:,0]==0,])
plt.scatter(Xs[configparam['confounds']['test'][:,0]==1,],Ys[configparam['confounds']['test'][:,0]==1,])
plt.plot(Xs[configparam['confounds']['test'][:,0]==0,],yhat[[configparam['confounds']['test'][:,0]==0,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==0,], 
                 yhat[configparam['confounds']['test'][:,0]==0,] - np.sqrt(s2[configparam['confounds']['test'][:,0]==0,]), 
                 yhat[configparam['confounds']['test'][:,0]==0,] + np.sqrt(s2[configparam['confounds']['test'][:,0]==0,]),
                 color='gray', alpha=0.2)
plt.plot(Xs[configparam['confounds']['test'][:,0]==1,],yhat[[configparam['confounds']['test'][:,0]==1,]])
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==1,], 
                 yhat[configparam['confounds']['test'][:,0]==1,] - np.sqrt(s2[configparam['confounds']['test'][:,0]==1,]), 
                 yhat[configparam['confounds']['test'][:,0]==1,] + np.sqrt(s2[configparam['confounds']['test'][:,0]==1,]),
                 color='gray', alpha=0.2)

########################### TESTING NON-LINEAR HBR ############################

# Simulating the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range= (-1,1))

X = np.random.randint(10,91,100)
Y = -X**2 * 0.02 + X * 2 + 5*np.random.randn(100)

Xs = np.arange(10,91)
Ys = -Xs**2 * 0.02 + Xs * 2 + 5*np.random.randn(len(Xs))

configparam = dict()
configparam['model_type'] = 'nonlin'
configparam['confounds'] = dict()
configparam['confounds']['train'] = np.random.randint(0,2,[100,2]) 
configparam['confounds']['test'] = np.random.randint(0,1,[len(Xs),2])


Y[configparam['confounds']['train'][:,0]==1,] = Y[configparam['confounds']['train'][:,0]==1,] + 2
Ys[configparam['confounds']['test'][:,0]==1,] = Ys[configparam['confounds']['test'][:,0]==1,] + 2

Y = scaler.fit_transform(Y.reshape(-1, 1)).squeeze()
Ys = scaler.transform(Ys.reshape(-1, 1)).squeeze()

with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)

# Running the model
nm = norm_init(X, Y, alg='hbr', configparam='configs.pkl')
nm.estimate(X, Y)

yhat, s2 = nm.predict(X, Y, Xs)

plt.scatter(Xs[configparam['confounds']['test'][:,0]==0,],Ys[configparam['confounds']['test'][:,0]==0,])

plt.plot(Xs,yhat)
plt.fill_between(Xs[configparam['confounds']['test'][:,0]==0,], 
                 yhat[configparam['confounds']['test'][:,0]==0,] - np.sqrt(s2[configparam['confounds']['test'][:,0]==0,]), 
                 yhat[configparam['confounds']['test'][:,0]==0,] + np.sqrt(s2[configparam['confounds']['test'][:,0]==0,]),
                 color='gray', alpha=0.2)

###############################################################################
