#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:26:35 2019

@author: seykia
"""

import pickle
import numpy as np
from nispat.normative_model.norm_utils import norm_init

########################### TESTING HLR #######################################

# Simulating the data
X = np.random.randint(10,90,100)
Y = np.random.randn(100)
Xs = np.random.randint(10,90,50)
configparam = dict()
configparam['train'] = np.random.randint(0,2,[100,2])
configparam['test'] = np.random.randint(0,2,[50,2])
with open('configs.pkl', 'wb') as file:
    pickle.dump(configparam,file)

# Running the model
nm = norm_init(X, Y, alg='hlr', configparam='configs.pkl')
nm.estimate(X, Y)
yhat, s2 = nm.predict(X, Y, Xs)

###############################################################################
