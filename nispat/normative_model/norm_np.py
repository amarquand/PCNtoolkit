#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:28:22 2019

@author: seykia
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle

try:  # run as a package if installed
    from nispat.normative_model.normbase import NormBase
    from nispat.NP import NP, apply_dropout_test, np_loss
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from NP import NP, apply_dropout_test, np_loss
    from norm_base import NormBase
    
class Encoder(nn.Module):
    def __init__(self, x, y, args):
        super(Encoder, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.dp_level = args.dp
        self.hidden_neuron_num = args.hidden_neuron_num
        
        self.h_1 = nn.Linear(x.shape[1] + y.shape[1], self.hidden_neuron_num)
        self.h_2_dp = nn.Dropout(p=self.dp_level)
        self.h_2 = nn.Linear(self.hidden_neuron_num, self.hidden_neuron_num)
        self.h_3_dp = nn.Dropout(p=self.dp_level)
        self.h_3 = nn.Linear(self.hidden_neuron_num, self.r_dim)

    def forward(self, x, y):
        x_y = torch.cat([x, y], dim=2)
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(self.h_2_dp(x_y)))
        x_y = F.relu(self.h_3(self.h_3_dp(x_y)))
        r = torch.mean(x_y, dim=1)
        return r
    
    
class Decoder(nn.Module):
    def __init__(self, x, y, args):
        super(Decoder, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.dp_level = args.dp
        self.hidden_neuron_num = args.hidden_neuron_num
        
        self.g_1 = nn.Linear(self.z_dim + x.shape[1], self.hidden_neuron_num)
        self.g_2_dp = nn.Dropout(p=self.dp_level)
        self.g_2 = nn.Linear(self.hidden_neuron_num, self.hidden_neuron_num)
        self.g_3_dp = nn.Dropout(p=self.dp_level)
        self.g_3 = nn.Linear(self.hidden_neuron_num, y.shape[1])
    
    def forward(self, z_sample, x_target):
        z_x = torch.cat([z_sample, torch.mean(x_target,dim=1)], dim=1)
        z_x = F.relu(self.g_1(z_x))
        z_x = F.relu(self.g_2(self.g_2_dp(z_x)))
        y_hat = torch.sigmoid(self.g_3(self.g_3_dp(z_x)))
        return y_hat
   

class NormNP(NormBase):
    """ Classical GPR-based normative modelling approach
    """
    
    def __init__(self, X, y, configparam):
        self.configparam = configparam
        with open(configparam, 'rb') as handle:
             config = pickle.load(handle)
        class struct(object):
            pass
        args = struct()
        args.batch_size = config['batch_size']
        args.epochs = config['epochs']
        args.device = config['device']
        args.m = config['m']
        args.dp = config['dp']
        args.hidden_neuron_num = config['hidden_neuron_num']
        args.r_dim = config['r_dim']
        args.z_dim = config['z_dim']
        args.type = 'ST'
        
        if (X is not None):
            self.args = args
            self.encoder = Encoder(X, y, args)
            self.decoder = Decoder(X, y, args)
            self.model = NP(self.encoder, self.decoder, args)
        else:
            raise(ValueError, 'please specify covariates')
            return
        
    @property
    def n_params(self):
        return 1
    
    @property
    def neg_log_lik(self):
        return -1
    
    def estimate(self, X, y):
        sample_num = X.shape[0]
        batch_size = self.args.batch_size
        factor_num = self.args.m
        mini_batch_num = int(np.ceil(sample_num/batch_size))
        device = self.args.device
        
        self.scaler = MinMaxScaler()
        y = self.scaler.fit_transform(y)

        
        self.reg = []
        for i in range(factor_num):
            self.reg.append(LinearRegression())
            idx = np.random.randint(0, sample_num, 2)
            self.reg[i].fit(X[idx].reshape(-1, 1),y[idx,:])
        
        x_context = np.zeros([sample_num, factor_num, X.shape[1]])
        y_context = np.zeros([sample_num, factor_num, 1])
        
        
        for j in range(factor_num):
            x_context[:,j,:] = X
            y_context[:,j,:] = self.reg[j].predict(x_context[:,j,:])
        
        x_context = torch.tensor(x_context, device=device, dtype = torch.float)
        y_context = torch.tensor(y_context, device=device, dtype = torch.float)
        
        x_all = torch.tensor(X.reshape(-1, 1, 1), device=device, dtype = torch.float)
        y_all = torch.tensor(y.reshape(-1, 1, y.shape[1]), device=device, dtype = torch.float)        
        
        self.model.train()
        epochs = [int(self.args.epochs/4),int(self.args.epochs/2),int(self.args.epochs/5),
                  int(self.args.epochs-self.args.epochs/4-self.args.epochs/2-self.args.epochs/5)]
        k = 1
        for e in range(len(epochs)): 
            optimizer = optim.Adam(self.model.parameters(), lr=10**(-e-2))
            for j in range(epochs[e]):
                train_loss = 0
                for i in range(mini_batch_num):
                    optimizer.zero_grad()
                    idx = np.arange(i*batch_size,(i+1)*batch_size)
                    y_hat, z_all, z_context, dummy = self.model(x_context[idx,:,:], y_context[idx,:,:], x_all[idx,:,:], y_all[idx,:,:])
                    loss = np_loss(y_hat, y_all[idx,0,:], z_all, z_context)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                print('Epoch: %d, Loss:%f' %( k, train_loss))
                k += 1
        return None
        
    def predict(self, Xs): 
        sample_num = Xs.shape[0]
        factor_num = self.args.m
        x_context_test = np.zeros([sample_num, factor_num, Xs.shape[1]])
        y_context_test = np.zeros([sample_num, factor_num, 1])
        for j in range(factor_num):
            x_context_test[:,j,:] = Xs
            y_context_test[:,j,:] = self.reg[j].predict(x_context_test[:,j,:])
        x_context_test = torch.tensor(x_context_test, device=self.args.device, dtype = torch.float)
        y_context_test = torch.tensor(y_context_test, device=self.args.device, dtype = torch.float)
        self.model.eval()
        self.model.apply(apply_dropout_test)
        with torch.no_grad():
            y_hat, z_all, z_context, y_sigma = self.model(x_context_test, y_context_test, n = 100)
            
        y_hat = self.scaler.inverse_transform(y_hat.cpu().numpy())
        y_sigma = y_sigma.cpu().numpy() * (self.scaler.data_max_ - self.scaler.data_min_)

        return y_hat, y_sigma**2