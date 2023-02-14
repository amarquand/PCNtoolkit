#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:41:07 2019

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
    from pcntoolkit.normative_model.normbase import NormBase
    from pcntoolkit.model.NPR import NPR, np_loss
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from model.NPR import NPR, np_loss
    from norm_base import NormBase

class struct(object):
    pass
   
class Encoder(nn.Module):
    def __init__(self, x, y, args):
        super(Encoder, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.hidden_neuron_num = args.hidden_neuron_num
        self.h_1 = nn.Linear(x.shape[1] + y.shape[1], self.hidden_neuron_num)
        self.h_2 = nn.Linear(self.hidden_neuron_num, self.hidden_neuron_num)
        self.h_3 = nn.Linear(self.hidden_neuron_num, self.r_dim)

    def forward(self, x, y):
        x_y = torch.cat([x, y], dim=2)
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(x_y))
        x_y = F.relu(self.h_3(x_y))
        r = torch.mean(x_y, dim=1)
        return r
    
    
class Decoder(nn.Module):
    def __init__(self, x, y, args):
        super(Decoder, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.hidden_neuron_num = args.hidden_neuron_num
        
        self.g_1 = nn.Linear(self.z_dim, self.hidden_neuron_num)
        self.g_2 = nn.Linear(self.hidden_neuron_num, self.hidden_neuron_num)
        self.g_3 = nn.Linear(self.hidden_neuron_num, y.shape[1])
        
        self.g_1_84 = nn.Linear(self.z_dim, self.hidden_neuron_num)
        self.g_2_84 = nn.Linear(self.hidden_neuron_num, self.hidden_neuron_num)
        self.g_3_84 = nn.Linear(self.hidden_neuron_num, y.shape[1])
    
    def forward(self, z_sample):
        z_hat = F.relu(self.g_1(z_sample))
        z_hat = F.relu(self.g_2(z_hat))
        y_hat = torch.sigmoid(self.g_3(z_hat))
        
        z_hat_84 = F.relu(self.g_1(z_sample))
        z_hat_84 = F.relu(self.g_2_84(z_hat_84))
        y_hat_84 = torch.sigmoid(self.g_3_84(z_hat_84))
        
        return y_hat, y_hat_84
   


            
class NormNP(NormBase):
    """ Classical GPR-based normative modelling approach
    """
    
    def __init__(self, X, y, configparam=None):
        self.configparam = configparam
        if configparam is not None: 
            with open(configparam, 'rb') as handle:
                 config = pickle.load(handle)
            args = struct()
            if 'batch_size' in config:
                args.batch_size = config['batch_size']
            else:
                args.batch_size = 10
            if 'epochs' in config:
                args.epochs = config['epochs']
            else:
                args.epochs = 100
            if 'device' in config:
                args.device = config['device']
            else:
                args.device = torch.device('cpu')
            if 'm' in config:
                args.m = config['m']
            else:
                args.m = 200
            if 'hidden_neuron_num' in config:
                args.hidden_neuron_num = config['hidden_neuron_num']
            else:
                args.hidden_neuron_num = 10
            if 'r_dim' in config:
                args.r_dim = config['r_dim']
            else:
                args.r_dim = 5
            if 'z_dim' in config:
                args.z_dim = config['z_dim']
            else:
                args.z_dim = 3
            if 'nv' in config:
                args.nv = config['nv']
            else:
                args.nv = 0.01
        else:
            args = struct()
            args.batch_size = 10
            args.epochs = 100
            args.device = torch.device('cpu')
            args.m = 200
            args.hidden_neuron_num = 10
            args.r_dim = 5
            args.z_dim = 3
            args.nv = 0.01
        
        if y is not None:
            if y.ndim == 1:
                y = y.reshape(-1,1)
            self.args = args
            self.encoder = Encoder(X, y, args)
            self.decoder = Decoder(X, y, args)
            self.model = NPR(self.encoder, self.decoder, args)
       
        
    @property
    def n_params(self):
        return 1
    
    @property
    def neg_log_lik(self):
        return -1
    
    def estimate(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1,1)
        sample_num = X.shape[0]
        batch_size = self.args.batch_size
        factor_num = self.args.m
        mini_batch_num = int(np.floor(sample_num/batch_size))
        device = self.args.device
        
        self.scaler = MinMaxScaler()
        y = self.scaler.fit_transform(y)
 
        self.reg = []
        for i in range(factor_num):
            self.reg.append(LinearRegression())
            idx = np.random.randint(0, sample_num, sample_num)#int(sample_num/10))
            self.reg[i].fit(X[idx,:],y[idx,:])
        
        x_context = np.zeros([sample_num, factor_num, X.shape[1]])
        y_context = np.zeros([sample_num, factor_num, 1])
        
        s = X.std(axis=0)
        for j in range(factor_num):
            x_context[:,j,:] = X + np.sqrt(self.args.nv) * s * np.random.randn(X.shape[0], X.shape[1])
            y_context[:,j,:] = self.reg[j].predict(x_context[:,j,:])
        
        x_context = torch.tensor(x_context, device=device, dtype = torch.float)
        y_context = torch.tensor(y_context, device=device, dtype = torch.float)
        
        x_all = torch.tensor(np.expand_dims(X,axis=1), device=device, dtype = torch.float)
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
                    y_hat, y_hat_84, z_all, z_context, dummy, dummy = self.model(x_context[idx,:,:], y_context[idx,:,:], x_all[idx,:,:], y_all[idx,:,:])
                    loss = np_loss(y_hat, y_hat_84, y_all[idx,0,:], z_all, z_context)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                print('Epoch: %d, Loss:%f' %( k, train_loss))
                k += 1
        return self
        
    def predict(self, Xs, X=None, Y=None, theta=None): 
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
        with torch.no_grad():
            y_hat, y_hat_84, z_all, z_context, y_sigma, y_sigma_84 = self.model(x_context_test, y_context_test, n = 100)
            
        y_hat = self.scaler.inverse_transform(y_hat.cpu().numpy())
        y_hat_84 = self.scaler.inverse_transform(y_hat_84.cpu().numpy())
        y_sigma = y_sigma.cpu().numpy() * (self.scaler.data_max_ - self.scaler.data_min_)
        y_sigma_84 = y_sigma_84.cpu().numpy() * (self.scaler.data_max_ - self.scaler.data_min_)
        sigma_al = y_hat - y_hat_84
        return y_hat.squeeze(), (y_sigma**2 + sigma_al**2).squeeze() #, z_context[0].cpu().numpy(), z_context[1].cpu().numpy()
    