#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:06:06 2019

@author: seykia
"""

import torch
from torch import nn
from torch.nn import functional as F

##################################### NP Model ################################

class NP(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(NP, self).__init__()
        self.r_dim = encoder.r_dim
        self.z_dim = encoder.z_dim
        self.dp_level = encoder.dp_level
        self.encoder = encoder
        self.decoder = decoder
        self.r_to_z_mean_dp = nn.Dropout(p = self.dp_level)
        self.r_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar_dp = nn.Dropout(p = self.dp_level)
        self.r_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)
        self.device = args.device
        self.type = args.type
        
    def xy_to_z_params(self, x, y):
        r = self.encoder.forward(x, y)
        mu = self.r_to_z_mean(self.r_to_z_mean_dp(r))
        logvar = self.r_to_z_logvar(self.r_to_z_logvar_dp(r))
        return mu, logvar
      
    def reparameterise(self, z):
        mu, logvar = z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        return z_sample

    def forward(self, x_context, y_context, x_all=None, y_all=None, n = 10):
        y_sigma = None
        z_context = self.xy_to_z_params(x_context, y_context)
        if self.training:
            z_all = self.xy_to_z_params(x_all, y_all)
            z_sample = self.reparameterise(z_all)
            y_hat = self.decoder.forward(z_sample, x_all)
        else:  
            z_all = z_context
            if self.type == 'ST':
                temp = torch.zeros([n,y_context.shape[0], y_context.shape[2]], device = 'cpu')
            elif self.type == 'MT':
                temp = torch.zeros([n,y_context.shape[0],1,y_context.shape[2],y_context.shape[3],
                                y_context.shape[4]], device = 'cpu')                                
            for i in range(n):
                z_sample = self.reparameterise(z_all)
                temp[i,:] = self.decoder.forward(z_sample, x_context)
            y_hat = torch.mean(temp, dim=0).to(self.device)
            if n > 1:
                y_sigma = torch.std(temp, dim=0).to(self.device)
        return y_hat, z_all, z_context, y_sigma
    
###############################################################################
        
def apply_dropout_test(m):
    if type(m) == nn.Dropout:
        m.train()

def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / (var_p) \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div

def np_loss(y_hat, y, z_all, z_context):
    BCE = F.binary_cross_entropy(torch.squeeze(y_hat), torch.mean(y,dim=1), reduction="sum")
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD
