#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:32:37 2019

@author: seykia
"""

import torch
from torch import nn
from torch.nn import functional as F

##################################### NP Model ################################

class NPR(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(NPR, self).__init__()
        self.r_dim = encoder.r_dim
        self.z_dim = encoder.z_dim
        self.encoder = encoder
        self.decoder = decoder
        self.r_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)
        self.device = args.device
        
    def xy_to_z_params(self, x, y):
        r = self.encoder.forward(x, y)
        mu = self.r_to_z_mean(r)
        logvar = self.r_to_z_logvar(r)
        return mu, logvar
      
    def reparameterise(self, z):
        mu, logvar = z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        return z_sample

    def forward(self, x_context, y_context, x_all=None, y_all=None, n = 10):
        y_sigma = None
        y_sigma_84 = None
        z_context = self.xy_to_z_params(x_context, y_context)
        if self.training:
            z_all = self.xy_to_z_params(x_all, y_all)
            z_sample = self.reparameterise(z_all)
            y_hat, y_hat_84 = self.decoder.forward(z_sample)
        else:  
            z_all = z_context
            temp = torch.zeros([n,y_context.shape[0], y_context.shape[2]], device = self.device)
            temp_84 = torch.zeros([n,y_context.shape[0], y_context.shape[2]], device = self.device)
            for i in range(n):
                z_sample = self.reparameterise(z_all)
                temp[i,:], temp_84[i,:] = self.decoder.forward(z_sample)
            y_hat = torch.mean(temp, dim=0).to(self.device)
            y_hat_84 = torch.mean(temp_84, dim=0).to(self.device)
            if n > 1:
                y_sigma = torch.std(temp, dim=0).to(self.device)
                y_sigma_84 = torch.std(temp_84, dim=0).to(self.device)
        return y_hat, y_hat_84, z_all, z_context, y_sigma, y_sigma_84
    
###############################################################################

def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / (var_p) \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div

def np_loss(y_hat, y_hat_84, y, z_all, z_context):
    #PBL = pinball_loss(y, y_hat, 0.05)
    BCE = F.binary_cross_entropy(torch.squeeze(y_hat), torch.mean(y,dim=1), reduction="sum")
    idx1 = (y >= y_hat_84).squeeze()
    idx2 = (y < y_hat_84).squeeze()
    BCE84 = 0.84 * F.binary_cross_entropy(torch.squeeze(y_hat_84[idx1,:]), torch.mean(y[idx1,:],dim=1), reduction="sum") + \
            0.16 * F.binary_cross_entropy(torch.squeeze(y_hat_84[idx2,:]), torch.mean(y[idx2,:],dim=1), reduction="sum")
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD + BCE84

