#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:47:01 2019

@author: seykia
"""
# ------------------------------------------------------------------------------
#  Usage:
#  python normative_NP.py -r /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/responses.nii.gz 
#                           -c /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/covariates.pickle 
#                           --tr /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/test_responses.nii.gz 
#                           --tc /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/test_covariates.pickle 
#                           -o /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/Results
#
#
#  Written by S. M. Kia
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import sys
import argparse
import torch
from torch import optim
import numpy as np
import pickle
from NP import NP, apply_dropout_test, np_loss
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from architecture import Encoder, Decoder
from nispat.utils import compute_pearsonr, explained_var, compute_MSLL
from nispat.utils import extreme_value_prob, extreme_value_prob_fit, ravel_2D, unravel_2D
from nispat import fileio

############################ Parsing inputs ################################### 

def get_args(*args):
    """ Parse command line arguments"""

    # parse arguments
    parser = argparse.ArgumentParser(description='Neural Processes (NP) for Deep Normative Modeling')
    parser.add_argument("-r", help="Training response pickle file address", 
                        required=True, dest="respfile", default=None)
    parser.add_argument("-c", help="Training covariates pickle file address", 
                        required=True, dest="covfile", default=None)
    parser.add_argument("--tc", help="Test covariates pickle file address", 
                        required=True, dest="testcovfile", default=None)
    parser.add_argument("--tr", help="Test response pickle file address", 
                        required=True, dest="testrespfile", default=None)
    parser.add_argument("-o", help="Output directory address", dest="outdir", default=None)
    parser.add_argument('-m', type=int, default=10, dest='m',
                        help='number of fixed-effect estimations')
    parser.add_argument('--batchnum', type=int, default=10, dest='batchnum',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, dest='epochs',
                        help='number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', dest='device',
                        help='Either cpu or cuda')

    args = parser.parse_args()


    cuda = args.device=='cuda' and torch.cuda.is_available()
    args.device = torch.device("cuda" if cuda else "cpu")
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    return args
    
################################## Training ###################################
def estimate(args):
    print('Loading the input Data ...')
    responses = fileio.load_nifti(args.respfile, vol=True).transpose([3,0,1,2])
    with open(args.covfile, 'rb') as handle:
        covariates = pickle.load(handle)['covariates']
    test_responses = fileio.load_nifti(args.testrespfile, vol=True).transpose([3,0,1,2])
    with open(args.testcovfile, 'rb') as handle:
        test_covariates = pickle.load(handle)['test_covariates']
        
    
    mask = fileio.create_mask(responses[0,:,:,:], mask=None)
    
    response_shape = responses.shape
    test_responses_shape = test_responses.shape
    
    covariates_scaler = StandardScaler()
    covariates = covariates_scaler.fit_transform(covariates)
    test_covariates = covariates_scaler.transform(test_covariates)
    response_scaler = QuantileTransformer()
    responses = unravel_2D(response_scaler.fit_transform(ravel_2D(responses)), response_shape)
    test_responses = unravel_2D(response_scaler.transform(ravel_2D(test_responses)), test_responses_shape)
    
    responses =torch.tensor(responses, device = args.device)
    covariates =torch.tensor(covariates, device = args.device)
    test_responses =torch.tensor(test_responses, device = args.device)
    test_covariates =torch.tensor(test_covariates, device = args.device)
    
    factor = args.m
    
    x_context = torch.zeros([covariates.shape[0], factor, covariates.shape[1]], device=args.device)
    y_context = torch.zeros([responses.shape[0], factor, responses.shape[1], 
                             responses.shape[2], responses.shape[3]], device=args.device)
    x_all = torch.zeros([covariates.shape[0], factor, covariates.shape[1]], device=args.device)
    y_all = torch.zeros([responses.shape[0], factor, responses.shape[1], 
                             responses.shape[2], responses.shape[3]], device=args.device)
    x_context_test = torch.zeros([test_covariates.shape[0], factor, test_covariates.shape[1]], device=args.device)
    y_context_test = torch.zeros([test_responses.shape[0], factor, test_responses.shape[1], 
                             test_responses.shape[2], test_responses.shape[3]], device=args.device)
    
    for i in range(factor):
        x_context[:,i,:] = covariates[:,:]
        idx = np.random.randint(0,covariates.shape[0], covariates.shape[0])
        for j in range(responses.shape[1]):
            for k in range(responses.shape[2]):
                for l in range(responses.shape[3]):
                    reg = LinearRegression()
                    reg.fit(x_context[idx,i,:].cpu().numpy(), responses[idx,j,k,l].cpu().numpy())
                    y_context[:,i,j,k,l] = torch.tensor(reg.predict(x_context[:,i,:].cpu().numpy()), device=args.device)    
                    y_context_test[:,i,j,k,l] = torch.tensor(reg.predict(x_context_test[:,i,:].cpu().numpy()), device=args.device)
        print('Fixed-effect %d of %d is computed!' %(i+1, factor))
    
    x_all = x_context
    y_all = responses.unsqueeze(1).expand(-1,factor,-1,-1,-1)
    
    y_test = test_responses.unsqueeze(1)
    #x_test = test_covariates.view((test_covariates.shape[0],1,test_covariates.shape[1]))
    
    encoder = Encoder(x_context, y_context, args).to(args.device)
    decoder = Decoder(x_context, y_context, args).to(args.device)
    model = NP(encoder, decoder, args).to(args.device)
    
    k = 1
    epochs = [int(args.epochs/4),int(args.epochs/2),int(args.epochs/5),int(args.epochs-args.epochs/4-args.epochs/2-args.epochs/5)]
    mini_batch_num = args.batchnum
    batch_size = int(x_context.shape[0]/mini_batch_num)
    model.train()
    for e in range(len(epochs)): 
        optimizer = optim.Adam(model.parameters(), lr=10**(-e-2))
        for j in range(epochs[e]):
            train_loss = 0
            rand_idx = np.random.permutation(x_context.shape[0])
            for i in range(mini_batch_num):
                optimizer.zero_grad()
                idx = rand_idx[i*batch_size:(i+1)*batch_size]
                y_hat, z_all, z_context, dummy = model(x_context[idx,:,:], y_context[idx,:,:,:,:], x_all[idx,:,:], y_all[idx,:,:,:,:])
                loss = np_loss(y_hat, y_all[idx,:,:,:,:], z_all, z_context)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Epoch: %d, Loss:%f' %(k, train_loss))
            k += 1
            
    ################################## Evaluation #################################
            
    model.eval()
    model.apply(apply_dropout_test)
    with torch.no_grad():
        y_hat, z_all, z_context, y_sigma = model(x_context_test, y_context_test, n = 100)
        test_loss = np_loss(y_hat, y_test, z_all, z_context).item()
        print('Test Loss: %f' %(test_loss))
         
    RMSE = torch.sqrt(torch.mean((y_test - y_hat)**2, dim = 0)).squeeze().cpu().numpy() * np.int32(mask)
    SMSE = RMSE ** 2 / np.var(y_test.cpu().numpy(), axis=0).squeeze()
    Rho, pRho = compute_pearsonr(y_test.cpu().numpy().squeeze(), y_hat.cpu().numpy().squeeze())
    EXPV = explained_var(y_test.cpu().numpy().squeeze(), y_hat.cpu().numpy().squeeze()) * np.int32(mask)
    MSLL = compute_MSLL(y_test.cpu().numpy().squeeze(), y_hat.cpu().numpy().squeeze(), 
                        y_sigma.cpu().numpy().squeeze()**2, train_mean = y_test.cpu().numpy().mean(0), 
                        train_var = y_test.cpu().numpy().var(0)).squeeze() * np.int32(mask)
                        
    NPMs = (y_test - y_hat) / (y_sigma)
    NPMs = NPMs.squeeze()
    NPMs = NPMs.cpu().numpy() * np.int32(mask)
    NPMs = np.nan_to_num(NPMs)
    NPMs = NPMs.squeeze()
    
    temp=NPMs.reshape([NPMs.shape[0],NPMs.shape[1]*NPMs.shape[2]*NPMs.shape[3]])
    EVD_params = extreme_value_prob_fit(temp, 0.01)
    abnormal_probs = extreme_value_prob(EVD_params, temp, 0.01)
    
    ############################## SAVING RESULTS #################################
    
    if args.outdir is not None:
        print('Saving Results to: %s' %(args.outdir))
        exfile = args.testrespfile
        y_hat = y_hat.squeeze().cpu().numpy()
        y_hat = response_scaler.inverse_transform(ravel_2D(y_hat))
        y_hat = y_hat[:,mask.flatten()]
        fileio.save(y_hat.T, args.outdir + 
                    '/yhat.nii.gz', example=exfile, mask=mask)
        ys2 = y_sigma.squeeze().cpu().numpy()
        ys2 = response_scaler.inverse_transform(ravel_2D(ys2))
        ys2 = ys2**2
        ys2 = ys2[:,mask.flatten()]
        fileio.save(ys2.T, args.outdir + 
                    '/ys2.nii.gz', example=exfile, mask=mask) 
        NPMs = ravel_2D(NPMs)[:,mask.flatten()]
        fileio.save(NPMs.T, args.outdir +  
                    '/Z.nii.gz', example=exfile, mask=mask)
        fileio.save(Rho.flatten()[mask.flatten()], args.outdir +  
                    '/Rho.nii.gz', example=exfile, mask=mask)
        fileio.save(pRho.flatten()[mask.flatten()], args.outdir +  
                    '/pRho.nii.gz', example=exfile, mask=mask)
        fileio.save(RMSE.flatten()[mask.flatten()], args.outdir +  
                    '/rmse.nii.gz', example=exfile, mask=mask)
        fileio.save(SMSE.flatten()[mask.flatten()], args.outdir +  
                    '/smse.nii.gz', example=exfile, mask=mask)
        fileio.save(EXPV.flatten()[mask.flatten()], args.outdir +  
                    '/expv.nii.gz', example=exfile, mask=mask)
        fileio.save(MSLL.flatten()[mask.flatten()], args.outdir +  
                    '/msll.nii.gz', example=exfile, mask=mask)
        
        with open(args.outdir +'model.pkl', 'wb') as handle:
             pickle.dump({'model':model, 'reg':reg, 'covariates_scaler':covariates_scaler,
                          'response_scaler': response_scaler, 'EVD_params':EVD_params,
                          'abnormal_probs':abnormal_probs}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
###############################################################################
    print('DONE!')


def main(*args):
    """ Parse arguments and estimate model
    """
    
    np.seterr(invalid='ignore')
    args = get_args(args)
    estimate(args)
    
if __name__ == "__main__":
    main(sys.argv[1:])
