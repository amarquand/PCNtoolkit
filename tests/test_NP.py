#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:54:07 2019

@author: seykia
"""

from pcntoolkit.normative_NP import estimate


class struct(object):
    pass


args = struct()
args.batchnum = 10
args.epochs = 100
args.device = 'cuda'
args.estimator = 'ST'
args.mask = None
args.m = 10
args.respfile = '/project_freenas/3022017.02/Phenomics/responses.nii.gz'
args.covfile = '/project_freenas/3022017.02/Phenomics/covariates.pickle'
args.testrespfile = '/project_freenas/3022017.02/Phenomics/test_responses.nii.gz'
args.testcovfile = '/project_freenas/3022017.02/Phenomics/test_covariates.pickle'
args.outdir = '/project_freenas/3022017.02/Phenomics/Results'

estimate(args)
