#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:54:07 2019

@author: seykia
"""

from nispat.normative_NP import estimate

class struct(object):
    pass
args = struct()
args.batchnum = 10
args.epochs = 100
args.device = 'cuda'
args.m = 10
args.respfile = '/home/preclineu/andmar/data/seykia/ds000030_R1.0.5/responses.nii.gz'
args.covfile = '/home/preclineu/andmar/data/seykia/ds000030_R1.0.5/covariates.pickle'
args.testrespfile = '/home/preclineu/andmar/data/seykia/ds000030_R1.0.5/test_responses.nii.gz'
args.testcovfile = '/home/preclineu/andmar/data/seykia/ds000030_R1.0.5/test_covariates.pickle'
args.outdir = '/home/preclineu/andmar/data/seykia/ds000030_R1.0.5/Results'

estimate(args)
