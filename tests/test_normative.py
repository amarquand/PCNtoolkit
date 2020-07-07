# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:34:06 2017

@author: andmar
"""
#import pcntoolkit
import os
import sys
#from pcntoolkit.normative import estimate
sys.path.append('/home/preclineu/andmar/sfw/PCNtoolkit/pcntoolkit')
from normative import estimate

#wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
##wdir = '/Users/andre/data/normative_nimg'
#maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
#respfile = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
#covfile = os.path.join(wdir, 'covariates_basic_n50.txt')
#cvfolds = 2

# with test covariates
wdir = '/home/preclineu/andmar/py.sandbox/normative_nimg'
##wdir = '/Users/andre/data/normative_nimg'
maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
respfile = os.path.join(wdir, 'shoot_data_3mm_n500.nii.gz')
covfile = os.path.join(wdir, 'covariates_basic_n500.txt')
testresp = os.path.join(wdir, 'shoot_data_3mm_last100.nii.gz')
testcov = os.path.join(wdir, 'covariates_basic_last100.txt')
estimate(covfile, respfile, maskfile=maskfile, testresp=testresp, testcov=testcov,alg="blr")#, configparam=4)
#cvfolds = 2


#wdir = '/home/mrstats/andmar/py.sandbox/normative_hcp'
#filename = os.path.join(wdir, 'tfmri_gambling_cope2.dtseries.nii')
#covfile = os.path.join(wdir, 'ddscores.txt')
#Nfold = 2

#wdir = '/home/mrstats/andmar/py.sandbox/normative_oslo'
#respfile = os.path.join(wdir, 'ICA100_oslo15_v2.txt')
#covfile = os.path.join(wdir, 'cov_oslo15_v2.txt')
#cvfolds = 2
#pcntoolkit.normative.estimate(covfile, respfile,  cvfolds=cvfolds)

#wdir = '/home/mrstats/andmar/data/enigma_mdd'
#maskfile = None
#filename = os.path.join(wdir, 'Enigma_1st_100sub_resp.txt')
#covfile = os.path.join(wdir, 'Enigma_1st_100sub_cov.txt')
#Nfold = 2