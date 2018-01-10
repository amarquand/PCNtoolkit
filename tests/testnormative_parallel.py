# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:34:06 2017

@author: andmar
"""
#import nispat
import os
import sys
sys.path.append('/home/mrstats/andmar/sfw/nispat/nispat')
from normative import estimate
from normative_parallel import execute_nm, collect_nm, delete_nm

#wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
#maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
#respfile = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
#covfile = os.path.join(wdir, 'covariates_basic_n50.txt')
#cvfolds = 2
#estimate(respfile, covfile, maskfile, cvfolds=cvfolds)

wdir = '/home/mrstats/andmar/py.sandbox/normative_oslo/'
respfile = os.path.join(wdir, 'ICA100_oslo15_v2_spaces.txt')
covfile = os.path.join(wdir, 'cov_oslo15_spaces.txt')
cvfolds = 2
#estimate(respfile, covfile, cvfolds=cvfolds)

# with test covariates
#wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
#maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
#respfile = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
#covfile = os.path.join(wdir, 'covariates_basic_n50.txt')
#testresp = os.path.join(wdir, 'shoot_data_3mm_last100.nii.gz')
#testcov = os.path.join(wdir, 'covariates_basic_last100.txt')
#estimate(reszfile, covfile, maskfile, testresp=testresp, testcov=testcov)

python_path = '/home/mrstats/andmar/.conda/envs/py36/bin/python'
normative_path = '/home/mrstats/andmar/sfw/nispat/nispat/normative.py'
job_name = 'nmp_test'
batch_size = 10
memory = '4gb'
duration = '01:00:00'

execute_nm(wdir, python_path, normative_path, job_name, covfile,  respfile,
           batch_size, memory, duration, cv_folds=cvfolds)

#collect_nm(wdir)
#delete_nm(wdir)


#wdir = '/home/mrstats/andmar/py.sandbox/normative_hcp'
#filename = os.path.join(wdir, 'tfmri_gambling_cope2.dtseries.nii')
#covfile = os.path.join(wdir, 'ddscores.txt')
#Nfold = 2

#wdir = '/home/mrstats/andmar/data/enigma_mdd'
#maskfile = None
#filename = os.path.join(wdir, 'Enigma_1st_100sub_resp.txt')
#covfile = os.path.join(wdir, 'Enigma_1st_100sub_cov.txt')
#Nfold = 2