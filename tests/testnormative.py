# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:34:06 2017

@author: andmar
"""
from normative import main as run_normative

wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
#wdir = '/Users/andre/data/normative_nimg'
maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
filename = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
covfile = os.path.join(wdir, 'covariates_basic_n50.txt')
Nfold = 2

wdir = '/home/mrstats/andmar/py.sandbox/normative_hcp'
filename = os.path.join(wdir, 'tfmri_gambling_cope2.dtseries.nii')
covfile = os.path.join(wdir, 'ddscores.txt')
Nfold = 2

#wdir = '/home/mrstats/andmar/data/enigma_mdd'
#maskfile = None
#filename = os.path.join(wdir, 'Enigma_1st_100sub_resp.txt')
#covfile = os.path.join(wdir, 'Enigma_1st_100sub_cov.txt')
#Nfold = 2

os.chdir(wdir)
args = ['',filename,'-c ',covfile,'-k','2']
run_normative(args)