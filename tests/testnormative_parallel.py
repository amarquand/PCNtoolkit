# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:34:06 2017

@author: andmar
"""
#import nispat
import os
import sys
import time
sys.path.append('/home/mrstats/andmar/sfw/nispat/nispat')
from normative_parallel import execute_nm, collect_nm, delete_nm

#wdir = '/home/mrstats/andmar/py.sandbox/normative_oslo/'
#respfile = os.path.join(wdir, 'ICA100_oslo15_v2_spaces.txt')
#covfile = os.path.join(wdir, 'cov_oslo15_spaces.txt')
wdir = '/home/mrstats/andmar/py.sandbox/normative_batch_test/'
respfile = os.path.join(wdir, 'responses.txt')
covfile = os.path.join(wdir, 'covariates.txt')
cvfolds = 2
#estimate(respfile, covfile, cvfolds=cvfolds)

python_path = '/home/mrstats/andmar/sfw/anaconda3/envs/py36/bin/python'
normative_path = '/home/mrstats/andmar/sfw/nispat/nispat/normative.py'
job_name = 'nmp_test'
batch_size = 10
memory = '4gb'
duration = '01:00:00'
cluster = 'torque'

execute_nm(wdir, python_path, normative_path, job_name, covfile,  respfile,
           batch_size, memory, duration, cluster_spec=cluster, 
           cv_folds=cvfolds)#, alg='rfa')#, configparam=4)

print("waiting for jobs to finish ...")
time.sleep(60)

#collect_nm(wdir, collect=True)
#delete_nm(wdir)

