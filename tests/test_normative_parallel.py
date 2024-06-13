# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:34:06 2017

@author: andmar
"""

import os
import numpy as np
import pandas as pd
from pcntoolkit.normative_parallel import execute_nm

# configs
# specify your python path. Make sure you are using the Python in the right environement.
python_path = '/path/to/my/python'  

# specify the working directory to sacve the results.
processing_dir = '/path/to/my/test/directory/'
sample_num = 50
resp_num = 10
cov_num = 1

# simulating data
pd.DataFrame(np.random.random([sample_num, resp_num])).to_pickle(os.path.join(processing_dir,'train_resp.pkl'))
pd.DataFrame(np.random.random([sample_num, cov_num])).to_pickle(os.path.join(processing_dir,'train_cov.pkl'))
pd.DataFrame(np.random.random([sample_num, resp_num])).to_pickle(os.path.join(processing_dir,'test_resp.pkl'))
pd.DataFrame(np.random.random([sample_num, cov_num])).to_pickle(os.path.join(processing_dir,'test_cov.pkl'))


respfile = os.path.join(processing_dir,'train_resp.pkl')
covfile = os.path.join(processing_dir,'train_cov.pkl')

testresp = os.path.join(processing_dir,'test_resp.pkl')
testcov = os.path.join(processing_dir,'test_cov.pkl')

job_name = 'nmp_test'
batch_size = 1
memory = '4gb'
duration = '01:00:00'
cluster = 'slurm'
binary='True'

execute_nm(processing_dir, python_path, job_name, covfile, respfile,
           testcovfile_path=testcov, testrespfile_path=testresp, batch_size=batch_size, 
           memory=memory, duration=duration, cluster_spec=cluster,
           log_path=processing_dir, interactive='auto', binary=binary,
           savemodel='True', saveoutput='True')


