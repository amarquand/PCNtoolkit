import os
import sys
import glob
import shutil

# Unit testing script for testing the basic functionality of this package. 
# This largely only tests input and output routines. More detailed unit testing 
# of underlying algorithms etc. should be done separately.
# To fully evaluate the user test cases, this should be run in two ways:

# 1. as a package
from pcntoolkit.normative import estimate
from pcntoolkit.normative_parallel import execute_nm, collect_nm, delete_nm

## 2. by appending to the path
##sys.path.clear()
#sys.path.append('/home/preclineu/andmar/sfw/PCNtoolkit/pcntoolkit')
#from normative import estimate
#from normative_parallel import execute_nm, collect_nm, delete_nm

# ---------------- Config parameters -----------------------------------------

# General config parameters
normative_path = '/home/preclineu/andmar/sfw/PCNtoolkit/pcntoolkit/normative.py'
python_path='/home/preclineu/andmar/sfw/anaconda3/envs/py36/bin/python'
data_dir = '/home/preclineu/andmar/data/nispat_unit_test_data/'
test_dir = '/home/preclineu/andmar/py.sandbox/unittests/unit_test_results'
alt_alg = 'blr'             # algorithm to test in addition to GPR

# cluster paramateters
job_name = 'nm_unit_test'
batch_size = 10
memory = '4gb'
duration = '01:00:00'
cluster = 'torque'

# ---------------- Utility functions -----------------------------------------

def update_test_counter(test_num, root_dir):  
    test_num += 1
    test_out_dir = os.path.join(test_dir,'test_'+str(test_num))    
    os.makedirs(test_out_dir, exist_ok = True)

    
    return test_num, test_out_dir

def save_output(src_dir, dst_dir):
    
    files = []
    files.extend(glob.glob(os.path.join(src_dir,'Z*')))
    files.extend(glob.glob(os.path.join(src_dir,'yhat*')))
    files.extend(glob.glob(os.path.join(src_dir,'ys2*')))
    files.extend(glob.glob(os.path.join(src_dir,'Rho*')))
    files.extend(glob.glob(os.path.join(src_dir,'pRho*')))
    files.extend(glob.glob(os.path.join(src_dir,'RMSE*')))
    files.extend(glob.glob(os.path.join(src_dir,'SMSE*')))
    files.extend(glob.glob(os.path.join(src_dir,'MSLL*')))
    files.extend(glob.glob(os.path.join(src_dir,'EXPV*')))
    files.extend(glob.glob(os.path.join(src_dir,'Hyp*')))
    files.extend(glob.glob(os.path.join(src_dir,'Models')))
    for f in files:
        fdir, fnam = os.path.split(f)
        shutil.move(f, os.path.join(dst_dir,fnam))
    return

# ---------------- Unit tests ------------------------------------------------

print('Starting unit testing ...')
if os.path.exists(test_dir):
    print('Removing existing directory')
    shutil.rmtree(test_dir)
os.makedirs(test_dir, exist_ok=True)
test_num, tdir = update_test_counter(0, test_dir)
print("\n")

print(test_num, "Testing basic config (gpr with nii data)...")
print("----------------------------------------------------------------------")

mask_file_nii = os.path.join(data_dir, 'mask3.nii.gz')
resp_file_nii = os.path.join(data_dir, 'resp_n50.nii.gz')
cov_file_nii = os.path.join(data_dir, 'cov_n50.txt')
resp_file_nii_te = os.path.join(data_dir, 'resp_n100.nii.gz')
cov_file_nii_te = os.path.join(data_dir, 'cov_n100.txt')

estimate(cov_file_nii, resp_file_nii, maskfile=mask_file_nii, 
          testresp = resp_file_nii_te, testcov = cov_file_nii_te)

print(os.getcwd())
save_output(os.getcwd(), tdir)
test_num, tdir = update_test_counter(test_num, test_dir)

print(test_num, "Testing again using the same data under cross-validation")
print("----------------------------------------------------------------------")

estimate(cov_file_nii, resp_file_nii, maskfile = mask_file_nii, cvfolds = 2)

save_output(os.getcwd(), tdir)
test_num, tdir = update_test_counter(test_num, test_dir)

print(test_num, "Test again with txt files. This time using blr...")
print("----------------------------------------------------------------------")

resp_file_txt = os.path.join(data_dir, 'resp.txt')
cov_file_txt = os.path.join(data_dir, 'cov.txt')

estimate(cov_file_txt, resp_file_txt, testresp = resp_file_txt, 
          testcov = cov_file_txt ,alg=alt_alg, configparam=2)

save_output(os.getcwd(), tdir)
test_num, tdir = update_test_counter(test_num, test_dir)

print(test_num, "Testing again using the same data under cross-validation")
print("----------------------------------------------------------------------")
estimate(cov_file_txt, resp_file_txt, cvfolds=2 ,alg=alt_alg, configparam=2)

save_output(os.getcwd(), tdir)
test_num, tdir = update_test_counter(test_num, test_dir)

print(test_num, "Testing larger dataset (blr with pkl data)...")
print("----------------------------------------------------------------------")

cov_file_tr = os.path.join(data_dir,'cov_big_tr.txt')
cov_file_te = os.path.join(data_dir,'cov_big_te.txt')
resp_file_tr = os.path.join(data_dir,'resp_big_tr.txt')
resp_file_te = os.path.join(data_dir,'resp_big_te.txt')

estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, testcov=cov_file_te,
          alg=alt_alg, configparam=1, savemodel='True')

save_output(os.getcwd(), tdir)
test_num, tdir = update_test_counter(test_num, test_dir)

print(test_num, "Testing normative_parallel (gpr with pkl data under cv)...")
print("----------------------------------------------------------------------")

resp_file_par = os.path.join(data_dir, 'resp.pkl')
cov_file_par = os.path.join(data_dir, 'cov.pkl')
bin_flag = True

tdir += '/'
execute_nm(tdir, python_path, job_name, cov_file_par, 
           resp_file_par, batch_size, memory, duration, cluster_spec=cluster, 
           cv_folds=2, log_path=tdir, binary=bin_flag)

# to be run after qsub jobs complete
#collect_nm(tdir, job_name, collect=True, binary=bin_flag)
#delete_nm(tdir, binary=bin_flag)