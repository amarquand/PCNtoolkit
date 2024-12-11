#!/usr/bin/env python
import sys

from pcntoolkit.normative_parallel import execute_nm


def execute_nm_wrapper(*args):
    args_dict = {k:v for k,v in [arg.split('=') for arg in args]}

    func = args_dict.get('func')
    covfile_path = args_dict.get('covfile_path',None)
    respfile_path = args_dict.get('respfile_path',None)
    testcovfile_path = args_dict.get('testcovfile_path',None)
    testrespfile_path = args_dict.get('testrespfile_path',None) 

    execute_nm(
        python_path='/home/preclineu/stijdboe/.conda/envs/pcntk_dev/bin/python',
        normative_path="/home/preclineu/stijdboe/.conda/envs/pcntk_dev/lib/python3.12/site-packages/pcntoolkit/normative.py",
        job_name='test_normative_parallel',
        processing_dir='/project/3022000.05/projects/stijdboe/temp/parallel_processing',
        func=func,
        covfile_path=covfile_path,
        respfile_path=respfile_path,
        testcovfile_path=testcovfile_path,
        testrespfile_path=testrespfile_path,
        batch_size=2,
        memory='16G',
        duration='00:10:00',
        job_id=1,
        cv_folds = 5,
        alg='hbr',
        warp='WarpSinArcsinh',
        optimizer='l-bfgs-b',
        warp_reparam=True,
        inscaler='standardize',
        binary=False
        )


def main(*args):
    execute_nm_wrapper(*args)

if __name__ == "__main__":
    main(sys.argv[1:])
