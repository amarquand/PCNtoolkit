Hierarchical Bayesian Regressian
============================================================================================================

DRAFT
*******************************************

1. Estimate from scratch

OR

1. Transfer from already trained models, onto new sites/scanners

2. Predict

















ESTIMATING
*******************************************

.. code:: ipython3

 output_path = processing_dir + 'Models/'
    log_dir = output_path + 'log/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    
    respfile = processing_dir + 'Y_train.pkl'
    covfile = processing_dir + 'X_train.pkl'
    
    testrespfile_path = processing_dir + 'Y_test.pkl'
    testcovfile_path = processing_dir + 'X_test.pkl'
    
    trbefile = processing_dir + 'trbefile.pkl'
    tsbefile = processing_dir + 'tsbefile.pkl'
        
    python_path = '/home/preclineu/pieber/.conda/envs/trans/bin/python'
    
    batch_size = 1
    memory = '3gb'
    duration = '06:00:00'
    outputsuffix = '_estimate'
    
    parallel = True
    if not parallel:
        yhat, s2, z = ptk.normative.estimate(covfile=covfile, respfile=respfile,
                                        tsbefile=tsbefile, trbefile=trbefile, alg='hbr', log_path=log_dir, binary=True,
                                        standardize=standardize, output_path=output_path, testcov= testcovfile_path, 
                                        testresp = testrespfile_path,
                                        outputsuffix='out_version')
    else:
        ptk.normative_parallel.execute_nm(processing_dir, python_path=python_path,
                       job_name='Estimate_' + job_name, covfile_path=covfile, respfile_path=respfile, batch_size=batch_size, memory=memory, duration=duration,
                       normative_path=normative_path, func='estimate', alg='hbr', 
                       log_path=log_dir, binary=True, testcovfile_path = testcovfile_path, 
                       testrespfile_path = testrespfile_path, #standardize=standardize, 
                       trbefile=trbefile, tsbefile=tsbefile, model_type=method,
                       random_intercept=random_intercept, random_slope=random_slope, 
                       random_noise=random_noise, hetero_noise= hetero_noise, 
                       savemodel='True', saveoutput='True', outputsuffix=outputsuffix,
                       n_samples='1000')
        
        
        ptk.normative_parallel.collect_nm(processing_dir, 'Estimate_'+ job_name, func='estimate', 
                                      collect=False, binary=True, batch_size=batch_size,
                                      outputsuffix='_estimate')
        ptk.normative_parallel.rerun_nm(processing_dir, log_dir, memory, duration, binary=True)
        ptk.normative_parallel.collect_nm(processing_dir, 'Estimate_'+ job_name, func='estimate', 
                                      collect=True, binary=True, batch_size=batch_size,
                                      outputsuffix='_estimate')


PREDICTING
*******************************************



TRANSFERING
*******************************************
