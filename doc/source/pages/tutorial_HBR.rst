Hierarchical Bayesian Regression Tutorial
============================================================================================================

Hierarchical Bayesian regression (HBR) is especially suited to deal with multi-site datasets, and allows for transfer learning (e.g., prediction for unseen sites) and can be estimated in a federated learning framework

DRAFT
*******************************************

0. Estimate from scratch

OR

1. Transfer from already trained models, onto new sites/scanners

2. Predict



INPUTS
*******************************************

You can (should?) have a look at how the input data needs to be formatted/curated in the associated tutorial made by Saige on normative models using Bayesian Linear Regression (BLR).
Here we will assume you have the data ready in such files, (step 4 done on the BLR tutorial):

.. code:: ipython3

    processing_dir = "/home/"    # replace with a path to your working directory
    respfile = processing_dir + 'Y_train.pkl'       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
    covfile = processing_dir + 'X_train.pkl'        # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)

    testrespfile_path = processing_dir + 'Y_test.pkl'       # measurements  for the testing samples
    testcovfile_path = processing_dir + 'X_test.pkl'        # covariate file for the testing samples

    trbefile = processing_dir + 'trbefile.pkl'      # training batch effects file (eg scanner_id, gender)  (columns: the various batch effects, rows: observations or subjects)
    tsbefile = processing_dir + 'tsbefile.pkl'      # testing batch effects file











ESTIMATING
*******************************************
In the BLR tutorial, models are estimated one by one for each ROI, here we show how to evaluate the whole data matix at once.


.. code:: ipython3

    output_path = processing_dir + 'Models/'    #  output path, where the models will be written
    log_dir = output_path + 'log/'              #
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
        
    outputsuffix = '_estimate'      # a string to name the output files, of use only to you, so adapt it for your needs.
        
    python_path = '/home/.conda/envs/dev/bin/python'         # path to your python install, here is mine as an example, within a conda environment.
    

    

    yhat, s2, z = ptk.normative.estimate(covfile=covfile, respfile=respfile,
                                         tsbefile=tsbefile, trbefile=trbefile, alg='hbr', log_path=log_dir, binary=True,
                                         standardize=standardize, output_path=output_path, testcov= testcovfile_path, 
                                         testresp = testrespfile_path,
                                         outputsuffix=outputsuffix)
   

PREDICTING
*******************************************
.. code:: ipython3

    model_path = processing_dir + 'Models/'
    output_path = os.path.join(processing_dir, 'output/')
    log_dir = output_path + 'log/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    

    # obviously here you only need 'test' data, validation dataa, 
    test_suffix = 'psy'         # naming 
    covfile = os.path.join(output_path, f'X_{test_suffix}.pkl')         # just point to where you have your 'new' data
    respfile = os.path.join(output_path, f'Y_{test_suffix}.pkl')
    tsbefile = os.path.join(output_path, f'tsbefile_{test_suffix}.pkl')

    yhat, s2, z = ptk.normative.predict(covfile=covfile, respfile=respfile, model_path=model_path,
                                        tsbefile=tsbefile, alg='hbr', log_path=log_dir, binary=True,
                                        standardize=standardize, output_path=output_path,
                                        outputsuffix='_psy_pred')

        
    shutil.move(processing_dir + f'yhat_{sample}.pkl', output_path + f'yhat_{sample}.pkl')
    shutil.move(processing_dir + f'ys2_{sample}.pkl', output_path + f'ys2_{sample}.pkl')
    
    for f in glob.glob(processing_dir + f'*_{sample}.pkl'):
        os.remove(f)



TRANSFERING
*******************************************
One major benefit of this HBR approahc is the possibility to transfer the models to unseen data while taking advantage of the previously learned distributions.
