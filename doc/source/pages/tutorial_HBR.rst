Hierarchical Bayesian Regression Tutorial
============================================================================================================

Hierarchical Bayesian regression (HBR) is especially suited to deal with multi-site datasets, and allows for transfer learning (e.g., prediction for unseen sites) and can be estimated in a federated learning framework

DRAFT
*******************************************

0. Prepare inputs

1. Estimate from data OR Transfer from already trained models, onto new sites/scanners

2. Predict unseen data from known scanners



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
   

Similarly to BLR, once estimated, you can check the models performance (eg Pearsons' correlations, errors) and obviously the predictive mean and variance, and  associted Z scores, of the models for the various features (ROIs) of the test data:

Interpreting model performance
*****************************************

Output evaluation metrics definitions

=================   ======================================================================================================
**key value**       **Description** 
-----------------   ------------------------------------------------------------------------------------------------------ 
yhat                predictive mean 
ys2                 predictive variance 
nm                  normative model 
Z                   deviance scores 
Rho                 Pearson correlation between true and predicted responses 
pRho                parametric p-value for this correlation 
RMSE                root mean squared error between true/predicted responses 
SMSE                standardised mean squared error 
EV                  explained variance 
MSLL                mean standardized log loss `See page 23 <http://www.gaussianprocess.org/gpml/chapters/RW2.pdf>`_
=================   ======================================================================================================



TRANSFERING
*******************************************
One major benefit of this HBR approahc is the possibility to transfer the models to unseen data while taking advantage of the previously learned distributions.



PREDICTING
*******************************************

Naturally, you may then want to apply these normative models onto new data coming from the same scanner sites used in the estimation of the models.
The process is very similar, but as you do not need to retrain the model, there is obviously no need for training data files. The predict() function thus requires only the covariates file. If the test responses are also specified then quantities that depend on those will also be returned (Z scores and error metrics).

.. code:: ipython3

    model_path = processing_dir + 'Models/'  # point to wherever you have stored the normative models estimated previously.
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
                                        outputsuffix='_predict')


Similarly to the estimate() function, you will get the predictive mean and variance and,  along with the Z scores for each of the provided sample and ROI.


