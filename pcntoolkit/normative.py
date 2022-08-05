#!/Users/andre/sfw/anaconda3/bin/python

# ------------------------------------------------------------------------------
#  Usage:
#  python normative.py -m [maskfile] -k [number of CV folds] -c <covariates>
#                      -t [test covariates] -r [test responses] <infile>
#
#  Either the -k switch or -t switch should be specified, but not both.
#  If -t is selected, a set of responses should be provided with the -r switch
#
#  Written by A. Marquand
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import argparse
import pickle
import glob

from sklearn.model_selection import KFold
try:  # run as a package if installed
    from pcntoolkit import configs
    from pcntoolkit.dataio import fileio
    from pcntoolkit.normative_model.norm_utils import norm_init
    from pcntoolkit.util.utils import compute_pearsonr, CustomCV, explained_var
    from pcntoolkit.util.utils import compute_MSLL, scaler, get_package_versions
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
        #sys.path.append(os.path.join(path,'normative_model'))
    del path
    
    import configs
    from dataio import fileio

    from util.utils import compute_pearsonr, CustomCV, explained_var, compute_MSLL
    from util.utils import scaler, get_package_versions
    from normative_model.norm_utils import norm_init

PICKLE_PROTOCOL = configs.PICKLE_PROTOCOL

def load_response_vars(datafile, maskfile=None, vol=True):
    """ load response variables (of any data type)"""

    if fileio.file_type(datafile) == 'nifti':
        dat = fileio.load_nifti(datafile, vol=vol)
        volmask = fileio.create_mask(dat, mask=maskfile)
        Y = fileio.vol2vec(dat, volmask).T
    else:
        Y = fileio.load(datafile)
        volmask = None
        if fileio.file_type(datafile) == 'cifti':
            Y = Y.T

    return Y, volmask


def get_args(*args):
    """ Parse command line arguments"""

    # parse arguments
    parser = argparse.ArgumentParser(description="Normative Modeling")
    parser.add_argument("responses")
    parser.add_argument("-f", help="Function to call", dest="func", 
                        default="estimate")
    parser.add_argument("-m", help="mask file", dest="maskfile", default=None)
    parser.add_argument("-c", help="covariates file", dest="covfile",
                        default=None)
    parser.add_argument("-k", help="cross-validation folds", dest="cvfolds",
                        default=None)
    parser.add_argument("-t", help="covariates (test data)", dest="testcov",
                        default=None)
    parser.add_argument("-r", help="responses (test data)", dest="testresp",
                        default=None)
    parser.add_argument("-a", help="algorithm", dest="alg", default="gpr")
    parser.add_argument("-x", help="algorithm specific config options", 
                        dest="configparam", default=None)
    # parser.add_argument('-s', action='store_false', 
    #                 help="Flag to skip standardization.", dest="standardize")
    parser.add_argument("keyword_args", nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Process required  arguemnts 
    wdir = os.path.realpath(os.path.curdir)
    respfile = os.path.join(wdir, args.responses)
    if args.covfile is None:
        raise(ValueError, "No covariates specified")
    else:
        covfile = args.covfile
    
    # Process optional arguments
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir, args.maskfile)
    if args.testcov is None and args.cvfolds is not None:
        testcov = None
        testresp = None
        cvfolds = int(args.cvfolds)
        print("Running under " + str(cvfolds) + " fold cross-validation.")
    else:
        print("Test covariates specified")
        testcov = args.testcov
        cvfolds = None
        if args.testresp is None:
            testresp = None
            print("No test response variables specified")
        else:
            testresp = args.testresp
        if args.cvfolds is not None:
            print("Ignoring cross-valdation specification (test data given)")

    # Process addtional keyword arguments. These are always added as strings
    kw_args = {}
    for kw in args.keyword_args:
        kw_arg = kw.split('=')
    
        exec("kw_args.update({'" +  kw_arg[0] + "' : " + 
                              "'" + str(kw_arg[1]) + "'" + "})")
    
    return respfile, maskfile, covfile, cvfolds, \
            testcov, testresp, args.func, args.alg, \
            args.configparam, kw_args
            

def evaluate(Y, Yhat, S2=None, mY=None, sY=None, nlZ=None, nm=None, Xz_tr=None, alg=None,
             metrics = ['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL']):
    ''' Compute error metrics
    This function will compute error metrics based on a set of predictions Yhat
    and a set of true response variables Y, namely:
    
    * Rho: Pearson correlation
    * RMSE: root mean squared error
    * SMSE: standardized mean squared error
    * EXPV: explained variance
        
    If the predictive variance is also specified the log loss will be computed
    (which also takes into account the predictive variance). If the mean and 
    standard deviation are also specified these will be used to standardize 
    this, yielding the mean standardized log loss
    
    :param Y: N x P array of true response variables
    :param Yhat: N x P array of predicted response variables
    :param S2: predictive variance
    :param mY: mean of the training set
    :param sY: standard deviation of the training set

    :returns metrics: evaluation metrics
    
    '''
    
    feature_num = Y.shape[1]
    
    # Remove metrics that cannot be computed with only a single data point 
    if Y.shape[0] == 1:
        if 'MSLL' in metrics:
            metrics.remove('MSLL')
        if 'SMSE' in metrics:
            metrics.remove('SMSE')
    
    # find and remove bad variables from the response variables
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]
    
    MSE = np.mean((Y - Yhat)**2, axis=0)
    
    results = dict()
    
    if 'RMSE' in metrics:
        RMSE = np.sqrt(MSE)
        results['RMSE'] = RMSE
    
    if 'Rho' in metrics:
        Rho = np.zeros(feature_num)
        pRho = np.ones(feature_num)    
        Rho[nz], pRho[nz] = compute_pearsonr(Y[:,nz], Yhat[:,nz])
        results['Rho'] = Rho
        results['pRho'] = pRho
        
    if 'SMSE' in metrics:
        SMSE = np.zeros_like(MSE)
        SMSE[nz] = MSE[nz] / np.var(Y[:,nz], axis=0)
        results['SMSE'] = SMSE
    
    if 'EXPV' in metrics:
        EXPV = np.zeros(feature_num)
        EXPV[nz] = explained_var(Y[:,nz], Yhat[:,nz])
        results['EXPV'] = EXPV
        
    if 'MSLL' in metrics:
        if ((S2 is not None) and (mY is not None) and (sY is not None)):
            MSLL = np.zeros(feature_num)
            MSLL[nz] = compute_MSLL(Y[:,nz], Yhat[:,nz], S2[:,nz], 
                                    mY.reshape(-1,1).T, 
                                    (sY**2).reshape(-1,1).T)
            results['MSLL'] = MSLL
            
    if 'NLL' in metrics:
        results['NLL'] = nlZ
    
    if 'BIC' in metrics:
        if hasattr(getattr(nm, alg), 'hyp'):
            n = Xz_tr.shape[0]
            k = len(getattr(nm, alg).hyp)
            BIC = k * np.log(n) + 2 * nlZ
            results['BIC'] = BIC    
    
    return results

def save_results(respfile, Yhat, S2, maskvol, Z=None, outputsuffix=None, 
                 results=None, save_path=''):
    
    print("Writing outputs ...")
    if respfile is None:
        exfile = None
        file_ext = '.pkl'
    else:
        if fileio.file_type(respfile) == 'cifti' or \
           fileio.file_type(respfile) == 'nifti':
            exfile = respfile
        else:
            exfile = None
        file_ext = fileio.file_extension(respfile)

    if outputsuffix is not None:
        ext = str(outputsuffix) + file_ext
    else:
        ext = file_ext

    fileio.save(Yhat, os.path.join(save_path, 'yhat' + ext), example=exfile, 
                                   mask=maskvol)
    fileio.save(S2, os.path.join(save_path, 'ys2' + ext), example=exfile, 
                mask=maskvol)
    if Z is not None:
        fileio.save(Z, os.path.join(save_path, 'Z' + ext), example=exfile, 
                    mask=maskvol)

    if results is not None:        
        for metric in list(results.keys()):
            if (metric == 'NLL' or metric == 'BIC') and file_ext == '.nii.gz':
                fileio.save(results[metric], os.path.join(save_path, metric + str(outputsuffix) + '.pkl'), 
                        example=exfile, mask=maskvol)
            else:
                fileio.save(results[metric], os.path.join(save_path, metric + ext), 
                            example=exfile, mask=maskvol)

def estimate(covfile, respfile, **kwargs):
    """ Estimate a normative model

    This will estimate a model in one of two settings according to 
    theparticular parameters specified (see below)
        
    * under k-fold cross-validation.
      requires respfile, covfile and cvfolds>=2
    * estimating a training dataset then applying to a second test dataset.
      requires respfile, covfile, testcov and testresp.
    * estimating on a training dataset ouput of forward maps mean and se. 
      requires respfile, covfile and testcov

    The models are estimated on the basis of data stored on disk in ascii or
    neuroimaging data formats (nifti or cifti). Ascii data should be in
    tab or space delimited format with the number of subjects in rows and the
    number of variables in columns. Neuroimaging data will be reshaped
    into the appropriate format

    Basic usage::

        estimate(covfile, respfile, [extra_arguments])

    where the variables are defined below. Note that either the cfolds
    parameter or (testcov, testresp) should be specified, but not both.

    :param respfile: response variables for the normative model
    :param covfile: covariates used to predict the response variable
    :param maskfile: mask used to apply to the data (nifti only)
    :param cvfolds: Number of cross-validation folds
    :param testcov: Test covariates
    :param testresp: Test responses
    :param alg: Algorithm for normative model
    :param configparam: Parameters controlling the estimation algorithm
    :param saveoutput: Save the output to disk? Otherwise returned as arrays
    :param outputsuffix: Text string to add to the output filenames
    :param inscale: Scaling approach for input covariates, could be 'None' (Default), 
                    'standardize', 'minmax', or 'robminmax'.
    :param outscale: Scaling approach for output responses, could be 'None' (Default), 
                    'standardize', 'minmax', or 'robminmax'.

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * yhat - predictive mean
              * ys2 - predictive variance
              * nm - normative model
              * Z - deviance scores
              * Rho - Pearson correlation between true and predicted responses
              * pRho - parametric p-value for this correlation
              * rmse - root mean squared error between true/predicted responses
              * smse - standardised mean squared error

    The outputsuffix may be useful to estimate multiple normative models in the
    same directory (e.g. for custom cross-validation schemes)
    """
    
    # parse keyword arguments 
    maskfile = kwargs.pop('maskfile',None)
    cvfolds = kwargs.pop('cvfolds', None)
    testcov = kwargs.pop('testcov', None)
    testresp = kwargs.pop('testresp',None)
    alg = kwargs.pop('alg','gpr')
    outputsuffix = kwargs.pop('outputsuffix','estimate')
    outputsuffix = "_" + outputsuffix.replace("_", "")  # Making sure there is only one 
                                                        # '_' is in the outputsuffix to 
                                                        # avoid file name parsing problem.
    inscaler = kwargs.pop('inscaler','None')
    outscaler = kwargs.pop('outscaler','None')
    warp = kwargs.get('warp', None)

    # convert from strings if necessary
    saveoutput = kwargs.pop('saveoutput','True')
    if type(saveoutput) is str:
        saveoutput = saveoutput=='True'
    savemodel = kwargs.pop('savemodel','False')
    if type(savemodel) is str:
        savemodel = savemodel=='True'
    
    if savemodel and not os.path.isdir('Models'):
        os.mkdir('Models')

    # which output metrics to compute
    metrics = ['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL','NLL', 'BIC']
    
    # load data
    print("Processing data in " + respfile)
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    Nmod = Y.shape[1]
    
    if (testcov is not None) and (cvfolds is None): # a separate test dataset
        
        run_cv = False
        cvfolds = 1
        Xte = fileio.load(testcov)
        if len(Xte.shape) == 1:
            Xte = Xte[:, np.newaxis]
        if testresp is not None:
            Yte, testmask = load_response_vars(testresp, maskfile)
            if len(Yte.shape) == 1:
                Yte = Yte[:, np.newaxis]
        else:
            sub_te = Xte.shape[0]
            Yte = np.zeros([sub_te, Nmod])
            
        # treat as a single train-test split
        testids = range(X.shape[0], X.shape[0]+Xte.shape[0])
        splits = CustomCV((range(0, X.shape[0]),), (testids,))

        Y = np.concatenate((Y, Yte), axis=0)
        X = np.concatenate((X, Xte), axis=0)
        
    else:
        run_cv = True
        # we are running under cross-validation
        splits = KFold(n_splits=cvfolds, shuffle=True)
        testids = range(0, X.shape[0])
        if alg=='hbr':
           trbefile = kwargs.get('trbefile', None) 
           if trbefile is not None:
                be = fileio.load(trbefile)
                if len(be.shape) == 1:
                    be = be[:, np.newaxis]
           else:
                print('No batch-effects file! Initilizing all as zeros!')
                be = np.zeros([X.shape[0],1])

    # find and remove bad variables from the response variables
    # note: the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    # run cross-validation loop
    Yhat = np.zeros_like(Y)
    S2 = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    nlZ = np.zeros((Nmod, cvfolds))
    
    scaler_resp = []
    scaler_cov = []
    mean_resp = [] # this is just for computing MSLL
    std_resp = [] # this is just for computing MSLL
    
    if warp is not None:
        Ywarp = np.zeros_like(Yhat)
        
        # for warping we need to compute metrics separately for each fold
        results_folds = dict()
        for m in metrics:
            results_folds[m]= np.zeros((Nmod, cvfolds))

    for idx in enumerate(splits.split(X)):

        fold = idx[0]
        tr = idx[1][0]
        ts = idx[1][1]

        # standardize responses and covariates, ignoring invalid entries
        iy_tr, jy_tr = np.ix_(tr, nz)
        iy_ts, jy_ts = np.ix_(ts, nz)
        mY = np.mean(Y[iy_tr, jy_tr], axis=0)
        sY = np.std(Y[iy_tr, jy_tr], axis=0)
        mean_resp.append(mY)
        std_resp.append(sY)
        
        if inscaler in ['standardize', 'minmax', 'robminmax']:
            X_scaler = scaler(inscaler)
            Xz_tr = X_scaler.fit_transform(X[tr, :])
            Xz_ts = X_scaler.transform(X[ts, :])
            scaler_cov.append(X_scaler)
        else:
            Xz_tr = X[tr, :]
            Xz_ts = X[ts, :]
            
        if outscaler in ['standardize', 'minmax', 'robminmax']:
            Y_scaler = scaler(outscaler)
            Yz_tr = Y_scaler.fit_transform(Y[iy_tr, jy_tr])
            scaler_resp.append(Y_scaler)
        else:
            Yz_tr = Y[iy_tr, jy_tr]
        
        if (run_cv==True and alg=='hbr'):
            fileio.save(be[tr,:], 'be_kfold_tr_tempfile.pkl')
            fileio.save(be[ts,:], 'be_kfold_ts_tempfile.pkl')
            kwargs['trbefile'] = 'be_kfold_tr_tempfile.pkl'
            kwargs['tsbefile'] = 'be_kfold_ts_tempfile.pkl'

        # estimate the models for all subjects
        for i in range(0, len(nz)):              
            print("Estimating model ", i+1, "of", len(nz))
            nm = norm_init(Xz_tr, Yz_tr[:, i], alg=alg, **kwargs)
                
            try:
                nm = nm.estimate(Xz_tr, Yz_tr[:, i], **kwargs)     
                yhat, s2 = nm.predict(Xz_ts, Xz_tr, Yz_tr[:, i], **kwargs)
                
                if savemodel:
                    nm.save('Models/NM_' + str(fold) + '_' + str(nz[i]) + 
                            outputsuffix + '.pkl' )
                
                if outscaler == 'standardize': 
                    Yhat[ts, nz[i]] = Y_scaler.inverse_transform(yhat, index=i)
                    S2[ts, nz[i]] = s2 * sY[i]**2
                elif outscaler in ['minmax', 'robminmax']:
                    Yhat[ts, nz[i]] = Y_scaler.inverse_transform(yhat, index=i)
                    S2[ts, nz[i]] = s2 * (Y_scaler.max[i] - Y_scaler.min[i])**2
                else:
                    Yhat[ts, nz[i]] = yhat
                    S2[ts, nz[i]] = s2
                    
                nlZ[nz[i], fold] = nm.neg_log_lik
                
                if (run_cv or testresp is not None):
                    if warp is not None:
                        # TODO: Warping for scaled data
                        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
                        Ywarp[ts, nz[i]] = nm.blr.warp.f(Y[ts, nz[i]], warp_param)
                        Ytest = Ywarp[ts, nz[i]]
                        
                        # Save warped mean of the training data (for MSLL)
                        yw = nm.blr.warp.f(Y[tr, nz[i]], warp_param)
                        
                        # create arrays for evaluation
                        Yhati = Yhat[ts, nz[i]]
                        Yhati = Yhati[:, np.newaxis]
                        S2i = S2[ts, nz[i]]
                        S2i = S2i[:, np.newaxis]

                        # evaluate and save results
                        mf = evaluate(Ytest[:, np.newaxis], Yhati, S2=S2i, 
                                      mY=np.mean(yw), sY=np.std(yw), 
                                      nlZ=nm.neg_log_lik, nm=nm, Xz_tr=Xz_tr, 
                                      alg=alg, metrics = metrics)
                        for k in metrics:
                            results_folds[k][nz[i]][fold] = mf[k]
                    else:
                        Ytest = Y[ts, nz[i]] 
                    
                    Z[ts, nz[i]] = (Ytest - Yhat[ts, nz[i]]) / \
                                    np.sqrt(S2[ts, nz[i]])       
                    
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("Model ", i+1, "of", len(nz),
                      "FAILED!..skipping and writing NaN to outputs")
                print("Exception:")
                print(e)
                print(exc_type, fname, exc_tb.tb_lineno)

                Yhat[ts, nz[i]] = float('nan')
                S2[ts, nz[i]] = float('nan')
                nlZ[nz[i], fold] = float('nan')
                if testcov is None:
                    Z[ts, nz[i]] = float('nan')
                else:
                    if testresp is not None:
                        Z[ts, nz[i]] = float('nan')


    if savemodel:
        print('Saving model meta-data...')
        v  = get_package_versions()
        with open('Models/meta_data.md', 'wb') as file:
            pickle.dump({'valid_voxels':nz, 'fold_num':cvfolds, 
                         'mean_resp':mean_resp, 'std_resp':std_resp, 
                         'scaler_cov':scaler_cov, 'scaler_resp':scaler_resp, 
                         'regressor':alg, 'inscaler':inscaler, 
                         'outscaler':outscaler, 'versions':v}, 
                        file, protocol=PICKLE_PROTOCOL)    

    # compute performance metrics
    if (run_cv or testresp is not None):
        print("Evaluating the model ...")
        if warp is None:
            results = evaluate(Y[testids, :], Yhat[testids, :], 
                               S2=S2[testids, :], mY=mean_resp[0], 
                               sY=std_resp[0], nlZ=nlZ, nm=nm, Xz_tr=Xz_tr, alg=alg,
                               metrics = metrics)
        else:
            # for warped data we just aggregate across folds
            results = dict()
            for m in ['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL']:
                results[m] = np.mean(results_folds[m], axis=1)
            results['NLL'] = results_folds['NLL']
            results['BIC'] = results_folds['BIC']            
        
    # Set writing options
    if saveoutput:
        if (run_cv or testresp is not None):
            save_results(respfile, Yhat[testids, :], S2[testids, :], maskvol, 
                         Z=Z[testids, :], results=results, 
                         outputsuffix=outputsuffix)
            
        else:
            save_results(respfile, Yhat[testids, :], S2[testids, :], maskvol,
                         outputsuffix=outputsuffix)
                
    else:
        if (run_cv or testresp is not None):
            output = (Yhat[testids, :], S2[testids, :], nm, Z[testids, :], 
                      results)
        else:
            output = (Yhat[testids, :], S2[testids, :], nm)
        
        return output


def fit(covfile, respfile, **kwargs):
    
    # parse keyword arguments 
    maskfile = kwargs.pop('maskfile',None)
    alg = kwargs.pop('alg','gpr')
    savemodel = kwargs.pop('savemodel','True')=='True'
    outputsuffix = kwargs.pop('outputsuffix','fit')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inscaler = kwargs.pop('inscaler','None')
    outscaler = kwargs.pop('outscaler','None')
    
    if savemodel and not os.path.isdir('Models'):
        os.mkdir('Models')

    # load data
    print("Processing data in " + respfile)
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    # find and remove bad variables from the response variables
    # note: the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]        
    
    scaler_resp = []
    scaler_cov = []
    mean_resp = [] # this is just for computing MSLL
    std_resp = []   # this is just for computing MSLL
    
    # standardize responses and covariates, ignoring invalid entries
    mY = np.mean(Y[:, nz], axis=0)
    sY = np.std(Y[:, nz], axis=0)
    mean_resp.append(mY)
    std_resp.append(sY)
    
    if inscaler in ['standardize', 'minmax', 'robminmax']:
        X_scaler = scaler(inscaler)
        Xz = X_scaler.fit_transform(X)
        scaler_cov.append(X_scaler)
    else:
        Xz = X
        
    if outscaler in ['standardize', 'minmax', 'robminmax']:
        Yz = np.zeros_like(Y)
        Y_scaler = scaler(outscaler)
        Yz[:, nz] = Y_scaler.fit_transform(Y[:, nz])
        scaler_resp.append(Y_scaler)
    else:
        Yz = Y

    # estimate the models for all subjects
    for i in range(0, len(nz)):  
        print("Estimating model ", i+1, "of", len(nz))
        nm = norm_init(Xz, Yz[:, nz[i]], alg=alg, **kwargs)
        nm = nm.estimate(Xz, Yz[:, nz[i]], **kwargs)     
            
        if savemodel:
            nm.save('Models/NM_' + str(0) + '_' + str(nz[i]) + outputsuffix + 
                    '.pkl' )

    if savemodel:
        print('Saving model meta-data...')
        v  = get_package_versions()
        with open('Models/meta_data.md', 'wb') as file:
            pickle.dump({'valid_voxels':nz,
                         'mean_resp':mean_resp, 'std_resp':std_resp, 
                         'scaler_cov':scaler_cov, 'scaler_resp':scaler_resp, 
                         'regressor':alg, 'inscaler':inscaler,
                         'outscaler':outscaler, 'versions':v}, 
                        file, protocol=PICKLE_PROTOCOL)
        
    return nm

    
def predict(covfile, respfile, maskfile=None, **kwargs):
    '''
    Make predictions on the basis of a pre-estimated normative model 
    If only the covariates are specified then only predicted mean and variance 
    will be returned. If the test responses are also specified then quantities
    That depend on those will also be returned (Z scores and error metrics)

    Basic usage::

        predict(covfile, [extra_arguments])

    where the variables are defined below.

    :param covfile: test covariates used to predict the response variable
    :param respfile: test response variables for the normative model
    :param maskfile: mask used to apply to the data (nifti only)
    :param model_path: Directory containing the normative model and metadata.
     When using parallel prediction, do not pass the model path. It will be 
     automatically decided.
    :param outputsuffix: Text string to add to the output filenames
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param fold: which cross-validation fold to use (default = 0)
    :param fold: list of model IDs to predict (if not specified all are computed)

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * Yhat - predictive mean
              * S2 - predictive variance
              * Z - Z scores
    '''
    
    
    model_path = kwargs.pop('model_path', 'Models')
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    outputsuffix = kwargs.pop('outputsuffix', 'predict')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    alg = kwargs.pop('alg')
    fold = kwargs.pop('fold',0)
    models = kwargs.pop('models', None)
        
    if respfile is not None and not os.path.exists(respfile):
        print("Response file does not exist. Only returning predictions")
        respfile = None
    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            inscaler = meta_data['inscaler']
            outscaler = meta_data['outscaler']
            mY = meta_data['mean_resp']
            sY = meta_data['std_resp']
            scaler_cov = meta_data['scaler_cov']
            scaler_resp = meta_data['scaler_resp']
            meta_data = True
        else:
            print("No meta-data file is found!")
            inscaler = 'None'
            outscaler = 'None'
            meta_data = False

    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1

    
    # load data
    print("Loading data ...")
    X = fileio.load(covfile)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    sample_num = X.shape[0]
    if models is not None:
        feature_num = len(models)
    else:
        feature_num = len(glob.glob(os.path.join(model_path, 'NM_'+ str(fold) +
                                                 '_*' + inputsuffix + '.pkl')))
        models = range(feature_num)

    Yhat = np.zeros([sample_num, feature_num])
    S2 = np.zeros([sample_num, feature_num])
    Z = np.zeros([sample_num, feature_num])
    
    if inscaler in ['standardize', 'minmax', 'robminmax']:
        Xz = scaler_cov[0].transform(X)
    else:
        Xz = X
    
    # estimate the models for all subjects
    for i, m in enumerate(models):
        print("Prediction by model ", i+1, "of", feature_num)      
        nm = norm_init(Xz)
        nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' + 
                                  str(m) + inputsuffix + '.pkl'))
        if (alg!='hbr' or nm.configs['transferred']==False):
            yhat, s2 = nm.predict(Xz, **kwargs)
        else:
            tsbefile = kwargs.get('tsbefile') 
            batch_effects_test = fileio.load(tsbefile)
            yhat, s2 = nm.predict_on_new_sites(Xz, batch_effects_test)
        
        if outscaler == 'standardize': 
            Yhat[:, i] = scaler_resp[0].inverse_transform(yhat, index=i)
            S2[:, i] = s2.squeeze() * sY[0][i]**2
        elif outscaler in ['minmax', 'robminmax']:
            Yhat[:, i] = scaler_resp[0].inverse_transform(yhat, index=i)
            S2[:, i] = s2 * (scaler_resp[0].max[i] - scaler_resp[0].min[i])**2
        else:
            Yhat[:, i] = yhat.squeeze()
            S2[:, i] = s2.squeeze()

    if respfile is None:
        save_results(None, Yhat, S2, None, outputsuffix=outputsuffix)
        
        return (Yhat, S2)
    
    else:
        Y, maskvol = load_response_vars(respfile, maskfile)
        if models is not None and len(Y.shape) > 1:
            Y = Y[:, models]
            if meta_data:
                mY = mY[models]
                sY = sY[models]
        
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        
        # warp the targets?
        if 'blr' in dir(nm):
            if nm.blr.warp is not None:
                warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
                Y = nm.blr.warp.f(Y, warp_param)
        
        Z = (Y - Yhat) / np.sqrt(S2)
        
        print("Evaluating the model ...")
        if meta_data:
            results = evaluate(Y, Yhat, S2=S2, mY=mY[0], sY=sY[0])
        else:    
            results = evaluate(Y, Yhat, S2=S2, 
                           metrics = ['Rho', 'RMSE', 'SMSE', 'EXPV'])
        
        print("Evaluations Writing outputs ...")
        save_results(respfile, Yhat, S2, maskvol, Z=Z, 
                     outputsuffix=outputsuffix, results=results)
        
        return (Yhat, S2, Z)

    
def transfer(covfile, respfile, testcov=None, testresp=None, maskfile=None, 
             **kwargs):
    '''
    Transfer learning on the basis of a pre-estimated normative model by using 
    the posterior distribution over the parameters as an informed prior for 
    new data. currently only supported for HBR.
    
    Basic usage::

        transfer(covfile, respfile [extra_arguments])

    where the variables are defined below.

    :param covfile: test covariates used to predict the response variable
    :param respfile: test response variables for the normative model
    :param maskfile: mask used to apply to the data (nifti only)
    :param testcov: Test covariates
    :param testresp: Test responses
    :param model_path: Directory containing the normative model and metadata
    :param trbefile: Training batch effects file
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * Yhat - predictive mean
              * S2 - predictive variance
              * Z - Z scores
    '''
    
    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Model transferring is only possible for HBR models.')
        return
    elif (not 'model_path' in list(kwargs.keys())) or \
        (not 'output_path' in list(kwargs.keys())) or \
        (not 'trbefile' in list(kwargs.keys())):
            print('InputError: Some mandatory arguments are missing.')
            return
    else:
        model_path = kwargs.pop('model_path')
        output_path = kwargs.pop('output_path')
        trbefile = kwargs.pop('trbefile')
        batch_effects_train = fileio.load(trbefile)
    
    outputsuffix = kwargs.pop('outputsuffix', 'transfer')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    tsbefile = kwargs.pop('tsbefile', None)
    
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1
    
    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            inscaler = meta_data['inscaler']
            outscaler = meta_data['outscaler']
            scaler_cov = meta_data['scaler_cov']
            scaler_resp = meta_data['scaler_resp']
            meta_data = True
        else:
            print("No meta-data file is found!")
            inscaler = 'None'
            outscaler = 'None'
            meta_data = False
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)    
       
    # load data
    print("Loading data ...")
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
        
    if inscaler in ['standardize', 'minmax', 'robminmax']:
        X = scaler_cov[0].transform(X)
    
    feature_num = Y.shape[1]
    mY = np.mean(Y, axis=0)
    sY = np.std(Y, axis=0)  
    
    if outscaler in ['standardize', 'minmax', 'robminmax']:
        Y = scaler_resp[0].transform(Y)
    
    if testcov is not None:
        # we have a separate test dataset
        Xte = fileio.load(testcov)
        if len(Xte.shape) == 1:
            Xte = Xte[:, np.newaxis]
        ts_sample_num = Xte.shape[0]
        if inscaler in ['standardize', 'minmax', 'robminmax']:
            Xte = scaler_cov[0].transform(Xte)
        
        if testresp is not None:
            Yte, testmask = load_response_vars(testresp, maskfile)
            if len(Yte.shape) == 1:
                Yte = Yte[:, np.newaxis]
        else:
            Yte = np.zeros([ts_sample_num, feature_num])
        
        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:
            batch_effects_test = np.zeros([Xte.shape[0],2])  
    else:
        ts_sample_num = 0     

    Yhat = np.zeros([ts_sample_num, feature_num])
    S2 = np.zeros([ts_sample_num, feature_num])
    Z = np.zeros([ts_sample_num, feature_num])
    
    # estimate the models for all subjects
    for i in range(feature_num):
              
        nm = norm_init(X)
        if batch_size is not None: # when using normative_parallel
            print("Transferring model ", job_id*batch_size+i)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + 
                                      str(job_id*batch_size+i) + inputsuffix + 
                                      '.pkl'))
        else:
            print("Transferring model ", i+1, "of", feature_num)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + str(i) + 
                                      inputsuffix + '.pkl'))
        
        nm = nm.estimate_on_new_sites(X, Y[:,i], batch_effects_train)
        if batch_size is not None: 
            nm.save(os.path.join(output_path, 'NM_0_' + 
                             str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm.save(os.path.join('Models', 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))
        else:
            nm.save(os.path.join(output_path, 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))
        
        if testcov is not None:
            yhat, s2 = nm.predict_on_new_sites(Xte, batch_effects_test)
            if outscaler == 'standardize': 
                Yhat[:, i] = scaler_resp[0].inverse_transform(yhat, index=i)
                S2[:, i] = s2.squeeze() * sY[0][i]**2
            elif outscaler in ['minmax', 'robminmax']:
                Yhat[:, i] = scaler_resp[0].inverse_transform(yhat, index=i)
                S2[:, i] = s2 * (scaler_resp[0].max[i] - scaler_resp[0].min[i])**2
            else:
                Yhat[:, i] = yhat.squeeze()
                S2[:, i] = s2.squeeze()
   
    if testresp is None:
        save_results(respfile, Yhat, S2, maskvol, outputsuffix=outputsuffix)
        return (Yhat, S2)
    else:
        Z = (Yte - Yhat) / np.sqrt(S2)
    
        print("Evaluating the model ...")
        results = evaluate(Yte, Yhat, S2=S2, mY=mY, sY=sY)
                
        save_results(respfile, Yhat, S2, maskvol, Z=Z, results=results,
                     outputsuffix=outputsuffix)
        
        return (Yhat, S2, Z)


def extend(covfile, respfile, maskfile=None, **kwargs):
    
    '''
    This function extends an existing HBR model with data from new sites/scanners.
    
    Basic usage::

        extend(covfile, respfile [extra_arguments])

    where the variables are defined below.

    :param covfile: covariates for new data
    :param respfile: response variables for new data
    :param maskfile: mask used to apply to the data (nifti only)
    :param model_path: Directory containing the normative model and metadata
    :param trbefile: file address to batch effects file for new data
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param output_path: the path for saving the  the extended model
    :param informative_prior: a flag to decide whether to use the initial model 
    prior or learn it from scrach (default is False).
    :param generation_factor: the number of samples generated for each combination
    of covariates and batch effects. Default is 10.
    

    All outputs are written to disk in the same format as the input.
    
    '''
    
    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Model extention is only possible for HBR models.')
        return
    elif (not 'model_path' in list(kwargs.keys())) or \
        (not 'output_path' in list(kwargs.keys())) or \
        (not 'trbefile' in list(kwargs.keys())):
            print('InputError: Some mandatory arguments are missing.')
            return
    else:
        model_path = kwargs.pop('model_path')
        output_path = kwargs.pop('output_path')
        trbefile = kwargs.pop('trbefile')
    
    outputsuffix = kwargs.pop('outputsuffix', 'extend')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    informative_prior = kwargs.pop('informative_prior', 'False') == 'True'
    generation_factor = int(kwargs.pop('generation_factor', '10'))
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1
     
    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            if (meta_data['inscaler'] != 'None' or 
                meta_data['outscaler'] != 'None'):
                print('Models extention on scaled data is not possible!')
                return
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
            
    # load data
    print("Loading data ...")
    X = fileio.load(covfile)    
    Y, maskvol = load_response_vars(respfile, maskfile)
    batch_effects_train = fileio.load(trbefile)
    
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    feature_num = Y.shape[1]
    
    # estimate the models for all subjects
    for i in range(feature_num):
              
        nm = norm_init(X)
        if batch_size is not None: # when using nirmative_parallel
            print("Extending model ", job_id*batch_size+i)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + 
                                      str(job_id*batch_size+i) + inputsuffix + 
                                      '.pkl'))
        else:
            print("Extending model ", i+1, "of", feature_num)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + str(i) + 
                                      inputsuffix +'.pkl'))
        
        nm = nm.extend(X, Y[:,i:i+1], batch_effects_train, 
                       samples=generation_factor, 
                       informative_prior=informative_prior)
        
        if batch_size is not None: 
            nm.save(os.path.join(output_path, 'NM_0_' + 
                             str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm.save(os.path.join('Models', 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))
        else:
            nm.save(os.path.join(output_path, 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))
            
            

def tune(covfile, respfile, maskfile=None, **kwargs):
    
    '''
    This function tunes an existing HBR model with real data.
    
    Basic usage::

        tune(covfile, respfile [extra_arguments])

    where the variables are defined below.

    :param covfile: covariates for new data
    :param respfile: response variables for new data
    :param maskfile: mask used to apply to the data (nifti only)
    :param model_path: Directory containing the normative model and metadata
    :param trbefile: file address to batch effects file for new data
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param output_path: the path for saving the  the extended model
    :param informative_prior: a flag to decide whether to use the initial model 
    prior or learn it from scrach (default is False).
    :param generation_factor: the number of samples generated for each combination
    of covariates and batch effects. Default is 10.
    

    All outputs are written to disk in the same format as the input.
    
    '''
    
    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Model extention is only possible for HBR models.')
        return
    elif (not 'model_path' in list(kwargs.keys())) or \
        (not 'output_path' in list(kwargs.keys())) or \
        (not 'trbefile' in list(kwargs.keys())):
            print('InputError: Some mandatory arguments are missing.')
            return
    else:
        model_path = kwargs.pop('model_path')
        output_path = kwargs.pop('output_path')
        trbefile = kwargs.pop('trbefile')
    
    outputsuffix = kwargs.pop('outputsuffix', 'tuned')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    informative_prior = kwargs.pop('informative_prior', 'False') == 'True'
    generation_factor = int(kwargs.pop('generation_factor', '10'))
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1
     
    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            if (meta_data['inscaler'] != 'None' or 
                meta_data['outscaler'] != 'None'):
                print('Models extention on scaled data is not possible!')
                return
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
            
    # load data
    print("Loading data ...")
    X = fileio.load(covfile)    
    Y, maskvol = load_response_vars(respfile, maskfile)
    batch_effects_train = fileio.load(trbefile)
    
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    feature_num = Y.shape[1]
    
    # estimate the models for all subjects
    for i in range(feature_num):
              
        nm = norm_init(X)
        if batch_size is not None: # when using nirmative_parallel
            print("Tuning model ", job_id*batch_size+i)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + 
                                      str(job_id*batch_size+i) + inputsuffix + 
                                      '.pkl'))
        else:
            print("Tuning model ", i+1, "of", feature_num)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + str(i) + 
                                      inputsuffix +'.pkl'))
        
        nm = nm.tune(X, Y[:,i:i+1], batch_effects_train, 
                       samples=generation_factor, 
                       informative_prior=informative_prior)
        
        if batch_size is not None: 
            nm.save(os.path.join(output_path, 'NM_0_' + 
                             str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm.save(os.path.join('Models', 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))
        else:
            nm.save(os.path.join(output_path, 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))


def merge(covfile=None, respfile=None, **kwargs):
    
    '''
    This function extends an existing HBR model with data from new sites/scanners.
    
    Basic usage::

        merge(model_path1, model_path2 [extra_arguments])

    where the variables are defined below.

    :param covfile: Not required. Always set to None.
    :param respfile: Not required. Always set to None.
    :param model_path1: Directory containing the normative model and metadata 
    of the first model.
    :param model_path2: Directory containing the normative model and metadata 
    of the second model.
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param output_path: the path for saving the  the extended model
    :param generation_factor: the number of samples generated for each combination
    of covariates and batch effects. Default is 10.
    

    All outputs are written to disk in the same format as the input.
    
    '''
    
    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Merging models is only possible for HBR models.')
        return
    elif (not 'model_path1' in list(kwargs.keys())) or \
        (not 'model_path2' in list(kwargs.keys())) or \
        (not 'output_path' in list(kwargs.keys())):
            print('InputError: Some mandatory arguments are missing.')
            return
    else:
        model_path1 = kwargs.pop('model_path1')
        model_path2 = kwargs.pop('model_path2')
        output_path = kwargs.pop('output_path')
    
    outputsuffix = kwargs.pop('outputsuffix', 'merge')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    generation_factor = int(kwargs.pop('generation_factor', '10'))
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1
     
    if (not os.path.isdir(model_path1)) or (not os.path.isdir(model_path2)):
        print('Models directory does not exist!')
        return
    else:
        if batch_size is None:
            with open(os.path.join(model_path1, 'meta_data.md'), 'rb') as file:
                meta_data1 = pickle.load(file)
            with open(os.path.join(model_path2, 'meta_data.md'), 'rb') as file:
                meta_data2 = pickle.load(file)
            if meta_data1['valid_voxels'].shape[0] != meta_data2['valid_voxels'].shape[0]:
                print('Two models are trained on different features!')
                return
            else:
                feature_num = meta_data1['valid_voxels'].shape[0]
        else:
            feature_num = batch_size
            
            
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    # mergeing the models
    for i in range(feature_num):
              
        nm1 = norm_init(np.random.rand(100,10))
        nm2 = norm_init(np.random.rand(100,10))
        if batch_size is not None: # when using nirmative_parallel
            print("Merging model ", job_id*batch_size+i)
            nm1 = nm1.load(os.path.join(model_path1, 'NM_0_' + 
                                      str(job_id*batch_size+i) + inputsuffix + 
                                      '.pkl'))
            nm2 = nm2.load(os.path.join(model_path2, 'NM_0_' + 
                                      str(job_id*batch_size+i) + inputsuffix + 
                                      '.pkl'))
        else:
            print("Merging model ", i+1, "of", feature_num)
            nm1 = nm1.load(os.path.join(model_path1, 'NM_0_' + str(i) + 
                                      inputsuffix +'.pkl'))
            nm2 = nm1.load(os.path.join(model_path2, 'NM_0_' + str(i) + 
                                      inputsuffix +'.pkl'))
        
        nm_merged = nm1.merge(nm2, samples=generation_factor)
        
        if batch_size is not None: 
            nm_merged.save(os.path.join(output_path, 'NM_0_' + 
                             str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm_merged.save(os.path.join('Models', 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))
        else:
            nm_merged.save(os.path.join(output_path, 'NM_0_' + 
                             str(i) + outputsuffix + '.pkl'))


def main(*args):
    """ Parse arguments and estimate model
    """

    np.seterr(invalid='ignore')

    rfile, mfile, cfile, cv, tcfile, trfile, func, alg, cfg, kw = get_args(args)
    
    # collect required arguments
    pos_args = ['cfile', 'rfile']
    
    # collect basic keyword arguments controlling model estimation
    kw_args = ['maskfile=mfile',
               'cvfolds=cv',
               'testcov=tcfile',
               'testresp=trfile',
               'alg=alg',
               'configparam=cfg']
    
    # add additional keyword arguments
    for k in kw:
        kw_args.append(k + '=' + "'" + kw[k] + "'")
    all_args = ', '.join(pos_args + kw_args)

    # Executing the target function
    exec(func + '(' + all_args + ')')

# For running from the command line:
if __name__ == "__main__":
    main(sys.argv[1:])
