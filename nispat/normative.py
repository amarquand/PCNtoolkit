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
    from nispat import fileio
    from nispat.normative_model.norm_utils import norm_init
    from nispat.utils import compute_pearsonr, CustomCV, explained_var, compute_MSLL
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
        #sys.path.append(os.path.join(path,'normative_model'))
    del path

    import fileio
    from utils import compute_pearsonr, CustomCV, explained_var, compute_MSLL
    from normative_model.norm_utils import norm_init


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
    parser.add_argument('-s', action='store_false', 
                        help="Flag to skip standardization.", dest="standardize")
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
    if args.testcov is None:
        testcov = None
        testresp = None
        cvfolds = int(args.cvfolds)
        print("Running under " + str(cvfolds) + " fold cross-validation.")
    else:
        print("Test covariates specified")
        testcov = args.testcov
        cvfolds = 1
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
            testcov, testresp, args.alg, \
            args.configparam, args.standardize, kw_args

def estimate(respfile, covfile, **kwargs):
    """ Estimate a normative model

    This will estimate a model in one of two settings according to the
    particular parameters specified (see below):

    * under k-fold cross-validation
        required settings 1) respfile 2) covfile 3) cvfolds>2
    * estimating a training dataset then applying to a second test dataset
        required sessting 1) respfile 2) covfile 3) testcov 4) testresp
    * estimating on a training dataset ouput of forward maps mean and se
        required sessting 1) respfile 2) covfile 3) testcov

    The models are estimated on the basis of data stored on disk in ascii or
    neuroimaging data formats (nifti or cifti). Ascii data should be in
    tab or space delimited format with the number of subjects in rows and the
    number of variables in columns. Neuroimaging data will be reshaped
    into the appropriate format

    Basic usage::

        estimate(respfile, covfile, [extra_arguments])

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
    saveoutput = kwargs.pop('saveoutput','True')=='True'
    savemodel = kwargs.pop('savemodel','False')=='True'
    outputsuffix = kwargs.pop('outputsuffix',None)
    standardize = kwargs.pop('standardize',True)
    trbefile = kwargs.pop('trbefile',None) # tarining batch effects file address
    tsbefile = kwargs.pop('tsbefile',None) # test batch effects file address
    nstsfile = kwargs.pop('nstsfile',None) # New site traininging samples file

    # load data
    print("Processing data in " + respfile)
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    Nmod = Y.shape[1]
    
    if trbefile is not None:
        batch_effects_train = fileio.load(trbefile)
    else:
        batch_effects_train = np.zeros([X.shape[0],2])
    kwargs['batch_effects_train'] = batch_effects_train
    
    if nstsfile is not None:
        newsite_training_idx = fileio.load(nstsfile)
    else:
        newsite_training_idx = None
    kwargs['newsite_training_idx'] = newsite_training_idx

    if testcov is not None:
        # we have a separate test dataset
        Xte = fileio.load(testcov)
        testids = range(X.shape[0], X.shape[0]+Xte.shape[0])
        if len(Xte.shape) == 1:
            Xte = Xte[:, np.newaxis]
        if testresp is not None:
            Yte, testmask = load_response_vars(testresp, maskfile)
            if len(Yte.shape) == 1:
                Yte = Yte[:, np.newaxis]
        else:
            sub_te = Xte.shape[0]
            Yte = np.zeros([sub_te, Nmod])
        
        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:
            batch_effects_test = np.zeros([Xte.shape[0],2])
        kwargs['batch_effects_test'] = batch_effects_test
    
        # Initialise normative model
        #nm = norm_init(X, alg=alg, configparam=configparam)
        
        # treat as a single train-test split
        splits = CustomCV((range(0, X.shape[0]),), (testids,))

        Y = np.concatenate((Y, Yte), axis=0)
        X = np.concatenate((X, Xte), axis=0)

        # force the number of cross-validation folds to 1
        if cvfolds is not None and cvfolds != 1:
            print("Ignoring cross-valdation specification (test data given)")
        cvfolds = 1
        
    else:
        # we are running under cross-validation
        splits = KFold(n_splits=cvfolds)
        testids = range(0, X.shape[0])
        # Initialise normative model
        #nm = norm_init(X, alg=alg, configparam=configparam)

    if savemodel and not os.path.isdir('Models'):
        os.mkdir('Models')
    
    # find and remove bad variables from the response variables
    # note: the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    # run cross-validation loop
    Yhat = np.zeros_like(Y)
    S2 = np.zeros_like(Y)
    #Hyp = np.zeros((Nmod, nm.n_params, cvfolds))

    Z = np.zeros_like(Y)
    nlZ = np.zeros((Nmod, cvfolds))
    
    mean_resp = []
    std_resp = []
    mean_cov = []
    std_cov = []

    for idx in enumerate(splits.split(X)):

        fold = idx[0]
        tr = idx[1][0]
        te = idx[1][1]

        # standardize responses and covariates, ignoring invalid entries
        iy, jy = np.ix_(tr, nz)
        mY = np.mean(Y[iy, jy], axis=0)
        sY = np.std(Y[iy, jy], axis=0)
        if standardize:
            Yz = np.zeros_like(Y)
            Yz[:, nz] = (Y[:, nz] - mY) / sY
            mX = np.mean(X[tr, :], axis=0)
            sX = np.std(X[tr, :],  axis=0)
            Xz = (X - mX) / sX
            mean_resp.append(mY)
            std_resp.append(sY)
            mean_cov.append(mX)
            std_cov.append(sX)
        else:
            Yz = Y
            Xz = X

        # estimate the models for all subjects
        for i in range(0, len(nz)):  # range(0, Nmod):
            print("Estimating model ", i+1, "of", len(nz))
            nm = norm_init(Xz[tr, :], Yz[tr, nz[i]], alg=alg, **kwargs)
            try: 
                nm = nm.estimate(Xz[tr, :], Yz[tr, nz[i]])     
                if (alg == 'hbr'):
                    if nm.configs['new_site'] == True:
                        nm = nm.estimate_on_new_sites(Xz[te, :], Y[te, nz[i]]) # The test/train division is done internally
                        yhat, s2 = nm.predict_on_new_sites(Xz[te, :])
                    else:    
                        yhat, s2 = nm.predict(Xz[te, :], Xz[tr, :], Yz[tr, nz[i]], **kwargs)
                else:
                    yhat, s2 = nm.predict(Xz[te, :], Xz[tr, :], Yz[tr, nz[i]], **kwargs)
                
                if savemodel:
                    nm.save('Models/NM_' + str(fold) + '_' + str(nz[i]) + '.pkl' )
                
                if standardize:
                    Yhat[te, nz[i]] = yhat * sY[i] + mY[i]
                    S2[te, nz[i]] = s2 * sY[i]**2
                else:
                    Yhat[te, nz[i]] = yhat
                    S2[te, nz[i]] = s2
                    
                nlZ[nz[i], fold] = nm.neg_log_lik
                if testcov is None:
                    Z[te, nz[i]] = (Y[te, nz[i]] - Yhat[te, nz[i]]) / \
                                   np.sqrt(S2[te, nz[i]])
                else:
                    if testresp is not None:
                        Z[te, nz[i]] = (Y[te, nz[i]] - Yhat[te, nz[i]]) / \
                                       np.sqrt(S2[te, nz[i]])

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("Model ", i+1, "of", len(nz),
                      "FAILED!..skipping and writing NaN to outputs")
                print("Exception:")
                print(e)
                print(exc_type, fname, exc_tb.tb_lineno)
                #Hyp[nz[i], :, fold] = float('nan')

                Yhat[te, nz[i]] = float('nan')
                S2[te, nz[i]] = float('nan')
                nlZ[nz[i], fold] = float('nan')
                if testcov is None:
                    Z[te, nz[i]] = float('nan')
                else:
                    if testresp is not None:
                        Z[te, nz[i]] = float('nan')
    if savemodel:
        print('Saving model meta-data...')
        with open('Models/meta_data.md', 'wb') as file:
            pickle.dump({'valid_voxels':nz, 'fold_num':cvfolds, 
                         'mean_resp':mean_resp, 'std_resp':std_resp, 
                         'mean_cov':mean_cov, 'std_cov':std_cov, 
                         'regressor':alg, 'standardize':standardize}, file)    

    # compute performance metrics
    if testcov is None:
        MSE = np.mean((Y[testids, :] - Yhat[testids, :])**2, axis=0)
        RMSE = np.sqrt(MSE)
        # for the remaining variables, we need to ignore zero variances
        SMSE = np.zeros_like(MSE)
        Rho = np.zeros(Nmod)
        pRho = np.ones(Nmod)
        EXPV = np.zeros(Nmod)
        MSLL = np.zeros(Nmod)
        iy, jy = np.ix_(testids, nz)  # ids for tested samples nonzero values
        SMSE[nz] = MSE[nz] / np.var(Y[iy, jy], axis=0)
        Rho[nz], pRho[nz] = compute_pearsonr(Y[iy, jy], Yhat[iy, jy])
        EXPV[nz] = explained_var(Y[iy, jy], Yhat[iy, jy])
        MSLL[nz] = compute_MSLL(Y[iy, jy], Yhat[iy, jy], S2[iy, jy], 
            mY.reshape(-1,1).T, (sY**2).reshape(-1,1).T)
    else:
        if testresp is not None:
            MSE = np.mean((Y[testids, :] - Yhat[testids, :])**2, axis=0)
            RMSE = np.sqrt(MSE)
            # for the remaining variables, we need to ignore zero variances
            SMSE = np.zeros_like(MSE)
            Rho = np.zeros(Nmod)
            pRho = np.ones(Nmod)
            EXPV = np.zeros(Nmod)
            MSLL = np.zeros(Nmod)
            iy, jy = np.ix_(testids, nz)  # ids tested samples nonzero values
            SMSE[nz] = MSE[nz] / np.var(Y[iy, jy], axis=0)
            Rho[nz], pRho[nz] = compute_pearsonr(Y[iy, jy], Yhat[iy, jy])
            EXPV[nz] = explained_var(Y[iy, jy], Yhat[iy, jy])
            MSLL[nz] = compute_MSLL(Y[iy, jy], Yhat[iy, jy], S2[iy, jy], 
                mY.reshape(-1,1).T, (sY**2).reshape(-1,1).T)
            
    # Set writing options
    if saveoutput:
        print("Writing output ...")
        if fileio.file_type(respfile) == 'cifti' or \
           fileio.file_type(respfile) == 'nifti':
            exfile = respfile
        else:
            exfile = None
        if outputsuffix is not None:
            ext = str(outputsuffix) + fileio.file_extension(respfile)
        else:
            ext = fileio.file_extension(respfile)

        # Write output
        if testcov is None:
            fileio.save(Yhat[testids, :], 'yhat' + ext,
                        example=exfile, mask=maskvol)
            fileio.save(S2[testids, :], 'ys2' + ext,
                        example=exfile, mask=maskvol)
            fileio.save(Z[testids, :], 'Z' + ext, example=exfile,
                        mask=maskvol)
            fileio.save(Rho, 'Rho' + ext, example=exfile, mask=maskvol)
            fileio.save(pRho, 'pRho' + ext, example=exfile, mask=maskvol)
            fileio.save(RMSE, 'rmse' + ext, example=exfile, mask=maskvol)
            fileio.save(SMSE, 'smse' + ext, example=exfile, mask=maskvol)
            fileio.save(EXPV, 'expv' + ext, example=exfile, mask=maskvol)
            fileio.save(MSLL, 'msll' + ext, example=exfile, mask=maskvol)
            #if cvfolds is None:
            #    fileio.save(Hyp[:,:,0], 'Hyp' + ext, example=exfile, mask=maskvol)
            #else:
            #    for idx in enumerate(splits.split(X)):
            #        fold = idx[0]
            #        fileio.save(Hyp[:, :, fold].T, 'Hyp_' + str(fold+1) +
            #                    ext, example=exfile, mask=maskvol)
        else:
            if testresp is None:
                fileio.save(Yhat[testids, :], 'yhat' + ext,
                            example=exfile, mask=maskvol)
                fileio.save(S2[testids, :], 'ys2' + ext,
                            example=exfile, mask=maskvol)
                #fileio.save(Hyp[:,:,0], 'Hyp' + ext,
                #            example=exfile, mask=maskvol)
            else:
                fileio.save(Yhat[testids, :], 'yhat' + ext,
                            example=exfile, mask=maskvol)
                fileio.save(S2[testids, :], 'ys2' + ext,
                            example=exfile, mask=maskvol)
                fileio.save(Z[testids, :], 'Z' + ext, example=exfile,
                            mask=maskvol)
                fileio.save(Rho, 'Rho' + ext, example=exfile, mask=maskvol)
                fileio.save(pRho, 'pRho' + ext, example=exfile, mask=maskvol)
                fileio.save(RMSE, 'rmse' + ext, example=exfile, mask=maskvol)
                fileio.save(SMSE, 'smse' + ext, example=exfile, mask=maskvol)
                fileio.save(EXPV, 'expv' + ext, example=exfile, mask=maskvol)
                fileio.save(MSLL, 'msll' + ext, example=exfile, mask=maskvol)
                #if cvfolds is None:
                #    fileio.save(Hyp[:,:,0].T, 'Hyp' + ext,
                #                example=exfile, mask=maskvol)
                #else:
                #    for idx in enumerate(splits.split(X)):
                #        fold = idx[0]
                #        fileio.save(Hyp[:, :, fold].T, 'Hyp_' + str(fold+1) +
                #                    ext, example=exfile, mask=maskvol)
    else:
        if testcov is None:
            output = (Yhat[testids, :], S2[testids, :], nm, Z[testids, :], Rho, 
                      pRho, RMSE, SMSE, EXPV, MSLL)
        else:
            if testresp is None:
                output = (Yhat[testids, :], S2[testids, :], nm)
            else:
                output = (Yhat[testids, :], S2[testids, :], nm, Z[testids, :], 
                          Rho, pRho, RMSE, SMSE, EXPV, MSLL)
        return output

def predict(model_path, covfile, respfile=None, output_path=None,  
            maskfile=None, **kwargs):
    
    if not os.path.isdir(model_path):
        print('Model directory does not exist!')
        return
    else:
        with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
            meta_data = pickle.load(file)
        cvfolds = meta_data['fold_num']
        standardize = meta_data['standardize']
        mY = meta_data['mean_resp']
        sY = meta_data['std_resp']
        mX = meta_data['mean_cov']
        sX = meta_data['std_cov']
    
    batch_effects_test = kwargs.pop('batch_effects_test',None)
    
    # load data
    print("Loading data ...")
    X = fileio.load(covfile)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    sample_num = X.shape[0]
    feature_num = len(glob.glob(os.path.join(model_path, 'NM_*.pkl')))

    # run cross-validation loop
    Yhat = np.zeros([sample_num, feature_num])
    S2 = np.zeros([sample_num, feature_num])
    Z = np.zeros([sample_num, feature_num])
    
    for fold in range(cvfolds):
        
        if standardize:
            Xz = (X - mX[fold]) / sX[fold]
        else:
            Xz = X
            
        # estimate the models for all subjects
        for i in range(feature_num):
            print("Prediction by model ", i+1, "of", feature_num)      
            nm = norm_init(Xz)
            nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' + str(i) + '.pkl'))
            nm.configs['batch_effects_test'] = batch_effects_test
            yhat, s2 = nm.predict(Xz)
            if standardize:
                Yhat[:, i] = yhat * sY[fold][i] + mY[fold][i]
                S2[:, i] = s2 * sY[fold][i]**2
            else:
                Yhat[:, i] = yhat
                S2[:, i] = s2
        
        if respfile is None:
            return (Yhat, S2)
        
        else:
            # Set writing options
            print("Evaluations and Writing outputs ...")      
            if fileio.file_type(respfile) == 'cifti' or \
               fileio.file_type(respfile) == 'nifti':
                exfile = respfile
            else:
                exfile = None
            ext = fileio.file_extension(respfile)    
            
            Y, maskvol = load_response_vars(respfile, maskfile)
            if len(Y.shape) == 1:
                Y = Y[:, np.newaxis]
            Nmod = Y.shape[1]
            
            Z = (Y - Yhat) / np.sqrt(S2)
            
            MSE = np.mean((Y - Yhat)**2, axis=0)
            RMSE = np.sqrt(MSE)
            # for the remaining variables, we need to ignore zero variances
            SMSE = np.zeros_like(MSE)
            Rho = np.zeros(Nmod)
            pRho = np.ones(Nmod)
            EXPV = np.zeros(Nmod)
            MSLL = np.zeros(Nmod)
            iy, jy = np.ix_(range(Y.shape[0]), range(Y.shape[1]))  # ids tested samples nonzero values
            SMSE = MSE / np.var(Y[iy, jy], axis=0)
            Rho, pRho = compute_pearsonr(Y[iy, jy], Yhat[iy, jy])
            EXPV = explained_var(Y[iy, jy], Yhat[iy, jy])
            #MSLL = compute_MSLL(Y[iy, jy], Yhat[iy, jy], S2[iy, jy], 
            #    mY.reshape(-1,1).T, (sY**2).reshape(-1,1).T)
            
            # Write output
            fileio.save(Yhat, os.path.join(output_path, 'yhat_' + str(fold) + ext),
                        example=exfile, mask=maskvol)
            fileio.save(S2, os.path.join(output_path,'ys2_' + str(fold) + ext),
                        example=exfile, mask=maskvol)
            fileio.save(Z, os.path.join(output_path,'Z_' + str(fold) + ext), 
                        example=exfile, mask=maskvol)
            fileio.save(Rho, os.path.join(output_path,'Rho_' + str(fold) + ext), 
                        example=exfile, mask=maskvol)
            fileio.save(pRho, os.path.join(output_path,'pRho_' + str(fold) + ext), 
                        example=exfile, mask=maskvol)
            fileio.save(RMSE, os.path.join(output_path,'rmse_' + str(fold) + ext),
                        example=exfile, mask=maskvol)
            fileio.save(SMSE, os.path.join(output_path,'smse_' + str(fold) + ext), 
                        example=exfile, mask=maskvol)
            fileio.save(EXPV, os.path.join(output_path,'expv_' + str(fold) + ext), 
                        example=exfile, mask=maskvol)
            fileio.save(MSLL, os.path.join(output_path,'msll_' + str(fold) + ext), 
                        example=exfile, mask=maskvol)
            
            return (Yhat, S2, Z)

def main(*args):
    """ Parse arguments and estimate model
    """

    np.seterr(invalid='ignore')

    rfile, mfile, cfile, cv, tcfile, trfile, alg, cfg, std, kw = get_args(args)
    
    # collect required arguments
    pos_args = ['rfile', 'cfile']
    
    # collect basic keyword arguments controlling model estimation
    kw_args = ['maskfile=mfile',
               'cvfolds=cv',
               'testcov=tcfile',
               'testresp=trfile',
               'alg=alg',
               'configparam=cfg',
               'standardize=std']
    
    # add additional keyword arguments
    for k in kw:
        kw_args.append(k + '=' + "'" + kw[k] + "'")
    all_args = ', '.join(pos_args + kw_args)

    # estimate normative model
    exec('estimate(' + all_args + ')')
    #estimate(rfile, cfile, maskfile=mfile, cvfolds=cv,testcov=tcfile,
    #         testresp=trfile, alg=alg,configparam=cfg, saveoutput=True, 
    #         standardize=std)

# For running from the command line:
if __name__ == "__main__":
    main(sys.argv[1:])
