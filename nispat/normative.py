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

from sklearn.model_selection import KFold
if __name__ == "__main__":
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if path not in sys.path:
        sys.path.append(path)
    del path

    import fileio
    from gp import GPR, CovSum
    from utils import compute_pearsonr, CustomCV
else:  # Run as a package (assumes the package is installed)
    from nispat import fileio
    from nispat.gp import GPR, CovSum
    from nispat.utils import compute_pearsonr, CustomCV


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
    parser = argparse.ArgumentParser(description="Trend surface model")
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
    args = parser.parse_args()
    wdir = os.path.realpath(os.path.curdir)
    respfile = os.path.join(wdir, args.responses)
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir, args.maskfile)
    if args.covfile is None:
        raise(ValueError, "No covariates specified")
    else:
        covfile = args.covfile

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
            raise(ValueError, "No test response variables specified")
        else:
            testresp = args.testresp
        if args.cvfolds is not None:
            print("Ignoring cross-valdation specification (test data given)")

    return respfile, maskfile, covfile, cvfolds, testcov, testresp


def estimate(respfile, covfile, maskfile=None, cvfolds=None,
             testcov=None, testresp=None, outputsuffix=None):

    # load data
    print("Processing data in " + respfile)
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    Nmod = Y.shape[1]

    if testcov is not None:
        # we have a separate test dataset
        Xte = fileio.load(testcov)
        Yte, testmask = load_response_vars(testresp, maskfile)
        testids = range(X.shape[0], X.shape[0]+Xte.shape[0])

        # treat as a single train-test split
        splits = CustomCV((range(0, X.shape[0]),), (testids,))

        Y = np.concatenate((Y, Yte), axis=0)
        X = np.concatenate((X, Xte), axis=0)
    else:
        # we are running under cross-validation
        splits = KFold(n_splits=cvfolds)
        testids = range(0, X.shape[0])

    # find and remove bad variables from the response variables
    # note: the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    # starting hyperparameters. Could also do random restarts here
    covfunc = CovSum(X, ('CovLin', 'CovSqExpARD'))
    hyp0 = np.zeros(covfunc.get_n_params() + 1)

    # run cross-validation loop
    Yhat = np.zeros_like(Y)
    S2 = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    nlZ = np.zeros((Nmod, cvfolds))
    Hyp = np.zeros((Nmod, len(hyp0), cvfolds))
    for idx in enumerate(splits.split(X)):
        fold = idx[0]
        tr = idx[1][0]
        te = idx[1][1]

        # standardize responses and covariates, ignoring invalid entries
        iy, jy = np.ix_(tr, nz)
        mY = np.mean(Y[iy, jy], axis=0)
        sY = np.std(Y[iy, jy], axis=0)
        Yz = np.zeros_like(Y)
        Yz[:, nz] = (Y[:, nz] - mY) / sY
        mX = np.mean(X[tr, :], axis=0)
        sX = np.std(X[tr, :],  axis=0)
        Xz = (X - mX) / sX

        # estimate the models for all subjects
        for i in range(0, len(nz)):  # range(0, Nmod):
            print("Estimating model ", i+1, "of", len(nz))
            gpr = GPR(hyp0, covfunc, Xz[tr, :], Yz[tr, nz[i]])
            Hyp[nz[i], :, fold] = gpr.estimate(hyp0, covfunc, Xz[tr, :],
                                               Yz[tr, nz[i]])

            yhat, s2 = gpr.predict(Hyp[nz[i], :, fold], Xz[tr, :],
                                   Yz[tr, nz[i]], Xz[te, :])

            Yhat[te, nz[i]] = yhat * sY[i] + mY[i]
            S2[te, nz[i]] = np.diag(s2) * sY[i]**2
            Z[te, nz[i]] = (Y[te, nz[i]] - Yhat[te, nz[i]]) / \
                           np.sqrt(S2[te, nz[i]])
            nlZ[nz[i], fold] = gpr.nlZ

    # compute performance metrics
    MSE = np.mean((Y[testids, :] - Yhat[testids, :])**2, axis=0)
    RMSE = np.sqrt(MSE)
    # for the remaining variables, we need to ignore zero variances
    SMSE = np.zeros_like(MSE)
    Rho = np.zeros(Nmod)
    pRho = np.ones(Nmod)
    iy, jy = np.ix_(testids, nz)  # ids for tested samples with nonzero values
    SMSE[nz] = MSE[nz] / np.var(Y[iy, jy], axis=0)
    Rho[nz], pRho[nz] = compute_pearsonr(Y[iy, jy], Yhat[iy, jy])

    # Set writing options
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
    fileio.save(Yhat[testids, :].T, 'yhat' + ext, example=exfile, mask=maskvol)
    fileio.save(S2[testids, :].T, 'ys2' + ext, example=exfile, mask=maskvol)
    fileio.save(Z[testids, :].T, 'Z' + ext, example=exfile, mask=maskvol)
    fileio.save(Rho, 'Rho' + ext, example=exfile, mask=maskvol)
    fileio.save(pRho, 'pRho' + ext, example=exfile, mask=maskvol)
    fileio.save(RMSE, 'rmse' + ext, example=exfile, mask=maskvol)
    fileio.save(SMSE, 'smse' + ext, example=exfile, mask=maskvol)


def main(*args):

    np.seterr(invalid='ignore')

    respfile, maskfile, covfile, cvfolds, testcov, testresp = get_args(args)
    estimate(respfile, covfile, maskfile, cvfolds, testcov, testresp)

# For running from the command line:
if __name__ == "__main__":
    main(sys.argv[1:])
