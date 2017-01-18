#!/Users/andre/sfw/anaconda3/bin/python

# ------------------------------------------------------------------------------
#  Usage:
#  python normative.py -m [maskfile] -k [folds] -c <covariates> <infile>
#
#  Written by A. Marquand
# ------------------------------------------------------------------------------

from __future__ import print_function

import os
import sys
import numpy as np
import argparse

from scipy import stats
from sklearn.model_selection import KFold

# Test whether this module is being invoked as a script or part of a package
if __name__ == "__main__":
    # running as a script
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if path not in sys.path:
        sys.path.append(path)
    del path
    import fileio
    from gp import GPR, covSqExp
    # from fileio import load_nifti, save_nifti, create_mask


def load_data(datafile, covfile, maskfile=None, vol=True):
    """ load dataset (covariates + responses) """

    # load covariates
    X = fileio.load(covfile)

    # load responses
    if fileio.file_type(datafile) == 'nifti':
        dat = fileio.load_nifti(datafile, vol=vol)
        volmask = fileio.create_mask(dat, mask=maskfile)
        Y = fileio.vol2vec(dat, volmask).T
    else:
        Y = fileio.load(datafile)
        volmask = None
        if fileio.file_type(datafile) == 'cifti':
            Y = Y.T

    return X, Y, volmask


def compute_pearsonr(A, B):
    """ computes correlation manually to save memory """

    N = A.shape[1]

    # first mean centre
    Am = A - np.mean(A, axis=0)
    Bm = B - np.mean(B, axis=0)
    # then normalize
    An = Am / np.sqrt(np.sum(Am**2, axis=0))
    Bn = Bm / np.sqrt(np.sum(Bm**2, axis=0))
    del(Am, Bm)

    Rho = np.sum(An * Bn, axis=0)
    del(An, Bn)

    # Fisher r-to-z
    Zr = (np.arctanh(Rho) - np.arctanh(0)) * np.sqrt(N - 3)
    N = stats.norm()
    pRho = 1-N.cdf(Zr)

    return Rho, pRho


def main(*args):
    np.seterr(invalid='ignore')

    # parse arguments
    parser = argparse.ArgumentParser(description="Trend surface model")
    parser.add_argument("filename")
    parser.add_argument("-m", help="mask file", dest="maskfile", default=None)
    parser.add_argument("-c", help="covariates file", dest="covfile",
                        default=None)
    parser.add_argument("-k", help="cross-validation folds", dest="Nfold",
                        default=None)
    args = parser.parse_args()
    wdir = os.path.realpath(os.path.curdir)
    filename = os.path.join(wdir, args.filename)
    Nfold = int(args.Nfold)
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir, args.maskfile)
    if args.covfile is None:
        raise(ValueError, "No covariates specified")
    else:
        covfile = args.covfile

    # load data
    print("Processing data in", filename)
    X, Y, maskvol = load_data(filename, covfile, maskfile=maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    Nsub, Nmod = Y.shape

    # find and remove bad variables from the response variables
    # note that the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    # set basic parameters
    hyp0 = np.zeros(3)
    kfcv = KFold(n_splits=Nfold)
    fold = 0

    # run cross-validation loop
    Yhat = np.zeros_like(Y)
    S2 = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    nlZ = np.zeros((Nmod, Nfold))
    Hyp = np.zeros((Nmod, len(hyp0), Nfold))
    for tr, te in kfcv.split(X):

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
            gpr = GPR(hyp0, covSqExp, Xz[tr, :], Yz[tr, nz[i]])
            Hyp[nz[i], :, fold] = gpr.estimate(hyp0, covSqExp, Xz[tr, :],
                                               Yz[tr, nz[i]])

            yhat, s2 = gpr.predict(Hyp[nz[i], :, fold], Xz[tr, :],
                                   Yz[tr, nz[i]], Xz[te, :])

            Yhat[te, nz[i]] = yhat * sY[i] + mY[i]
            S2[te, nz[i]] = np.diag(s2) * sY[i]**2
            Z[te, nz[i]] = (Y[te, nz[i]] - Yhat[te, nz[i]]) / \
                           np.sqrt(S2[te, nz[i]])
            nlZ[nz[i], fold] = gpr.nlZ

        fold += 1

    # compute performance metrics
    MSE = np.mean((Y - Yhat)**2, axis=0)
    RMSE = np.sqrt(MSE)
    SMSE = np.zeros_like(MSE)
    SMSE[nz] = MSE[nz] / np.var(Y[:, nz], axis=0)
    Rho = np.zeros(Nmod)
    pRho = np.ones(Nmod)
    Rho[nz], pRho[nz] = compute_pearsonr(Y[:, nz], Yhat[:, nz])

    # Write output
    print("Writing output ...")
    if fileio.file_type(filename) == 'cifti' or \
       fileio.file_type(filename) == 'nifti':
        exfile = filename
    else:
        exfile = None
    ext = fileio.file_extension(filename)

    fileio.save(Yhat.T, 'yhat' + ext, example=exfile, mask=maskvol)
    fileio.save(S2.T, 'ys2' + ext, example=exfile, mask=maskvol)
    fileio.save(Z.T, 'Z' + ext, example=exfile, mask=maskvol)
    fileio.save(Rho, 'Rho' + ext, example=exfile, mask=maskvol)
    fileio.save(pRho, 'pRho' + ext, example=exfile, mask=maskvol)
    fileio.save(RMSE, 'rmse' + ext, example=exfile, mask=maskvol)
    fileio.save(SMSE, 'smse' + ext, example=exfile, mask=maskvol)

#wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
##wdir = '/Users/andre/data/normative_nimg'
#maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
#filename = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
#covfile = os.path.join(wdir, 'covariates_basic_n50.txt')
#Nfold = 2

#wdir = '/home/mrstats/andmar/py.sandbox/normative_hcp'
#filename = os.path.join(wdir, 'tfmri_gambling_cope2.dtseries.nii')
#covfile = os.path.join(wdir, 'ddscores.txt')
#Nfold = 2

#wdir = '/home/mrstats/andmar/data/enigma_mdd'
#maskfile = None
#filename = os.path.join(wdir, 'Enigma_1st_100sub_resp.txt')
#covfile = os.path.join(wdir, 'Enigma_1st_100sub_cov.txt')
#Nfold = 2

# For running from the command line:
if __name__ == "__main__":
    main(sys.argv[1:])