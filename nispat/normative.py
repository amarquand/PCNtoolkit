#!/home/mrstats/maamen/Software/python/bin/python2.7

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
from sklearn.model_selection import KFold

#from gp import GPR, covSqExp

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


def load_data(datafile, covfile, maskfile=None):
    """ load 4d nifti data """
    if datafile.endswith("nii.gz") or datafile.endswith("nii"):
        dat = fileio.load_nifti(datafile, vol=True)
        dim = dat.shape
        if len(dim) <= 3:
            dim = dim + (1,)
    else:
        raise ValueError("No routine to handle non-nifti data")

    mask = fileio.create_mask(dat, mask=maskfile)

    X = fileio.load(covfile)
    Y = fileio.vol2vec(dat, mask).T

    return X, Y, mask


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
    X, Y, mask = load_data(filename, covfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    Nsub, Nmod = Y.shape
    
    # set starting hyperparamters
    hyp0 = np.zeros(3)
    
    # run cross-validation loop
    kfcv = KFold(n_splits=Nfold)
    Yhat = np.zeros_like(Y)
    S2 = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    nlZ = np.zeros((Nmod, Nfold))
    Hyp = np.zeros((Nmod, len(hyp0), Nfold))
    fold = 1
    for tr, te in kfcv.split(X):
    
        # standardize responses and covariates
        mY = np.mean(Y[tr, :], axis=0)
        sY = np.std(Y[tr, :], axis=0)
        Yz = (Y - mY) / sY
        mX = np.mean(X[tr, :], axis=0)
        sX = np.std(X[tr, :],  axis=0)
        Xz = (X - mX) / sX  

        # estimate the models for all subjects
        for i in range(0, Nmod):
            print("Estimating model ", i+1, "of", Nmod)            
            gpr = GPR(hyp0, covSqExp, Xz[tr, :], Yz[tr, i])
            Hyp[i, :, fold] = gpr.estimate(hyp0, covSqExp, Xz[tr,:], Yz[tr, i])

            yhat, s2 = gpr.predict(Hyp[i, :, fold], Xz[tr, :], Yz[tr, i], Xz[te, :])
            
            Yhat[te, i] = yhat * sY[i] + mY[i]
            S2[te, i] = np.diag(s2) * sY[i]**2
            Z[te, i] = (Y[te, i] - Yhat[te, i]) / np.sqrt(S2[te, i])
            nlZ[i, fold] = gpr.nlZ

        rmse[i] = np.sqrt(np.mean((Y[:, i] - yhat[:, i]) ** 2))
        ev[i] = 100*(1 - (np.var(yhat[:, i] - Y[:, i]) / np.var(Y[:, i])))

        print("Variance explained =", ev[i], "% RMSE =", rmse[i])

    print("Mean (std) variance explained =", ev.mean(), "(", ev.std(), ")")
    print("Mean (std) RMSE =", rmse.mean(), "(", rmse.std(), ")")

    # Write output
    print("Writing output ...")
    np.savetxt("trendcoeff.txt", m, delimiter='\t', fmt='%5.8f')
    np.savetxt("negloglik.txt", nlZ, delimiter='\t', fmt='%5.8f')
    np.savetxt("hyp.txt", hyp, delimiter='\t', fmt='%5.8f')
    np.savetxt("explainedvar.txt", ev, delimiter='\t', fmt='%5.8f')
    np.savetxt("rmse.txt", rmse, delimiter='\t', fmt='%5.8f')
    fileio.save_nifti(yhat, 'yhat.nii.gz', filename, mask)
    fileio.save_nifti(ys2, 'ys2.nii.gz', filename, mask)

#wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
#maskfile = os.path.join(wdir, 'mask_3mm_left_striatum.nii.gz')
#filename = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
#covfile = os.path.join(wdir, 'covariates_basic_n50.txt')
#main(filename + '-m ' + maskfile + '-c ' + covfile)
main()

# For running from the command line:
#if __name__ == "__main__":
#    main(sys.argv[1:])