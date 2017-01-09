#!/home/mrstats/maamen/Software/python/bin/python2.7

# ------------------------------------------------------------------------------
#  Usage:
#  python normative.py -m [maskfile] -c [covariates] <infile>
#
#  Written by A. Marquand
# ------------------------------------------------------------------------------

from __future__ import print_function

import os
import sys
import numpy as np
import argparse

from gp import GPR, covSqExp

# Test whether this module is being invoked as a script or part of a package
if __name__ == "__main__":
    # running as a script
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if path not in sys.path:
        sys.path.append(path)
    del path
    from bayesreg import BLR
    import fileio
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
    Y = fileio.vol2vec(dat, mask)

    return X, Y, mask


def main(*args):
    np.seterr(invalid='ignore')

    # parse arguments
    parser = argparse.ArgumentParser(description="Trend surface model")
    parser.add_argument("filename")
    parser.add_argument("-m", help="mask file", dest="maskfile", default=None)
    parser.add_argument("-c", help="covariates file", dest="covfile",
                        default=None)
    args = parser.parse_args()
    wdir = os.path.realpath(os.path.curdir)
    filename = os.path.join(wdir, args.filename)
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir, args.maskfile)
    if args.covfile is None:
        raise(ValueError, "No covariates specified")

    # load data
    print("Processing data in", filename)
    Y, X, mask = load_data(filename, covfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    N = Y.shape[1]

    # standardize responses and covariates
    mY = np.mean(Y, axis=0)
    sY = np.std(Y, axis=0)
    Yz = (Y - mY) / sY
    mX = np.mean(X, axis=0)
    sX = np.std(X, axis=0)
    Xz = (X - mX) / sX

    # set starting hyperparamters
    hyp0 = np.zeros(2)
    
    G = GPR(hyp0, covSqExp, Xz[ids,:], y[ids])
    hyp = G.estimate(hyp0,covSqExp, Phi[yid,:], y[yid])
    yhat, s2 = G.predict(hyp0,Phi[yid,:],y[yid],Phi)

    # estimate the models for all subjects
    yhat = np.zeros_like(Yz)
    ys2 = np.zeros_like(Yz)
    nlZ = np.zeros(N)
    hyp = np.zeros((N, len(hyp0)))
    rmse = np.zeros(N)
    ev = np.zeros(N)
    m = np.zeros((N, Phi.shape[1]))
    for i in range(0, N):
        print("Estimating model ", i+1, "of", N)
        breg = BLR()
        hyp[i, :] = breg.estimate(hyp0, Phi, Yz[:, i])
        print(hyp)
        print(breg.nlZ)
        m[i, :] = breg.m
        nlZ[i] = breg.nlZ

        # compute predictions and errors
        yhat[:, i], ys2[:, i] = breg.predict(hyp[i, :], Phi, Yz[:, i], Phi)
        yhat[:, i] = yhat[:, i]*sY[i] + mY[i]
        plt.plot(yhat)
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

wdir = '/home/mrstats/andmar/py.sandbox/normative_nimg'
maskfile = os.path.join(wdir, 'mask_3mm.nii.gz')
datafile = os.path.join(wdir, 'shoot_data_3mm_n50.nii.gz')
covfile = os.path.join(wdir, 'covariates_n50.txt')
main(datafile + '-m ' + maskfile + '-c ' + covfile)

# For running from the command line:
#if __name__ == "__main__":
#    main(sys.argv[1:])