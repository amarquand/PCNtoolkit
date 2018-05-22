#!/Users/andre/sfw/anaconda3/bin/python

# ------------------------------------------------------------------------------
#  Usage:
#  python trendsurf.py -m [maskfile] -b [basis] -c [covariates] <infile>
#
#  Written by A. Marquand
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import nibabel as nib
import argparse

try:  # Run as a package if installed
    from nispat import fileio
    from nispat.bayesreg import BLR
except ImportError:
    pass
    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    import fileio
    from bayesreg import BLR



def load_data(datafile, maskfile=None):
    """ load 4d nifti data """
    if datafile.endswith("nii.gz") or datafile.endswith("nii"):
        # we load the data this way rather than fileio.load() because we need
        # access to the volumetric representation (to know the # coordinates)
        dat = fileio.load_nifti(datafile, vol=True)
        dim = dat.shape
        if len(dim) <= 3:
            dim = dim + (1,)
    else:
        raise ValueError("No routine to handle non-nifti data")

    mask = fileio.create_mask(dat, mask=maskfile)

    dat = fileio.vol2vec(dat, mask)
    maskid = np.where(mask.ravel())[0]

    # generate voxel coordinates
    i, j, k = np.meshgrid(np.linspace(0, dim[0]-1, dim[0]),
                          np.linspace(0, dim[1]-1, dim[1]),
                          np.linspace(0, dim[2]-1, dim[2]), indexing='ij')

    # voxel-to-world mapping
    img = nib.load(datafile)
    world = np.vstack((i.ravel(), j.ravel(), k.ravel(),
                       np.ones(np.prod(i.shape), float))).T
    world = np.dot(world, img.affine.T)[maskid, 0:3]

    return dat, world, mask


def create_basis(X, basis, mask):
    """ Create a (polynomial) basis set """

    # check whether we are using a polynomial basis set
    if type(basis) is int or (type(basis) is str and len(basis) == 1):
        dimpoly = int(basis)
        dimx = X.shape[1]
        print('Generating polynomial basis set of degree', dimpoly, '...')
        Phi = np.zeros((X.shape[0], X.shape[1]*dimpoly))
        colid = np.arange(0, dimx)
        for d in range(1, dimpoly+1):
            Phi[:, colid] = X ** d
            colid += dimx
    else:  # custom basis set
        if type(basis) is str:
            print('Loading custom basis set from', basis)

            # Phi_vol = fileio.load_data(basis)
            # we load the data this way instead so we can apply the same mask
            Phi_vol = fileio.load_nifti(basis, vol=True)
            Phi = fileio.vol2vec(Phi_vol, mask)
            print('Basis set consists of', Phi.shape[1], 'basis functions.')
            # maskid = np.where(mask.ravel())[0]
        else:
            raise ValueError("I don't know what to do with basis:", basis)

    return Phi


def write_nii(data, filename, examplenii, mask):
    """ Write output to nifti """

    # load example image
    ex_img = nib.load(examplenii)
    dim = ex_img.shape[0:3]
    nvol = int(data.shape[1])

    # write data
    array_data = np.zeros((np.prod(dim), nvol))
    array_data[mask.flatten(), :] = data
    array_data = np.reshape(array_data, dim+(nvol,))
    array_img = nib.Nifti1Image(array_data,
                                ex_img.get_affine(),
                                ex_img.get_header())
    nib.save(array_img, filename)


def get_args(*args):
    # parse arguments
    parser = argparse.ArgumentParser(description="Trend surface model")
    parser.add_argument("filename")
    parser.add_argument("-b", help="basis set", dest="basis", default=3)
    parser.add_argument("-m", help="mask file", dest="maskfile", default=None)
    parser.add_argument("-c", help="covariates file", dest="covfile",
                        default=None)
    parser.add_argument("-a", help="use ARD", action='store_true')
    parser.add_argument("-o", help="output all measures", action='store_true')
    args = parser.parse_args()
    wdir = os.path.realpath(os.path.curdir)
    filename = os.path.join(wdir, args.filename)
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir, args.maskfile)
    basis = args.basis
    if args.covfile is not None:
        raise(NotImplementedError, "Covariates not implemented yet.")

    return filename, maskfile, basis, args.a, args.o


def estimate(filename, maskfile, basis, ard=False, outputall=False,
             saveoutput=True):
    """ Estimate a trend surface model

    This will estimate a trend surface model, independently for each subject.
    This is currently fit using a polynomial model of a specified degree.
    The models are estimated on the basis of data stored on disk in ascii or
    neuroimaging data formats (currently nifti only). Ascii data should be in
    tab or space delimited format with the number of voxels in rows and the
    number of subjects in columns. Neuroimaging data will be reshaped
    into the appropriate format

    Basic usage::

        estimate(filename, maskfile, basis)

    where the variables are defined below. Note that either the cfolds
    parameter or (testcov, testresp) should be specified, but not both.

    :param filename: 4-d nifti file containing the images to be estimated
    :param maskfile: nifti mask used to apply to the data
    :param basis: model order for the interpolating polynomial

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * yhat - predictive mean
              * ys2 - predictive variance
              * trendcoeff - coefficients from the trend surface model
              * negloglik - Negative log marginal likelihood
              * hyp - hyperparameters
              * explainedvar - explained variance
              * rmse - standardised mean squared error
    """

    # load data
    print("Processing data in", filename)
    Y, X, mask = load_data(filename, maskfile)
    Y = np.round(10000*Y)/10000  # truncate precision to avoid numerical probs
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

    # create basis set and set starting hyperparamters
    Phi = create_basis(Xz, basis, mask)
    if ard is True:
        hyp0 = np.zeros(Phi.shape[1]+1)
    else:
        hyp0 = np.zeros(2)

    # estimate the models for all subjects
    if ard:
        print('ARD is enabled')
    yhat = np.zeros_like(Yz)
    ys2 = np.zeros_like(Yz)
    nlZ = np.zeros(N)
    hyp = np.zeros((N, len(hyp0)))
    rmse = np.zeros(N)
    ev = np.zeros(N)
    m = np.zeros((N, Phi.shape[1]))
    bs2 = np.zeros((N, Phi.shape[1]))
    for i in range(0, N):
        print("Estimating model ", i+1, "of", N)
        breg = BLR()
        hyp[i, :] = breg.estimate(hyp0, Phi, Yz[:, i])
        m[i, :] = breg.m
        nlZ[i] = breg.nlZ

        # compute extra measures (e.g. marginal variances)?
        if outputall:
            bs2[i] = np.sqrt(np.diag(np.linalg.inv(breg.A)))

        # compute predictions and errors
        yhat[:, i], ys2[:, i] = breg.predict(hyp[i, :], Phi, Yz[:, i], Phi)
        yhat[:, i] = yhat[:, i]*sY[i] + mY[i]
        rmse[i] = np.sqrt(np.mean((Y[:, i] - yhat[:, i]) ** 2))
        ev[i] = 100*(1 - (np.var(Y[:, i] - yhat[:, i]) / np.var(Y[:, i])))

        print("Variance explained =", ev[i], "% RMSE =", rmse[i])

    print("Mean (std) variance explained =", ev.mean(), "(", ev.std(), ")")
    print("Mean (std) RMSE =", rmse.mean(), "(", rmse.std(), ")")

    # Write output
    if saveoutput:
        print("Writing output ...")
        np.savetxt("trendcoeff.txt", m, delimiter='\t', fmt='%5.8f')
        np.savetxt("negloglik.txt", nlZ, delimiter='\t', fmt='%5.8f')
        np.savetxt("hyp.txt", hyp, delimiter='\t', fmt='%5.8f')
        np.savetxt("explainedvar.txt", ev, delimiter='\t', fmt='%5.8f')
        np.savetxt("rmse.txt", rmse, delimiter='\t', fmt='%5.8f')
        fileio.save_nifti(yhat, 'yhat.nii.gz', filename, mask)
        fileio.save_nifti(ys2, 'ys2.nii.gz', filename, mask)

        if outputall:
            np.savetxt("trendcoeffvar.txt", bs2, delimiter='\t', fmt='%5.8f')
    else:
        out = [yhat, ys2, nlZ, hyp, rmse, ev, m]
        if outputall:
            out.append(bs2)
        return out

def main(*args):
    np.seterr(invalid='ignore')

    filename, maskfile, basis, ard, outputall = get_args(args)
    estimate(filename, maskfile, basis, ard, outputall)

# For running from the command line:
if __name__ == "__main__":
    main(sys.argv[1:])
