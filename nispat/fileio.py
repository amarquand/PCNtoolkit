from __future__ import print_function

import os
import sys
import numpy as np
import nibabel as nib
import tempfile


def load_nifti(datafile):

    print('Loading nifti input ...')
    img = nib.load(datafile)
    dat = img.get_data()

    return dat


def vol2vec(dat, mask=None):

    if mask is None:
        print('No mask specified. Generating one automatically ...')
        if len(dat.shape) < 4:
            mask = dat[:, :, :] != 0
            dim = (np.prod(dat.shape[0:3]), 1)
            multivol = False
        else:
            mask = dat[:, :, :, 0] != 0
            dim = (np.prod(dat.shape[0:3]), dat.shape[3])
            multivol = True
    else:
        mask = load_nifti(mask)
        mask = mask != 0

    maskid = np.where(mask.ravel())[0]

    # mask the image
    dat = np.reshape(dat, dim)
    dat = dat[maskid, :]

    # convert to 1-d array if the file only contains one volume
    if not multivol:
        dat = dat.ravel()

    return dat


def load_cifti(filename, vol=False, mask=None):

    # parse the name
    dnam, fnam = os.path.split(filename)
    fpref, fext = os.path.splitext(fnam)
    outstem = os.path.join(tempfile.gettempdir(),
                           str(os.getpid()) + "-" + fpref)

    # extract surface data from the cifti file
    print("extracting surface data to ", outstem, '*.gii', sep="")
    giinamel = outstem + '-left.gii'
    giinamer = outstem + '-right.gii'
    os.system('wb_command -cifti-separate ' + filename +
              ' COLUMN -metric CORTEX_LEFT ' + giinamel)
    os.system('wb_command -cifti-separate ' + filename +
              ' COLUMN -metric CORTEX_RIGHT ' + giinamer)

    giil = nib.load(giinamel)
    giir = nib.load(giinamer)
    out = np.concatenate((giil.darrays[0].data, giir.darrays[0].data), axis=0)

    nimg = len(giil.darrays)
    nvert = len(giil.darrays[0].data)
    Gl = np.zeros((nvert, nimg))
    Gr = np.zeros((nvert, nimg))
    for i in range(0, nimg):
        Gl[:, i] = giil.darrays[i].data
        Gr[:, i] = giir.darrays[i].data

    if vol:
        niiname = outstem + '-vol.nii'
        print("extracting volume data to ", niiname, sep="")
        os.system('wb_command -cifti-separate ' + filename +
                  ' COLUMN -volume-all ' + niiname)
        vol = load_nifti(niiname)
        out = np.concatenate((out, vol2vec(vol, mask)), axis=0)

    return out


# def load_ascii(filename):


def load(filename, mask=None, ascii=False):

    if filename.endswith(('.dtseries.nii', '.dscalar.nii', '.dlabel.nii')):
        x = load_cifti(filename, vol=True)
    elif filename.endswith(('.nii.gz', '.nii', '.img', '.hdr')):
        v = load_nifti(filename)
        x = vol2vec(v, mask)
    # elif ascii or filename.endswith(('.txt', '.csv', '.tsv', '.asc')):
    #     x = load_ascii(filename)
    else:
        raise ValueError("I don't know what to do with " + filename)
    return x
