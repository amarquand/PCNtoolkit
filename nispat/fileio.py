from __future__ import print_function

import os
import numpy as np
import nibabel as nib
import tempfile

# ------------------------
# general utility routines
# ------------------------


def create_mask(data_array, mask=None):
    # create a (volumetric) mask either from an input nifti or the nifti itself

    if mask is not None:
        print('Loading ROI mask ...')
        maskvol = load_nifti(mask, vol=True)
        maskvol = maskvol != 0
    else:
        if len(data_array.shape) < 4:
            dim = data_array.shape[0:3] + (1,)
        else:
            dim = data_array.shape[0:3] + (data_array.shape[3],)

        print('Generating mask automatically ...')
        if dim[3] == 1:
            maskvol = data_array[:, :, :] != 0
        else:
            maskvol = data_array[:, :, :, 0] != 0

    return maskvol


def vol2vec(dat, mask=None):
    # vectorise a 3d image

    if len(dat.shape) < 4:
        dim = dat.shape[0:3] + (1,)
    else:
        dim = dat.shape[0:3] + (dat.shape[3],)

    if mask is None:
        mask = create_mask(dat)

    # mask the image
    maskid = np.where(mask.ravel())[0]
    dat = np.reshape(dat, (np.prod(dim[0:3]), dim[3]))
    dat = dat[maskid, :]

    # convert to 1-d array if the file only contains one volume
    if dim[3] == 1:
        dat = dat.ravel()

    return dat


def file_type(filename):
    # routine to determine filetype

    if filename.endswith(('.dtseries.nii', '.dscalar.nii', '.dlabel.nii')):
        ftype = 'cifti'
    elif filename.endswith(('.nii.gz', '.nii', '.img', '.hdr')):
        ftype = 'nifti'
    elif filename.endswith(('.txt', '.csv', '.tsv', '.asc')):
        ftype = 'text'
    else:
        raise ValueError("I don't know what to do with " + filename)

    return ftype

# --------------
# nifti routines
# --------------


def load_nifti(datafile, mask=None, vol=False):

    print('Loading nifti: ' + datafile + ' ...')
    img = nib.load(datafile)
    dat = img.get_data()

    if not vol:
        dat = vol2vec(dat, mask)

    return dat


def save_nifti(data, filename, examplenii, mask):
    """ Write output to nifti """

    # load example image
    ex_img = nib.load(examplenii)
    ex_img.shape
    dim = ex_img.shape[0:3]
    if len(data.shape) < 2:
        nvol = 1
        data = data[:, np.newaxis]
    else:
        nvol = int(data.shape[1])

    # write data
    array_data = np.zeros((np.prod(dim), nvol))
    array_data[mask.flatten(), :] = data
    array_data = np.reshape(array_data, dim+(nvol,))
    array_img = nib.Nifti1Image(array_data, ex_img.affine, ex_img.header)
    nib.save(array_img, filename)

# --------------
# cifti routines
# --------------


def load_cifti(filename, vol=False, mask=None):

    # parse the name
    dnam, fnam = os.path.split(filename)
    fpref, fext = os.path.splitext(fnam)
    outstem = os.path.join(tempfile.gettempdir(),
                           str(os.getpid()) + "-" + fpref)

    # extract surface data from the cifti file
    print("extracting cifti surface data to ", outstem, '*.gii', sep="")
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
        print("extracting cifti volume data to ", niiname, sep="")
        os.system('wb_command -cifti-separate ' + filename +
                  ' COLUMN -volume-all ' + niiname)
        vol = load_nifti(niiname)
        out = np.concatenate((out, vol2vec(vol, mask)), axis=0)

    return out

# --------------
# ascii routines
# --------------


def load_ascii(filename):
    # currently very basic functionality.

    x = np.loadtxt(filename)
    return x


def save_ascii(data, filename):

    np.savetxt(filename, data)

# ----------------
# generic routines
# ----------------


def save(data, filename, example=None, mask=None, text=False):

    if file_type(filename) == 'cifti':
        x = save_cifti(data, filename, vol=True)
    elif file_type(filename) == 'nifti':
        x = save_nifti(data, filename, example, mask)
    elif text or file_type(filename) == 'text':
        x = save_ascii(data, filename)


def load(filename, mask=None, text=False):

    if file_type(filename) == 'cifti':
        x = load_cifti(filename, vol=True)
    elif file_type(filename) == 'nifti':
        x = load_nifti(filename, mask)
    elif text or file_type(filename) == 'text':
        x = load_ascii(filename)
        
    return x
