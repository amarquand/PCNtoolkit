from __future__ import print_function

import os
import re
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

try:  # run as a package if installed
    from pcntoolkit import configs
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.dirname(path)  # parent directory
    if path not in sys.path:
        sys.path.append(path)
    del path
    import configs

CIFTI_MAPPINGS = (
    "dconn",
    "dtseries",
    "pconn",
    "ptseries",
    "dscalar",
    "dlabel",
    "pscalar",
    "pdconn",
    "dpconn",
    "pconnseries",
    "pconnscalar",
)

CIFTI_VOL_ATLAS = "Atlas_ROIs.2.nii.gz"

PICKLE_PROTOCOL = configs.PICKLE_PROTOCOL

# ------------------------
# general utility routines
# ------------------------


def predictive_interval(s2_forward, cov_forward, multiplicator):
    """
    Calculates a predictive interval for the forward model
    """
    # calculates a predictive interval

    PI = np.zeros(len(cov_forward))
    for i, xdot in enumerate(cov_forward):
        s = np.sqrt(s2_forward[i])
        PI[i] = multiplicator * s
    return PI


def create_mask(data_array, mask, verbose=False):
    """
    Create a mask from a data array or a nifti file

    Basic usage::

            create_mask(data_array, mask, verbose)

    :param data_array: numpy array containing the data to write out
    :param mask: nifti image containing a mask for the image
    :param verbose: verbose output
    """

    # create a (volumetric) mask either from an input nifti or the nifti itself

    if mask is not None:
        if verbose:
            print("Loading ROI mask ...")
        maskvol = load_nifti(mask, vol=True)
        maskvol = maskvol != 0
    else:
        if len(data_array.shape) < 4:
            dim = data_array.shape[0:3] + (1,)
        else:
            dim = data_array.shape[0:3] + (data_array.shape[3],)

        if verbose:
            print("Generating mask automatically ...")
        if dim[3] == 1:
            maskvol = data_array[:, :, :] != 0
        else:
            maskvol = data_array[:, :, :, 0] != 0

    return maskvol


def vol2vec(dat, mask, verbose=False):
    """
    Vectorise a 3d image

    Basic usage::

                vol2vec(dat, mask, verbose)

    :param dat: numpy array containing the data to write out
    :param mask: nifti image containing a mask for the image
    :param verbose: verbose output
    """
    # vectorise a 3d image

    if len(dat.shape) < 4:
        dim = dat.shape[0:3] + (1,)
    else:
        dim = dat.shape[0:3] + (dat.shape[3],)

    # mask = create_mask(dat, mask=mask, verbose=verbose)
    if mask is None:
        mask = create_mask(dat, mask=mask, verbose=verbose)

    # mask the image
    maskid = np.where(mask.ravel())[0]
    dat = np.reshape(dat, (np.prod(dim[0:3]), dim[3]))
    dat = dat[maskid, :]

    # convert to 1-d array if the file only contains one volume
    if dim[3] == 1:
        dat = dat.ravel()

    return dat


def file_type(filename):
    """
    Determine the file type of a file

    Basic usage::

                    file_type(filename)

    :param filename: name of the file to check
    """
    # routine to determine filetype

    if filename.endswith((".dtseries.nii", ".dscalar.nii", ".dlabel.nii")):
        ftype = "cifti"
    elif filename.endswith((".nii.gz", ".nii", ".img", ".hdr")):
        ftype = "nifti"
    elif filename.endswith((".txt", ".csv", ".tsv", ".asc")):
        ftype = "text"
    elif filename.endswith((".pkl")):
        ftype = "binary"
    else:
        raise ValueError("I don't know what to do with " + filename)

    return ftype


def file_extension(filename):
    """
    Determine the file extension of a file (e.g. .nii.gz)

    Basic usage::

                        file_extension(filename)

    :param filename: name of the file to check
    """

    # routine to get the full file extension (e.g. .nii.gz, not just .gz)

    parts = filename.split(os.extsep)

    if parts[-1] == "gz":
        if parts[-2] == "nii" or parts[-2] == "img" or parts[-2] == "hdr":
            ext = parts[-2] + "." + parts[-1]
        else:
            ext = parts[-1]
    elif parts[-1] == "nii":
        if parts[-2] in CIFTI_MAPPINGS:
            ext = parts[-2] + "." + parts[-1]
        else:
            ext = parts[-1]
    else:
        ext = parts[-1]

    ext = "." + ext
    return ext


def file_stem(filename):
    """
    Determine the file stem of a file (e.g. /path/to/file.nii.gz -> file)

    Basic usage::

                                file_stem(filename)

    :param filename: name of the file to check
    """
    idx = filename.find(file_extension(filename))
    stm = filename[0:idx]

    return stm


# --------------
# nifti routines
# --------------


def load_nifti(datafile, mask=None, vol=False, verbose=False):
    """
    Load a nifti file into a numpy array

    Basic usage::

                    load_nifti(datafile, mask, vol, verbose)

    :param datafile: name of the file to load
    :param mask: nifti image containing a mask for the image
    :param vol: whether to load the image as a volume
    :param verbose: verbose output
    """

    if verbose:
        print("Loading nifti: " + datafile + " ...")
    img = nib.load(datafile)
    dat = img.get_fdata()

    if mask is not None:
        mask = load_nifti(mask, vol=True)

    if not vol:
        dat = vol2vec(dat, mask)

    return dat


def save_nifti(data, filename, examplenii, mask, dtype=None):
    """
    Write output to nifti

    Basic usage::

        save_nifti(data, filename mask, dtype)

    :param data: numpy array containing the data to write out
    :param filename: where to store it
    :param examplenii: nifti to copy the geometry and data type from
    :mask: nifti image containing a mask for the image
    :param dtype: data type for the output image (if different from the image)
    """

    # load mask
    if isinstance(mask, str):
        mask = load_nifti(mask, vol=True)
        mask = mask != 0

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
    array_data = np.reshape(array_data, dim + (nvol,))
    hdr = ex_img.header
    if dtype is not None:
        hdr.set_data_dtype(dtype)
        array_data = array_data.astype(dtype)
    array_img = nib.Nifti1Image(array_data, ex_img.affine, hdr)

    nib.save(array_img, filename)


# --------------
# cifti routines
# --------------


def load_cifti(filename, vol=False, mask=None, rmtmp=True):
    """
    Load a cifti file into a numpy array

    Basic usage::

                        load_cifti(filename, vol, mask, rmtmp)

    :param filename: name of the file to load
    :param vol: whether to load the image as a volume
    :param mask: nifti image containing a mask for the image
    :param rmtmp: whether to remove temporary files
    """
    # parse the name
    dnam, fnam = os.path.split(filename)
    fpref = file_stem(fnam)
    outstem = os.path.join(tempfile.gettempdir(), str(os.getpid()) + "-" + fpref)

    # extract surface data from the cifti file
    print("Extracting cifti surface data to ", outstem, "-*.func.gii", sep="")
    giinamel = outstem + "-left.func.gii"
    giinamer = outstem + "-right.func.gii"
    os.system(
        "wb_command -cifti-separate "
        + filename
        + " COLUMN -metric CORTEX_LEFT "
        + giinamel
    )
    os.system(
        "wb_command -cifti-separate "
        + filename
        + " COLUMN -metric CORTEX_RIGHT "
        + giinamer
    )

    # load the surface data
    giil = nib.load(giinamel)
    giir = nib.load(giinamer)
    Nimg = len(giil.darrays)
    Nvert = len(giil.darrays[0].data)
    if Nimg == 1:
        out = np.concatenate((giil.darrays[0].data, giir.darrays[0].data), axis=0)
    else:
        Gl = np.zeros((Nvert, Nimg))
        Gr = np.zeros((Nvert, Nimg))
        for i in range(0, Nimg):
            Gl[:, i] = giil.darrays[i].data
            Gr[:, i] = giir.darrays[i].data
        out = np.concatenate((Gl, Gr), axis=0)
    if rmtmp:
        # clean up temporary files
        os.remove(giinamel)
        os.remove(giinamer)

    if vol:
        niiname = outstem + "-vol.nii"
        print("Extracting cifti volume data to ", niiname, sep="")
        os.system(
            "wb_command -cifti-separate " + filename + " COLUMN -volume-all " + niiname
        )
        vol = load_nifti(niiname, vol=True)
        volmask = create_mask(vol)
        out = np.concatenate((out, vol2vec(vol, volmask)), axis=0)
        if rmtmp:
            os.remove(niiname)

    return out


def save_cifti(data, filename, example, mask=None, vol=True, volatlas=None):
    """
    Save a cifti file from a numpy array

    Basic usage::

                            save_cifti(data, filename, example, mask, vol, volatlas)

    :param data: numpy array containing the data to write out
    :param filename: where to store it
    :param example: example file to copy the geometry from
    :param mask: nifti image containing a mask for the image
    :param vol: whether to load the image as a volume
    :param volatlas: atlas to use for the volume
    """

    # do some sanity checks
    if data.dtype == "float32" or data.dtype == "float" or data.dtype == "float64":
        data = data.astype("float32")  # force 32 bit output
        dtype = "NIFTI_TYPE_FLOAT32"
    else:
        raise ValueError("Only float data types currently handled")

    if len(data.shape) == 1:
        Nimg = 1
        data = data[:, np.newaxis]
    else:
        Nimg = data.shape[1]

    # get the base filename
    dnam, fnam = os.path.split(filename)
    fstem = file_stem(fnam)

    # Split the template
    estem = os.path.join(tempfile.gettempdir(), str(os.getpid()) + "-" + fstem)
    giiexnamel = estem + "-left.func.gii"
    giiexnamer = estem + "-right.func.gii"
    os.system(
        "wb_command -cifti-separate "
        + example
        + " COLUMN -metric CORTEX_LEFT "
        + giiexnamel
    )
    os.system(
        "wb_command -cifti-separate "
        + example
        + " COLUMN -metric CORTEX_RIGHT "
        + giiexnamer
    )

    # write left hemisphere
    giiexl = nib.load(giiexnamel)
    Nvertl = len(giiexl.darrays[0].data)
    garraysl = []
    for i in range(0, Nimg):
        garraysl.append(
            nib.gifti.gifti.GiftiDataArray(data=data[0:Nvertl, i], datatype=dtype)
        )
    giil = nib.gifti.gifti.GiftiImage(darrays=garraysl)
    fnamel = fstem + "-left.func.gii"
    nib.save(giil, fnamel)

    # write right hemisphere
    giiexr = nib.load(giiexnamer)
    Nvertr = len(giiexr.darrays[0].data)
    garraysr = []
    for i in range(0, Nimg):
        garraysr.append(
            nib.gifti.gifti.GiftiDataArray(
                data=data[Nvertl : Nvertl + Nvertr, i], datatype=dtype
            )
        )
    giir = nib.gifti.gifti.GiftiImage(darrays=garraysr)
    fnamer = fstem + "-right.func.gii"
    nib.save(giir, fnamer)

    tmpfiles = [fnamer, fnamel, giiexnamel, giiexnamer]

    # process volumetric data
    if vol:
        niiexname = estem + "-vol.nii"
        os.system(
            "wb_command -cifti-separate " + example + " COLUMN -volume-all " + niiexname
        )
        niivol = load_nifti(niiexname, vol=True)
        if mask is None:
            mask = create_mask(niivol)

        if volatlas is None:
            volatlas = CIFTI_VOL_ATLAS
        fnamev = fstem + "-vol.nii"

        save_nifti(data[Nvertr + Nvertl :, :], fnamev, niiexname, mask)
        tmpfiles.extend([fnamev, niiexname])

    # write cifti
    fname = fstem + ".dtseries.nii"
    os.system(
        "wb_command -cifti-create-dense-timeseries "
        + fname
        + " -volume "
        + fnamev
        + " "
        + volatlas
        + " -left-metric "
        + fnamel
        + " -right-metric "
        + fnamer
    )

    # clean up
    for f in tmpfiles:
        os.remove(f)


# --------------
# ascii routines
# --------------


def load_pd(filename):
    """
    Load a csv file into a pandas dataframe

    Basic usage::

                    load_pd(filename)

    :param filename: name of the file to load
    """

    # based on pandas
    x = pd.read_csv(filename, sep=" ", header=None)
    return x


def save_pd(data, filename):
    """
    Save a pandas dataframe to a csv file

    Basic usage::

        save_pd(data, filename)

    :param data: pandas dataframe containing the data to write out
    :param filename: where to store it
    """
    # based on pandas
    data.to_csv(filename, index=None, header=None, sep=" ", na_rep="NaN")


def load_ascii(filename):
    """
    Load an ascii file into a numpy array

    Basic usage::

            load_ascii(filename)

    :param filename: name of the file to load
    """

    # based on pandas
    x = np.loadtxt(filename)
    return x


def save_ascii(data, filename):
    """
    Save a numpy array to an ascii file

    Basic usage::

        save_ascii(data, filename)

    :param data: numpy array containing the data to write out
    :param filename: where to store it
    """
    # based on pandas
    np.savetxt(filename, data)


# ----------------
# generic routines
# ----------------


def save(data, filename, example=None, mask=None, text=False, dtype=None):
    """
    Save a numpy array to a file

    Basic usage::

                save(data, filename, example, mask, text, dtype)

    :param data: numpy array containing the data to write out
    :param filename: where to store it
    :param example: example file to copy the geometry from
    :param mask: nifti image containing a mask for the image
    :param text: whether to write out a text file
    :param dtype: data type for the output image (if different from the image)
    """

    if file_type(filename) == "cifti":
        save_cifti(data.T, filename, example, vol=True)
    elif file_type(filename) == "nifti":
        save_nifti(data.T, filename, example, mask, dtype=dtype)
    elif text or file_type(filename) == "text":
        save_ascii(data, filename)
    elif file_type(filename) == "binary":
        data = pd.DataFrame(data)
        data.to_pickle(filename, protocol=PICKLE_PROTOCOL)


def load(filename, mask=None, text=False, vol=True):
    """
    Load a numpy array from a file

    Basic usage::

                    load(filename, mask, text, vol)

    :param filename: name of the file to load
    :param mask: nifti image containing a mask for the image
    :param text: whether to write out a text file
    :param vol: whether to load the image as a volume
    """

    if file_type(filename) == "cifti":
        x = load_cifti(filename, vol=vol)
    elif file_type(filename) == "nifti":
        x = load_nifti(filename, mask, vol=vol)
    elif text or file_type(filename) == "text":
        x = load_ascii(filename)
    elif file_type(filename) == "binary":
        x = pd.read_pickle(filename)
        x = x.to_numpy()
    return x


# -------------------
# sorting routines for batched in normative parallel
# -------------------


def tryint(s):
    """
    Try to convert a string to an integer

    Basic usage::

                    tryint(s)

    :param s: string to convert
    """

    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of numbers

    Basic usage::

                    alphanum_key(s)

    :param s: string to convert
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(l):
    """
    Sort a list of strings in a natural way

    Basic usage::

        sort_nicely(l)

    :param l: list of strings to sort
    """

    return sorted(l, key=alphanum_key)
