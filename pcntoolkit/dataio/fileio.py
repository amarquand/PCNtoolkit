# pcntoolkit/dataio/fileio.py
from __future__ import print_function

import os
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple  # Added for type hints

import nibabel as nib
import numpy as np
import pandas as pd

# Ensure your Output, Errors, Messages are correctly importable
# Example: from pcntoolkit.util.output import Errors, Messages, Output, Warnings
# For this standalone example, I'll assume they are available globally or adjust imports as per your structure.
# If pcntoolkit is a package, relative imports like:
from pcntoolkit.util.output import Errors, Messages, Output, Warnings

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.dirname(path)  # parent directory
if path not in sys.path:
    sys.path.append(path)
del path

CIFTI_MAPPINGS = (
    "dconn", "dtseries", "pconn", "ptseries", "dscalar", "dlabel",
    "pscalar", "pdconn", "dpconn", "pconnseries", "pconnscalar",
)
CIFTI_VOL_ATLAS = "Atlas_ROIs.2.nii.gz" # Make sure this atlas is accessible if needed by save_cifti

PICKLE_PROTOCOL = 4 # Standard pickle protocol

# --- Helper Functions (from your original file, ensure they are here) ---
def file_extension(filename: str) -> str:
    """Determine the file extension of a file (e.g. .nii.gz)"""
    # Handles compound extensions like .nii.gz correctly
    name, ext = os.path.splitext(filename)
    if ext == ".gz":
        name, ext2 = os.path.splitext(name)
        return ext2 + ext
    return ext

def file_stem(filename: str) -> str:
    """Determine the file stem of a file (e.g. /path/to/file.nii.gz -> file)"""
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    if _.lower() == ".gz": # Handle .nii.gz
        stem, _ = os.path.splitext(stem)
    return stem


def create_mask(data_array: np.ndarray, mask: Optional[np.ndarray] = None, verbose: bool = False) -> np.ndarray:
    """Create a mask from a data array or a nifti file (mask input is now ndarray)"""
    if mask is not None:
        if verbose: Output.print(Messages.LOADING_ROI_MASK) # Make sure Messages.LOADING_ROI_MASK exists
        return mask != 0  # Assume mask is already a loaded numpy array
    
    if verbose: Output.print(Messages.GENERATING_MASK_AUTOMATICALLY) # Make sure Messages.GENERATING_MASK_AUTOMATICALLY exists
    if data_array.ndim < 3: # Not volumetric, cannot create mask this way. Or adapt.
        raise ValueError("Cannot automatically create mask for non-volumetric data without explicit mask.")
    
    dims = data_array.shape
    if len(dims) < 4 or dims[3] == 1: # 3D or 4D with one volume
        return data_array[..., 0] != 0 if len(dims) > 3 else data_array != 0
    else: # 4D with multiple volumes, use first volume for mask
        return data_array[..., 0] != 0

def vol2vec(dat: np.ndarray, mask: Optional[np.ndarray], verbose: bool = False) -> np.ndarray:
    """Vectorise a 3d/4d image using a mask"""
    original_shape = dat.shape
    if len(original_shape) < 3: # Already flat or 2D
        if mask is not None and dat.ndim ==1: # Apply mask if 1D data and mask provided
             return dat[mask.ravel()]
        return dat # Or raise error if mask is expected

    if len(original_shape) < 4:
        dim = original_shape[0:3] + (1,)
        dat_reshaped = dat.reshape(dim) # Ensure it's 4D for consistent processing
    else:
        dim = original_shape
        dat_reshaped = dat

    if mask is None:
        current_mask = create_mask(dat_reshaped, mask=None, verbose=verbose)
    else:
        current_mask = mask

    if current_mask.shape != dim[0:3]:
        raise ValueError(f"Mask shape {current_mask.shape} does not match data's spatial dimensions {dim[0:3]}")

    mask_1d = current_mask.ravel()
    dat_vectorized = dat_reshaped.reshape(np.prod(dim[0:3]), dim[3])
    dat_masked = dat_vectorized[mask_1d, :]

    return dat_masked.ravel() if dim[3] == 1 else dat_masked


# --- File Type Detection ---
def detect_file_type(filename: str) -> str:
    """More robust file type detection."""
    fn_lower = filename.lower()
    ext = file_extension(fn_lower) # Use refined file_extension

    if ext in (".nii", ".nii.gz", ".img", ".hdr"):
        return "nifti"
    if "cifti" in filename_lower or any(fn_lower.endswith(c_ext) for c_ext in CIFTI_MAPPINGS) or \
       any(ext.endswith(c_simple_ext) for c_simple_ext in (".dtseries.nii", ".dscalar.nii", ".dlabel.nii")): # Simplified CIFTI check
        return "cifti"
    if ext == ".csv":
        return "csv"
    if ext == ".tsv":
        return "tsv"
    if ext in (".txt", ".asc", ".dat"): # Common text/ascii extensions
        return "text"
    if ext == ".pkl":
        return "pickle"
    
    # Fallback: try to infer from content if extension is ambiguous (e.g. .dat)
    if ext == ".dat": # .dat can be binary or text
        try:
            with open(filename, 'rb') as f: # read few bytes
                sample = f.read(1024)
            sample.decode('utf-8') # if it decodes, likely text
            return "text" 
        except UnicodeDecodeError:
            # Cannot decode as UTF-8, could be pickle or other binary.
            # For this toolkit, if .dat is not text, we'll assume user won't use it for pickle.
            Output.warning(f"File {filename} with .dat extension is binary but not identifiable as pickle. Treating as unknown.")
            raise ValueError(f"Unknown or ambiguous binary file type for: {filename}")

    raise ValueError(f"Unknown file type for: {filename}")


# --- Internal Loaders ---
def _load_nifti_internal(datafile: str, mask_path: Optional[str] = None, vol: bool = False) -> np.ndarray:
    img = nib.load(datafile)
    dat = img.get_fdata().copy() # Use .copy() to ensure data is writable and decoupled

    loaded_mask_array = None
    if mask_path is not None:
        mask_img = nib.load(mask_path)
        loaded_mask_array = mask_img.get_fdata().copy() != 0
    
    if not vol:
        # vol2vec expects mask as ndarray
        dat = vol2vec(dat, loaded_mask_array) 
    return dat

def _load_cifti_internal(filename: str, vol: bool = False, mask_path: Optional[str] = None, rmtmp: bool = True) -> np.ndarray:
    """Internal CIFTI loader using wb_command."""
    # This function relies heavily on `wb_command` being in PATH and functional.
    # Output/Messages/Errors need to be available.
    # file_stem, create_mask, _load_nifti_internal, vol2vec also need to be available.
    
    dnam, fnam = os.path.split(filename)
    fpref = file_stem(fnam) 
    # Create temp files in a more robust way if possible, or ensure permissions
    temp_dir = tempfile.gettempdir()
    outstem = os.path.join(temp_dir, f"{os.getpid()}-{fpref}")

    giinamel = f"{outstem}-left.func.gii"
    giinamer = f"{outstem}-right.func.gii"
    
    # Use subprocess for better error handling than os.system
    import subprocess

    def run_wb_command(cmd_list):
        try:
            # Output.print(f"Running: {' '.join(cmd_list)}") # For debugging
            process = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
            # if process.stdout: Output.print(process.stdout) # For debugging
            # if process.stderr: Output.warning(process.stderr) # For debugging
        except subprocess.CalledProcessError as e:
            Output.error(Errors.WB_COMMAND_FAILED, command=' '.join(e.cmd), error=e.stderr) # Ensure Errors.WB_COMMAND_FAILED exists
            raise
        except FileNotFoundError:
            Output.error(Errors.WB_COMMAND_NOT_FOUND) # Ensure Errors.WB_COMMAND_NOT_FOUND exists
            raise

    try:
        # Output.print(Messages.EXTRACTING_CIFTI_SURFACE_DATA, outstem=outstem) # Ensure Messages exists
        run_wb_command(["wb_command", "-cifti-separate", filename, "COLUMN", "-metric", "CORTEX_LEFT", giinamel])
        run_wb_command(["wb_command", "-cifti-separate", filename, "COLUMN", "-metric", "CORTEX_RIGHT", giinamer])

        giil = nib.load(giinamel)
        giir = nib.load(giinamer)
        
        # Ensure darrays exist and are not empty
        if not giil.darrays or not giir.darrays:
            raise ValueError("Gifti files created from CIFTI separation are empty or invalid.")

        Nimg = len(giil.darrays)
        Nvert_l = giil.darrays[0].data.shape[0]
        Nvert_r = giir.darrays[0].data.shape[0]

        if Nimg == 1:
            out = np.concatenate((giil.darrays[0].data.copy(), giir.darrays[0].data.copy()), axis=0)
        else:
            # Assuming all darrays have same number of vertices for their respective hemisphere
            Gl = np.zeros((Nvert_l, Nimg))
            Gr = np.zeros((Nvert_r, Nimg))
            for i in range(Nimg):
                Gl[:, i] = giil.darrays[i].data.copy()
                Gr[:, i] = giir.darrays[i].data.copy()
            out = np.concatenate((Gl, Gr), axis=0)

        if vol:
            niiname = f"{outstem}-vol.nii.gz" # Save as .nii.gz
            # Output.print(Messages.EXTRACTING_CIFTI_VOLUME_DATA, niiname=niiname) # Ensure Messages exists
            run_wb_command(["wb_command", "-cifti-separate", filename, "COLUMN", "-volume-all", niiname])
            
            vol_data = _load_nifti_internal(niiname, vol=True) # Load as volume
            
            loaded_mask_array = None
            if mask_path:
                mask_img = nib.load(mask_path)
                loaded_mask_array = mask_img.get_fdata().copy() != 0
            else: # Create mask from volume if no path provided
                loaded_mask_array = create_mask(vol_data, mask=None)
            
            vol_vec = vol2vec(vol_data, loaded_mask_array)
            out = np.concatenate((out, vol_vec), axis=0) # Concatenate vectorized volume data

    finally: # Cleanup
        if rmtmp:
            for f_to_remove in [giinamel, giinamer, niiname if vol else None]:
                if f_to_remove and os.path.exists(f_to_remove):
                    try:
                        os.remove(f_to_remove)
                    except OSError as e:
                        Output.warning(f"Could not remove temporary file {f_to_remove}: {e}")
    return out

# --- Main Load Function ---
def load(filename: str, **kwargs: Any) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Loads data from a file, dispatching to specific loaders.
    Returns a tuple of (numpy_array, column_names_list_or_None).
    kwargs are passed to specific loaders (e.g., pandas_kwargs for csv, nifti_kwargs for nifti).
    """
    ftype = detect_file_type(filename)
    data: np.ndarray
    columns: Optional[List[str]] = None

    try:
        if ftype == "csv":
            df = pd.read_csv(filename, **kwargs.get('pandas_kwargs', {}))
            data, columns = df.to_numpy(), df.columns.tolist()
        elif ftype == "tsv":
            df = pd.read_csv(filename, sep='\t', **kwargs.get('pandas_kwargs', {}))
            data, columns = df.to_numpy(), df.columns.tolist()
        elif ftype == "text": # .txt, .asc, .dat (if text)
            try: # Try pandas first for flexibility with headers/mixed types
                df = pd.read_csv(filename, delim_whitespace=True, **kwargs.get('pandas_kwargs', {}))
                data, columns = df.to_numpy(), df.columns.tolist()
            except Exception: # Fallback to np.loadtxt for simple numeric arrays
                data = np.loadtxt(filename, **kwargs.get('loadtxt_kwargs', {}))
        elif ftype == "pickle":
            obj = pd.read_pickle(filename) # Or use `import pickle; pickle.load(...)` for non-pandas
            if isinstance(obj, pd.DataFrame):
                data, columns = obj.to_numpy(), obj.columns.tolist()
            elif isinstance(obj, pd.Series):
                data, columns = obj.to_numpy(), ([obj.name] if obj.name else None)
            elif isinstance(obj, np.ndarray):
                data = obj
            else:
                raise TypeError(f"Pickled object in {filename} is {type(obj)}, expected DataFrame, Series, or ndarray.")
        elif ftype == "nifti":
            nifti_args = kwargs.get('nifti_kwargs', {})
            data = _load_nifti_internal(filename, mask_path=nifti_args.get('mask'), vol=nifti_args.get('vol', False))
        elif ftype == "cifti":
            cifti_args = kwargs.get('cifti_kwargs', {})
            data = _load_cifti_internal(filename, 
                                      vol=cifti_args.get('vol', False), 
                                      mask_path=cifti_args.get('mask'), 
                                      rmtmp=cifti_args.get('rmtmp', True))
        else:
            # This case should ideally be caught by detect_file_type raising an error
            raise ValueError(f"File type '{ftype}' for {filename} is known but not handled in load function.")

    except ValueError as ve: # Catch known errors from helpers
        raise ve
    except nib.filebasedimages.ImageFileError as nibe: # Specific NiBabel error
        Output.error(Errors.NIFTI_LOAD_FAILED, filename=filename, error_message=str(nibe)) # Ensure Errors.NIFTI_LOAD_FAILED exists
        raise
    except subprocess.CalledProcessError as cpe: # From wb_command
        # Already handled by run_wb_command, but re-raise to stop execution
        raise
    except Exception as e: # Catch-all for unexpected issues during loading a specific type
        # Use specific error messages from your Errors enum/class
        if ftype == "cifti": Output.warning(Warnings.LOAD_CIFTI_GENERIC_EXCEPTION.format(str(e)))
        elif ftype == "nifti": Output.warning(Warnings.LOAD_NIFTI_GENERIC_EXCEPTION.format(str(e)))
        Output.error(Errors.FILE_LOAD_FAILED, filename=filename, file_type=ftype, error_message=str(e)) # Ensure Errors.FILE_LOAD_FAILED exists
        raise RuntimeError(f"Failed to load {filename} ({ftype}): {e}") from e
        
    # Ensure data is 2D if it's meant to be tabular but loaded as 1D (e.g. single column/row)
    # This might be too aggressive; depends on expected output for single vectors.
    # if data.ndim == 1 and ftype in ["csv", "tsv", "text", "pickle"]:
    #     data = data[:, np.newaxis] # Or handle based on context

    return data, columns

# --- Save Routines (Keep your existing save functions) ---
# Remember to update them if they use the old `load` or `load_x` functions internally.
# For example, if `save_nifti` loads an example NIFTI, it should use `_load_nifti_internal`.

# Example: Your save_nifti might look like this (simplified)
def save_nifti(data: np.ndarray, filename: str, example_nii_path: str, mask_array: Optional[np.ndarray] = None, dtype=None):
    ex_img = nib.load(example_nii_path)
    # ... (rest of your save_nifti logic, using mask_array directly) ...
    # Create Nifti1Image and save
    hdr = ex_img.header
    if dtype is not None:
        hdr.set_data_dtype(dtype)
        array_data_typed = data.astype(dtype) # Ensure data is correct type before shaping
    else:
        array_data_typed = data
        # hdr.set_data_dtype(data.dtype) # Or set based on input data's dtype

    # Inverse of vol2vec essentially:
    # This part depends heavily on how your data and mask relate.
    # If data is already volumetric, just ensure shape matches example.
    # If data is vectorized, you need the mask to map it back.
    output_shape = ex_img.shape
    if len(output_shape) > 3 and data.shape[-1] != output_shape[-1] and data.ndim == ex_img.ndim-1: # data is (voxels, time) example is (x,y,z,time)
        if mask_array is None:
            raise ValueError("Mask is required to save vectorized data back to volume.")
        
        # Ensure mask is boolean
        mask_array_bool = mask_array.astype(bool)
        
        # Reconstruct volumetric data
        # Example assumes data is (n_voxels_in_mask, n_timepoints_or_features)
        # or (n_voxels_in_mask,)
        
        num_spatial_voxels = np.prod(output_shape[:3])
        num_features = data.shape[1] if data.ndim > 1 else 1
        
        # Initialize with zeros or appropriate fill value
        volumetric_data_flat = np.zeros((num_spatial_voxels, num_features), dtype=array_data_typed.dtype)
        
        if data.ndim == 1: # (n_voxels_in_mask,)
             volumetric_data_flat[mask_array_bool.ravel(), 0] = array_data_typed
        else: # (n_voxels_in_mask, n_features)
            volumetric_data_flat[mask_array_bool.ravel(), :] = array_data_typed
            
        final_data_shaped = volumetric_data_flat.reshape(output_shape[0:3] + (num_features,))

    elif array_data_typed.shape == output_shape : # Data is already in correct volumetric shape
        final_data_shaped = array_data_typed
    else:
        raise ValueError(f"Data shape {array_data_typed.shape} and example NIFTI shape {output_shape} are incompatible. Provide mask if data is vectorized.")

    array_img = nib.Nifti1Image(final_data_shaped, ex_img.affine, hdr)
    nib.save(array_img, filename)


# ... (Your other save functions: save_cifti, save_pd, save_ascii, and the generic `save`)
# For `save_cifti`, it also calls `load_nifti` internally, ensure it uses `_load_nifti_internal`.
# The generic `save` function will use `detect_file_type`.

def save(data: np.ndarray, filename: str, example: Optional[str] = None, mask: Optional[np.ndarray] = None, text: bool = False, dtype=None):
    """ Generic save function (adapted from yours) """
    ftype = detect_file_type(filename)

    if ftype == "cifti":
        if example is None:
            raise ValueError("Example CIFTI file is required to save data as CIFTI.")
        # save_cifti might need data transposed depending on its convention. User's original had data.T for NIFTI.
        # Assuming save_cifti handles its own data orientation.
        # Your save_cifti implementation details are important here.
        # save_cifti(data, filename, example, mask=mask, vol=True) # Assuming vol=True is default for save
        raise NotImplementedError("save_cifti integration needed with correct parameters.")
    elif ftype == "nifti":
        if example is None:
            raise ValueError("Example NIFTI file is required to save data as NIFTI.")
        save_nifti(data, filename, example, mask_array=mask, dtype=dtype) # Pass mask as array
    elif text or ftype == "text":
        np.savetxt(filename, data) # save_ascii(data, filename)
    elif ftype == "pickle": # Assuming data is suitable for pickling (e.g. DataFrame for pd.to_pickle)
        if isinstance(data, np.ndarray) and data.ndim <= 2 : # convert to DataFrame to save with column names if desired
             pd.DataFrame(data).to_pickle(filename, protocol=PICKLE_PROTOCOL)
        else: # Or just pickle the raw object if not DataFrame like
            with open(filename, 'wb') as f:
                import pickle
                pickle.dump(data, f, protocol=PICKLE_PROTOCOL)
    else:
        raise ValueError(f"Unsupported file type for saving: {ftype}")
        
# --- Backup function (from your original file) ---
def create_incremental_backup(filepath: str) -> str:
    """Creates an incremental backup of a file using the `.bak{n}` naming scheme."""
    # ... (Your existing implementation of create_incremental_backup) ...
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename) # os.path.splitext correctly handles "file.nii.gz" -> ("file.nii", ".gz")
    
    # Refined regex for extensions like .nii.gz
    # Base name part: re.escape(name). If name was "file.nii", this is "file\.nii"
    # Extension part: re.escape(ext). If ext was ".gz", this is "\.gz"
    
    # If filepath is "archive.tar.gz", name="archive.tar", ext=".gz"
    # If filepath is "image.nii.gz", name="image.nii", ext=".gz"
    # If filepath is "data.csv", name="data", ext=".csv"

    # We need to correctly handle the part of the name before ".bak".
    # If filename is "mydata.csv", backup is "mydata.bak1.csv"
    # If filename is "image.nii.gz", backup is "image.nii.bak1.gz"
    
    # Let's define stem as the part before the final extension, and final_ext as the final extension.
    # If there's a compound extension (like .nii.gz), we want to insert .bak before the first part of it.
    
    actual_stem = name # This is "image.nii" from "image.nii.gz" or "data" from "data.csv"
    actual_ext = ext  # This is ".gz" from "image.nii.gz" or ".csv" from "data.csv"

    # Special handling for known compound extensions that should be kept together *after* .bakN
    # For example, if we have "myfile.nii.gz", we want "myfile.bak1.nii.gz"
    # The current split gives name="myfile.nii", ext=".gz" -> "myfile.nii.bak1.gz"
    # If we have "myfile.tar.gz", name="myfile.tar", ext=".gz" -> "myfile.tar.bak1.gz"
    # This logic seems okay. The regex needs to match this.
    # name_part_for_regex = name # This is "file.nii" or "data"
    
    regex_pattern = rf"^{re.escape(actual_stem)}\.bak(\d+){re.escape(actual_ext)}$"
    regex = re.compile(regex_pattern)

    if not os.path.exists(filepath):
        # Create an empty file if it doesn't exist, then back it up
        Output.print(f"File {filepath} does not exist. Creating and backing it up.")
        with open(filepath, 'w') as f:
            pass # Create empty file
    
    existing_backups = [
        fn for fn in os.listdir(directory)
        if regex.match(fn)
    ]

    numbers = [
        int(regex.match(fn).group(1)) for fn in existing_backups
    ] if existing_backups else []

    next_n = max(numbers, default=0) + 1
    backup_name = f"{actual_stem}.bak{next_n}{actual_ext}"
    backup_path = os.path.join(directory, backup_name)

    shutil.copy2(filepath, backup_path)
    Output.print(f"Backup created: {backup_path}")
    return backup_path