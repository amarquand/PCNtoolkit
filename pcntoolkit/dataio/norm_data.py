import xarray as xr


class NormData(xr.Dataset):
    """This class is only here as a placeholder for now. It will be used to store the data for fitting normative models."""

    def __init__(self, data_vars, coords, attrs) -> None:
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

    @classmethod
    def from_fsl(cls, fsl_folder, config_params) -> 'NormData':
        """Load a normative dataset from a FSL file."""
        pass

    @classmethod
    def from_nifti(cls, nifti_folder, config_params) -> 'NormData':
        """Load a normative dataset from a Nifti file."""
        pass

    @classmethod
    def from_bids(cls, bids_folder, config_params) -> 'NormData':
        """Load a normative dataset from a BIDS dataset."""
        pass

    @classmethod
    def from_xarray(cls, xarray_dataset) -> 'NormData':
        """Load a normative dataset from an xarray dataset."""
        pass
