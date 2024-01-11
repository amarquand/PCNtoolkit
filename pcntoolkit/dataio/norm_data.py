from typing import Tuple, Union
import xarray as xr
from sklearn.model_selection import train_test_split
import numpy as np
class NormData(xr.Dataset):
    """This class is only here as a placeholder for now. It will be used to store the data for fitting normative models."""
    """Should keep track of the dimensions and coordinates of the data, and provide consistency between splits of the data."""


    def __init__(self, name, data_vars, coords, attrs) -> None:
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
        self.create_batch_effects_maps()
        self.name = name

    @classmethod
    def from_ndarrays(cls,name, X, y, batch_effects, attrs=None):
        return cls(name, {'X': (['datapoints', 'covariates'], X),'y':(['datapoints'],y),'batch_effects':(['datapoints', 'batch_effect_dims'],batch_effects)},coords={'datapoints':np.arange(X.shape[0]),'covariates':np.arange(X.shape[1]),'batch_effect_dims':[f"batch_effect_{i}" for i in range(batch_effects.shape[1])]}, attrs=attrs)

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

    @classmethod
    def from_dataframe(cls, name, dataframe, covariates, batch_effects, y, attrs=None):
        return cls(name, {'X': (['datapoints', 'covariates'], dataframe[covariates].to_numpy()),'y':(['datapoints'],dataframe[y].to_numpy()),'batch_effects':(['datapoints', 'batch_effect_dims'],dataframe[batch_effects].to_numpy())},coords={'datapoints':np.arange(dataframe.shape[0]),'covariates':covariates,'batch_effect_dims':batch_effects}, attrs=attrs)

    def split(self, splits: Tuple[float,...], split_names: Tuple[str,...] = None) -> Tuple['NormData',...]:
        """Split the data into 2 datasets."""
        
        if len(splits) != 2:
            raise NotImplementedError("Only 2 splits are supported for now.")
        
        if split_names is None:
            split_names = [f"split_{i}" for i in range(len(splits))]

        if len(splits) != len(split_names):
            raise ValueError("The number of splits must match the number of split names.")
    
        # the splits must sum to 1
        if sum(splits) != 1:
            raise ValueError("The splits must sum to 1.")

        # the splits must be between 0 and 1
        if any([split < 0 or split > 1 for split in splits]):
            raise ValueError("The splits must be between 0 and 1.")

        X1, X2, y1, y2, batch_effects1, batch_effects2 = train_test_split(self.X, self.y, self.batch_effects, test_size=splits[1], stratify=self.batch_effects,random_state=42)
        split1 = NormData(name=split_names[0], data_vars={'X': (['datapoints', 'covariates'], X1.data),'y':(['datapoints'],y1.data),'batch_effects':(['datapoints', 'batch_effect_dims'],batch_effects1.data)},coords={'datapoints':np.arange(X1.shape[0]),'covariates':self.covariates,'batch_effect_dims':self.batch_effect_dims}, attrs=self.attrs)
        split2 = NormData(name=split_names[1], data_vars={'X': (['datapoints', 'covariates'], X2.data),'y':(['datapoints'],y2.data),'batch_effects':(['datapoints', 'batch_effect_dims'],batch_effects2.data)},coords={'datapoints':np.arange(X2.shape[0]),'covariates':self.covariates,'batch_effect_dims':self.batch_effect_dims}, attrs=self.attrs)
        return split1, split2

                
    def create_batch_effects_maps(self):
        # create a dictionary with for each column in the batch effects, a dict from value to int
        self.batch_effects_maps = {}
        for i, dim in enumerate(self.batch_effect_dims.to_numpy()):
            self.batch_effects_maps[dim] = {value: j for j, value in enumerate(np.unique(self.batch_effects[:,i]))}

    def is_compatible_with(self, other):
        """Check if the data is compatible with another dataset."""
        same_covariates = np.all(self.covariates == other.covariates)
        same_batch_effect_dims = np.all(self.batch_effect_dims == other.batch_effect_dims)
        same_batch_effects_maps = self.batch_effects_maps == other.batch_effects_maps
        return same_covariates and same_batch_effect_dims and same_batch_effects_maps
    