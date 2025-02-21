import numpy as np
import xarray as xr

from pcntoolkit.regression_model.regression_model import RegressionModel


class TestModel(RegressionModel):
    """
    Test model for regression model testing.
    """

    def __init__(self, name: str):
        self.name = name

    def fit(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> None:
        self.is_fitted = True

    def forward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> xr.DataArray:
        return Y.copy()


    def backward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Z: xr.DataArray) -> xr.DataArray:
        return Z.copy()


    def elemwise_logp(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> xr.DataArray:
        logp = np.zeros(Y.shape)
        return xr.DataArray(logp, coords=Y.coords, dims=Y.dims, attrs=Y.attrs)


    def transfer(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> RegressionModel:
        return TestModel(self.name)
    
    @property
    def is_from_dict(self) -> bool:
        return self._is_from_dict   

    @is_from_dict.setter
    def is_from_dict(self, is_from_dict: bool) -> None:
        self._is_from_dict = is_from_dict

    def to_dict(self, path: str | None = None) -> dict:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, my_dict: dict, path: str) -> RegressionModel:
        return TestModel(my_dict["name"])

    @classmethod
    def from_args(cls, name: str, args: dict) -> RegressionModel:
        return TestModel(name)

    @property
    def has_batch_effect(self) -> bool:
        return False