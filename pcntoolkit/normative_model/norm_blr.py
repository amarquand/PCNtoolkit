import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.blr.blr_data import BLRData


class NormBLR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: BLRConf = None):
        super().__init__(norm_conf)
        if reg_conf is None:
            reg_conf = BLRConf
        self.default_reg_conf: BLRConf = reg_conf
        self.regression_model_type = BLR
        self.current_regression_model: BLR = None

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_args(args)
        hbrconf = BLRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    def _fit(self, data: NormData, hyp0=None):
        """
        Fit model on data.
        """

        blrdata = self.normdata_to_blrdata(data)
        self.current_regression_model.fit(blrdata, hyp0)
        # assert the model is fitted
        assert self.current_regression_model.is_fitted

    def _predict(
        self,
        data: NormData,
    ) -> NormData:
        """
        Make predictions on data using model.
        """
        blrdata = self.normdata_to_blrdata(data)
        self.current_regression_model.predict(blrdata)

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Fit and predict on data using self.model.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}"
        )

    def _transfer(self, data: NormData) -> NormBase:
        """
        Transfer the model to a new dataset.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Transfer method not implemented for {self.__class__.__name__}"
        )

    def _extend(self, data: NormData):
        """
        Extend the model to a new dataset.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Extend method not implemented for {self.__class__.__name__}"
        )

    def _tune(self, data: NormData):
        raise NotImplementedError(
            f"Tune method not implemented for {self.__class__.__name__}"
        )

    def _merge(self, other: NormBase):
        raise NotImplementedError(
            f"Merge method not implemented for {self.__class__.__name__}"
        )

    def _centiles(self, data: NormData, cdf: np.ndarray) -> xr.DataArray:
        """
        Compute centiles for the model at the given data points.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        The return type should be a DataArray with dimensions:
        - cdf
        - datapoints

        ```
        centiles = np.zeros((len(cdf), data.X.shape[0])))
        for i, zscore in enumerate(cdf):
            centiles[i, :] = *compute centiles for cummulative_density*

        return xr.DataArray(
            centiles,
            dims=["cdf", "datapoints"],
            coords={"cdf": cdf},
        )```
        """
        blrdata = self.normdata_to_blrdata(data)
        centiles = self.current_regression_model.centiles(blrdata, cdf)
        return xr.DataArray(
            centiles,
            dims=["cdf", "datapoints"],
            coords={"cdf": cdf},
        )

    def _zscores(self, data: NormData) -> xr.DataArray:
        """
        Compute zscores for the model at the given data points.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        The return type should be a DataArray with dimensions:
        - datapoints

        ```
        zscores = *compute zscores for data*
        return xr.DataArray(
            zscores,
            dims=["datapoints"],
        )```
        """
        blrdata = self.normdata_to_blrdata(data)
        zscores = self.current_regression_model.zscores(blrdata)
        return xr.DataArray(
            zscores,
            dims=["datapoints"],
        )

    def n_params(self) -> xr.DataArray:
        """
        compute the number of parameters for the model.
        """
        raise NotImplementedError(
            f"n_params method not implemented for {self.__class__.__name__}"
        )

    def create_design_matrix(
        self, data: NormData, linear=False, intercept=False, random_intercept=False
    ) -> np.ndarray:
        acc = []
        if linear:
            if hasattr(data, "Phi") and data.Phi is not None:
                acc.append(data.Phi.to_numpy())
            elif hasattr(data, "scaled_X") and data.scaled_X is not None:
                acc.append(data.scaled_X.to_numpy())
            else:
                acc.append(data.X.to_numpy())

        if intercept:
            acc.append(np.ones((data.X.shape[0], 1)))

        # Create one-hot encoding for random intercept
        if random_intercept:
            for i in data.batch_effect_dims:
                cur_be = data.batch_effects.sel(batch_effect_dims=i)
                cur_be_id = np.vectorize(
                    data.attrs["batch_effects_maps"][i.item()].get
                )(cur_be.values)
                acc.append(
                    np.eye(len(data.attrs["batch_effects_maps"][i.item()]))[cur_be_id],
                )
        if len(acc) == 0:
            return None
        return np.concatenate(acc, axis=1)

    def normdata_to_blrdata(self, data: NormData) -> BLRData:
        this_X = self.create_design_matrix(
            data,
            linear=True,
            intercept=self.current_regression_model.reg_conf.intercept,
            random_intercept=self.current_regression_model.reg_conf.random_intercept,
        )

        this_var_X = self.create_design_matrix(
            data,
            linear=self.current_regression_model.reg_conf.heteroskedastic,
            intercept=self.current_regression_model.reg_conf.intercept_var,
            random_intercept=self.current_regression_model.reg_conf.random_intercept_var,
        )

        if hasattr(data, "scaled_y") and data.scaled_y is not None:
            this_y = data.scaled_y.to_numpy()
        else:
            this_y = data.y.to_numpy()

        blrdata = BLRData(
            X=this_X,
            y=this_y,
            var_X=this_var_X,
            batch_effects=data.batch_effects.to_numpy(),
            response_var=data.response_vars.to_numpy().item(),
        )
        blrdata.set_batch_effects_maps(data.attrs["batch_effects_maps"])
        return blrdata
