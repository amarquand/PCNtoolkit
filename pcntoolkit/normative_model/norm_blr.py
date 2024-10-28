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

    def _centiles(
        self, data: NormData, cummulative_densities: list[float], *args, **kwargs
    ) -> xr.DataArray:
        """
        Compute centiles for the model at the given data points.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        The return type should be a DataArray with dimensions:
        - cummulative_densities
        - datapoints

        ```
        centiles = np.zeros((len(cummulative_densities), data.X.shape[0])))
        for i, zscore in enumerate(cummulative_densities):
            centiles[i, :] = *compute centiles for cummulative_density*

        return xr.DataArray(
            centiles,
            dims=["cummulative_densities", "datapoints"],
            coords={"cummulative_densities": cummulative_densities},
        )```
        """
        blrdata = self.normdata_to_blrdata(data)
        centiles = self.current_regression_model.centiles(
            blrdata, cummulative_densities
        )
        return xr.DataArray(
            centiles,
            dims=["cummulative_densities", "datapoints"],
            coords={"cummulative_densities": cummulative_densities},
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

    def normdata_to_blrdata(self, data: NormData, intercept=True) -> BLRData:
        if hasattr(data, "Phi") and data.Phi is not None:
            this_X = data.Phi.to_numpy()
        elif hasattr(data, "scaled_X") and data.scaled_X is not None:
            this_X = data.scaled_X.to_numpy()
        else:
            this_X = data.X.to_numpy()

        if hasattr(data, "scaled_y") and data.scaled_y is not None:
            this_y = data.scaled_y.to_numpy()
        else:
            this_y = data.y.to_numpy()

        # Create intercept
        if self.current_regression_model.reg_conf.intercept:
            this_X = np.hstack((this_X, np.ones((this_X.shape[0], 1))))

        # # For each of the columns in the batch_effects, create a one-hot encoding
        # for i in data.batch_effects.columns:
        #     eye = np.eye(i)
        #     this_X = np.hstack(
        #         (
        #             this_X,
        #             np.array(
        #                 [
        #                     [
        #                         1 if i[batch_effect] == j else 0
        #                         for j in range(len(i))
        #                     ]
        #                     for batch_effect in i[:, i]
        #                 ]
        #             ),
        #         )
        #     )

        blrdata = BLRData(
            X=this_X,
            y=this_y,
            batch_effects=data.batch_effects.to_numpy(),
            response_var=data.response_vars.to_numpy().item(),
        )
        blrdata.set_batch_effects_maps(data.attrs["batch_effects_maps"])
        return blrdata
