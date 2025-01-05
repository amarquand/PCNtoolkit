from typing import Any, Optional, Type

import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.blr.blr_data import BLRData
from pcntoolkit.regression_model.reg_conf import RegConf
from pcntoolkit.regression_model.regression_model import RegressionModel


class NormBLR(NormBase):
    """Bayesian Linear Regression implementation of the normative modeling framework.

    This class implements normative modeling using Bayesian Linear Regression (BLR) as the
    underlying regression model. It supports both homoscedastic and heteroscedastic noise models,
    as well as random intercepts for handling batch effects.

    Parameters
    ----------
    norm_conf : NormConf
        Normative model configuration object containing general settings.
    reg_conf : Optional[BLRConf], default=None
        BLR-specific regression configuration. If None, default configuration is used.

    Attributes
    ----------
    default_reg_conf : BLRConf
        Default regression configuration for BLR.
    regression_model_type : Type[BLR]
        Type of the regression model (BLR).
    current_regression_model : BLR
        Current active regression model instance.

    Notes
    -----
    The BLR implementation supports:
    - Linear and non-linear regression
    - Homoscedastic and heteroscedastic noise models
    - Random intercepts for batch effect correction
    - Computation of centiles and z-scores
    """

    def __init__(
        self,
        norm_conf: NormConf,
        reg_conf: Optional[BLRConf] = None,
        regression_model_type: Type[RegressionModel] = BLR,
    ) -> None:
        super().__init__(norm_conf)
        if reg_conf is None:
            reg_conf = BLRConf()
        self.default_reg_conf: BLRConf = reg_conf
        self.regression_model_type: Type[RegressionModel] = regression_model_type
        # self.current_regression_model: BLR = None  # type:ignore

    @classmethod
    def from_args(cls, args: Any) -> "NormBLR":
        norm_conf = NormConf.from_args(args)
        hbrconf = BLRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    def _fit(self, data: NormData, make_new_model: bool = False) -> None:
        blrdata = self.normdata_to_blrdata(data)
        self.focused_model.fit(blrdata)  # type: ignore
        assert self.focused_model.is_fitted

    def _predict(self, data: NormData) -> None:
        blrdata = self.normdata_to_blrdata(data)
        self.focused_model.predict(blrdata)  # type: ignore

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> None:
        self._fit(fit_data)
        self._predict(predict_data)

    def _generate_synthetic_data(
        self, data: NormData, n_synthetic_samples: int = 1000
    ) -> NormData:
        raise NotImplementedError(
            f"Generate synthetic data method not implemented for {self.__class__.__name__}"
        )

    def _transfer(self, data: NormData, **kwargs: Any) -> "BLR":
        raise NotImplementedError(
            f"Transfer method not implemented for {self.__class__.__name__}"
        )

    def _extend(self, data: NormData) -> "NormBLR":
        raise NotImplementedError(
            f"Extend method not implemented for {self.__class__.__name__}"
        )

    def _centiles(self, data: NormData, cdf: np.ndarray, **kwargs: Any) -> xr.DataArray:
        blrdata = self.normdata_to_blrdata(data)
        centiles = self.focused_model.centiles(blrdata, cdf)  # type: ignore
        return xr.DataArray(
            centiles,
            dims=["cdf", "datapoints"],
            coords={"cdf": cdf},
        )

    def _zscores(self, data: NormData) -> xr.DataArray:
        blrdata = self.normdata_to_blrdata(data)
        zscores = self.focused_model.zscores(blrdata)  # type: ignore
        return xr.DataArray(
            zscores,
            dims=["datapoints"],
        )

    def n_params(self) -> int:
        raise NotImplementedError(
            f"n_params method not implemented for {self.__class__.__name__}"
        )

    def create_design_matrix(
        self,
        data: NormData,
        linear: bool = False,
        intercept: bool = False,
        random_intercept: bool = False,
    ) -> np.ndarray:
        """Create design matrix for the model.

        Parameters
        ----------
        data : NormData
            Input data containing features and batch effects.
        linear : bool, default=False
            Include linear terms in the design matrix.
        intercept : bool, default=False
            Include intercept term in the design matrix.
        random_intercept : bool, default=False
            Include random intercepts for batch effects.

        Returns
        -------
        np.ndarray
            Design matrix combining all requested components.

        Raises
        ------
        ValueError
            If no components are selected for the design matrix.
        """
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
            mapped_batch_effects = self.map_batch_effects(data)
            for i, v in enumerate(self.unique_batch_effects.values()):
                acc.append(
                    np.eye(len(v))[mapped_batch_effects[:,i]],
                )
        if len(acc) == 0:
            raise ValueError("No design matrix created")

        return np.concatenate(acc, axis=1)

    def normdata_to_blrdata(self, data: NormData) -> BLRData:
        """Convert NormData to BLRData format.

        Parameters
        ----------
        data : NormData
            Input data in NormData format.

        Returns
        -------
        BLRData
            Converted data in BLRData format.

        Raises
        ------
        ValueError
            If regression configuration is not of type BLRConf.
        """
        reg_conf: RegConf = self.focused_model.reg_conf  # type: ignore
        if not isinstance(reg_conf, BLRConf):
            raise ValueError(
                f"Regression configuration is not of type BLRConf, got {type(reg_conf)}"
            )

        this_X = self.create_design_matrix(
            data,
            linear=True,
            intercept=reg_conf.intercept,
            random_intercept=reg_conf.random_intercept,
        )
        if self.focused_model.models_variance:
            this_var_X = self.create_design_matrix(
                data,
                linear=reg_conf.heteroskedastic,
                intercept=reg_conf.intercept_var,
                random_intercept=reg_conf.random_intercept_var,
            )
        else:
            this_var_X = None

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
        return blrdata

    @property
    def focused_model(self) -> BLR:
        """Get the currently focused BLR model.

        Returns
        -------
        BLR
            The currently focused Bayesian Linear Regression model.
        """
        return self[self.focused_var]  # type:ignore

    def make_serializable(self) -> None:
        """Nothing to do here"""
        pass
