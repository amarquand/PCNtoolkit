"""Warping functions for transforming non-Gaussian to Gaussian distributions.

This module implements various warping functions used to transform non-Gaussian 
response variables to Gaussian variables for use in Gaussian process models. The 
transformations are based on the compositionally-warped Gaussian processes framework 
from Rios and Torab (2019).

The module provides the following warping functions:
    - WarpLog: Logarithmic transformation
    - WarpAffine: Affine transformation (y = a + b*x)
    - WarpBoxCox: Box-Cox transformation
    - WarpSinArcsinh: Sinh-arcsinh transformation
    - WarpCompose: Composition of multiple warping functions

Each warping function implements three core methods:
    - f(): Forward transformation (non-Gaussian -> Gaussian)
    - invf(): Inverse transformation (Gaussian -> non-Gaussian)
    - df(): Derivative of the transformation

Example
-------
>>> from pcntoolkit.regression_model.blr.warp import WarpBoxCox, WarpCompose
>>> # Single warping function
>>> warp = WarpBoxCox()
>>> y_gaussian = warp.f(x, params=[0.5])
>>> # Composition of warping functions
>>> warp = WarpCompose(['WarpBoxCox', 'WarpAffine'])
>>> y_gaussian = warp.f(x, params=[0.5, 0.0, 1.0])

References
----------
.. [1] Rios, G., & Tobar, F. (2019). Compositionally-warped Gaussian processes.
       Neural Networks, 118, 235-246.
       https://www.sciencedirect.com/science/article/pii/S0893608019301856

See Also
--------
pcntoolkit.regression_model.blr : Bayesian linear regression module
"""

from abc import ABC, abstractmethod
from ast import literal_eval
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  #type: ignore


class WarpBase(ABC):
    """Base class for likelihood warping functions.

    This class implements warping functions following Rios and Torab (2019)
    Compositionally-warped Gaussian processes [1]_.

    All Warps must define the following methods:
        - get_n_params(): Return number of parameters
        - f(): Warping function (Non-Gaussian field -> Gaussian)
        - invf(): Inverse warp
        - df(): Derivatives
        - warp_predictions(): Compute predictive distribution

    References
    ----------
    .. [1] Rios, G., & Tobar, F. (2019). Compositionally-warped Gaussian processes.
           Neural Networks, 118, 235-246.
           https://www.sciencedirect.com/science/article/pii/S0893608019301856
    """

    def __init__(self) -> None:
        self.n_params: int = 0

    def get_n_params(self) -> int:
        """Return the number of parameters required by the warping function.

        Returns
        -------
        int
            Number of parameters

        Raises
        ------
        AssertionError
            If warp function is not initialized
        """
        assert not np.isnan(self.n_params), "Warp function not initialised"
        return self.n_params

    def warp_predictions(
        self,
        mu: NDArray[np.float64],
        s2: NDArray[np.float64],
        param: List[float],
        percentiles: List[float] | None= None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute warped predictions from a Gaussian predictive distribution.

        Parameters
        ----------
        mu : NDArray[np.float64]
            Gaussian predictive mean
        s2 : NDArray[np.float64]
            Predictive variance
        param : List[float]
            Warping parameters
        percentiles : List[float], optional
            Desired percentiles of the warped likelihood, by default [0.025, 0.975]

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            - median: Median of the predictive distribution
            - pred_interval: Predictive interval(s)
        """

        if percentiles is None:
            percentiles = [0.025, 0.975]

        # Compute percentiles of a standard Gaussian
        N = norm
        Z = N.ppf(percentiles)

        # find the median (using mu = median)
        median = self.invf(mu, param)

        # compute the predictive intervals (non-stationary)
        pred_interval = np.zeros((len(mu), len(Z)))
        for i, z in enumerate(Z):
            pred_interval[:, i] = self.invf(mu + np.sqrt(s2) * z, param)

        return median, pred_interval

    @abstractmethod
    def f(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Evaluate the warping function.

        Maps non-Gaussian response variables to Gaussian variables.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values to warp
        param : List[float]
            Warping parameters

        Returns
        -------
        NDArray[np.float64]
            Warped values
        """

    @abstractmethod
    def invf(self, y: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Evaluate the inverse warping function.

        Maps Gaussian latent variables to non-Gaussian response variables.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input values to inverse warp
        param : List[float]
            Warping parameters

        Returns
        -------
        NDArray[np.float64]
            Inverse warped values
        """

    @abstractmethod
    def df(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Return the derivative of the warp, dw(x)/dx.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        param : List[float]
            Warping parameters

        Returns
        -------
        NDArray[np.float64]
            Derivative values
        """


class WarpLog(WarpBase):
    """Logarithmic warping function.

    Implements y = log(x) warping transformation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_params = 0

    def f(
        self, x: NDArray[np.float64], param: Optional[List[float]] = None
    ) -> NDArray[np.float64]:
        """Apply logarithmic warping.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : Optional[List[float]], optional
            Not used for logarithmic warp, by default None

        Returns
        -------
        NDArray[np.float64]
            log(x)
        """
        return np.log(x)

    def invf(
        self, y: NDArray[np.float64], param: Optional[List[float]] = None
    ) -> NDArray[np.float64]:
        """Apply inverse logarithmic warping.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input values
        params : Optional[List[float]], optional
            Not used for logarithmic warp, by default None

        Returns
        -------
        NDArray[np.float64]
            exp(y)
        """
        return np.exp(y)

    def df(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Compute derivative of logarithmic warp.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Not used for logarithmic warp

        Returns
        -------
        NDArray[np.float64]
            1/x
        """
        return 1 / x


class WarpAffine(WarpBase):
    """Affine warping function.

    Implements affine transformation y = a + b*x where:
        - a: offset parameter
        - b: scale parameter (constrained positive through exp transform)
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_params = 2

    def _get_params(self, param: List[float]) -> Tuple[float, float]:
        """Extract and transform the affine parameters.

        Parameters
        ----------
        param : List[float]
            List containing [a, log(b)] where:
                a: offset parameter
                log(b): log of scale parameter

        Returns
        -------
        Tuple[float, float]
            Tuple of (a, b) parameters

        Raises
        ------
        ValueError
            If param length doesn't match n_params
        """
        if len(param) != self.n_params:
            raise ValueError("number of parameters must be " + str(self.n_params))
        return param[0], np.exp(param[1])

    def f(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply affine warping.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Affine parameters [a, log(b)]

        Returns
        -------
        NDArray[np.float64]
            a + b*x
        """
        a, b = self._get_params(param)
        y = a + b * x
        return y

    def invf(self, y: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply inverse affine warping.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input values
        params : List[float]
            Affine parameters [a, log(b)]

        Returns
        -------
        NDArray[np.float64]
            (y - a)/b
        """
        a, b = self._get_params(param)
        x = (y - a) / b
        return x

    def df(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Compute derivative of affine warp.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Affine parameters [a, log(b)]

        Returns
        -------
        NDArray[np.float64]
            Constant derivative b
        """
        _, b = self._get_params(param)
        df = np.ones(x.shape) * b
        return df


class WarpBoxCox(WarpBase):
    """Box-Cox warping function.

    Implements the Box-Cox transform with a single parameter (lambda):
    y = (sign(x) * abs(x) ** lambda - 1) / lambda

    This follows the generalization in Bicken and Doksum (1981) JASA 76
    and allows x to assume negative values.

    References
    ----------
    .. [1] Bickel, P. J., & Doksum, K. A. (1981). An analysis of transformations
           revisited. Journal of the American Statistical Association, 76(374), 296-311.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_params = 1

    def _get_params(self, param: List[float]) -> NDArray[np.float64]:
        """Extract and transform the Box-Cox parameter.

        Parameters
        ----------
        param : List[float]
            List containing [log(lambda)]

        Returns
        -------
        float
            Transformed lambda parameter
        """
        return np.exp(np.array(param))

    def f(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply Box-Cox warping.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Box-Cox parameter [log(lambda)]

        Returns
        -------
        NDArray[np.float64]
            Warped values
        """
        lam = self._get_params(param)

        if lam == 0:
            y = np.log(x)
        else:
            y = (np.sign(x) * np.abs(x) ** lam - 1) / lam
        return y

    def invf(self, y: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply inverse Box-Cox warping.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input values
        params : List[float]
            Box-Cox parameter [log(lambda)]

        Returns
        -------
        NDArray[np.float64]
            Inverse warped values
        """
        lam = self._get_params(param)

        if lam == 0:
            x = np.exp(y)
        else:
            x = np.sign(lam * y + 1) * np.abs(lam * y + 1) ** (1 / lam)
        return x

    def df(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Compute derivative of Box-Cox warp.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Box-Cox parameter [log(lambda)]

        Returns
        -------
        NDArray[np.float64]
            Derivative values
        """
        lam = self._get_params(param)
        dx = np.abs(x) ** (lam - 1)
        return dx


class WarpSinArcsinh(WarpBase):
    """Sin-hyperbolic arcsin warping function.

    Implements warping function y = sinh(b * arcsinh(x) - a) with two parameters:
        - a: controls skew
        - b: controls kurtosis (constrained positive through exp transform)

    Properties:
        - a = 0: symmetric
        - a > 0: positive skew
        - a < 0: negative skew
        - b = 1: mesokurtic
        - b > 1: leptokurtic
        - b < 1: platykurtic

    Uses alternative parameterization from Jones and Pewsey (2019) where:
    y = sinh(b * arcsinh(x) + epsilon * b) and a = -epsilon*b

    References
    ----------
    .. [1] Jones, M. C., & Pewsey, A. (2019). Sigmoid-type distributions:
           Generation and inference. Significance, 16(1), 12-15.
    .. [2] Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions.
           Biometrika, 96(4), 761-780.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_params = 2

    def _get_params(self, param: List[float]) -> Tuple[float, float]:
        """Extract and transform the sinh-arcsinh parameters.

        Parameters
        ----------
        param : List[float]
            List containing [epsilon, log(b)]

        Returns
        -------
        Tuple[float, float]
            Tuple of (a, b) parameters where a = -epsilon*b

        Raises
        ------
        ValueError
            If param length doesn't match n_params
        """
        if len(param) != self.n_params:
            raise ValueError("number of parameters must be " + str(self.n_params))

        epsilon = param[0]
        b = np.exp(param[1])
        a = -epsilon * b

        return a, b

    def f(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply sinh-arcsinh warping.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Parameters [epsilon, log(b)]

        Returns
        -------
        NDArray[np.float64]
            sinh(b * arcsinh(x) - a)
        """
        a, b = self._get_params(param)
        y = np.sinh(b * np.arcsinh(x) - a)
        return y

    def invf(self, y: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply inverse sinh-arcsinh warping.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input values
        params : List[float]
            Parameters [epsilon, log(b)]

        Returns
        -------
        NDArray[np.float64]
            sinh((arcsinh(y) + a) / b)
        """
        a, b = self._get_params(param)
        x = np.sinh((np.arcsinh(y) + a) / b)
        return x

    def df(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Compute derivative of sinh-arcsinh warp.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        params : List[float]
            Parameters [epsilon, log(b)]

        Returns
        -------
        NDArray[np.float64]
            (b * cosh(b * arcsinh(x) - a)) / sqrt(1 + x^2)
        """
        a, b = self._get_params(param)
        dx = (b * np.cosh(b * np.arcsinh(x) - a)) / np.sqrt(1 + x**2)
        return dx


class WarpCompose(WarpBase):
    """Composition of multiple warping functions.

    Allows chaining multiple warps together. Warps are applied in sequence:
    y = warp_n(...warp_2(warp_1(x)))

    Example
    -------
    W = WarpCompose(('WarpBoxCox', 'WarpAffine'))
    """

    def __init__(
        self, warpnames: Optional[List[str]] = None, debugwarp: bool = False
    ) -> None:
        """Initialize composed warp.

        Parameters
        ----------
        warpnames : Optional[List[str]], optional
            List of warp class names to compose, by default None
        debugwarp : bool, optional
            Enable debug printing, by default False

        Raises
        ------
        ValueError
            If warpnames is None
        """
        super().__init__()
        if warpnames is None:
            raise ValueError("A list of warp functions is required")
        self.debugwarp = debugwarp
        self.warps: List[WarpBase] = []
        self.n_params = 0
        for wname in warpnames:
            warp = literal_eval(wname + "()")  # type: ignore
            self.n_params += warp.get_n_params()
            self.warps.append(warp)

    def f(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply composed warping functions.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        param : List[float]
            Combined parameters for all warps

        Returns
        -------
        NDArray[np.float64]
            Warped values after applying all transforms
        """
        theta = param
        theta_offset = 0

        if self.debugwarp:
            print("begin composition")
        for ci, warp in enumerate(self.warps):
            n_params_c = warp.get_n_params()
            theta_c = [theta[c] for c in range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c

            if self.debugwarp:
                print("f:", ci, theta_c, warp)

            if ci == 0:
                fw = warp.f(x, theta_c)
            else:
                fw = warp.f(fw, theta_c)
        return fw

    def invf(self, y: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Apply inverse composed warping functions.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input values
        param : List[float]
            Combined parameters for all warps

        Returns
        -------
        NDArray[np.float64]
            Inverse warped values after applying all inverse transforms
        """
        theta = param

        n_params = 0
        n_warps = 0
        if self.debugwarp:
            print("begin composition")

        for ci, warp in enumerate(self.warps):
            n_params += warp.get_n_params()
            n_warps += 1
        theta_offset = n_params
        for ci, warp in reversed(list(enumerate(self.warps))):
            n_params_c = warp.get_n_params()
            theta_offset -= n_params_c
            theta_c = [theta[c] for c in range(theta_offset, theta_offset + n_params_c)]

            if self.debugwarp:
                print("invf:", theta_c, warp)

            if ci == n_warps - 1:
                finvw = warp.invf(y, theta_c)
            else:
                finvw = warp.invf(finvw, theta_c)

        return finvw

    def df(self, x: NDArray[np.float64], param: List[float]) -> NDArray[np.float64]:
        """Compute derivative of composed warping functions.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input values
        param : List[float]
            Combined parameters for all warps

        Returns
        -------
        NDArray[np.float64]
            Combined derivative values
        """
        theta = param
        theta_offset = 0
        if self.debugwarp:
            print("begin composition")
        for ci, warp in enumerate(self.warps):
            n_params_c = warp.get_n_params()

            theta_c = [theta[c] for c in range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c

            if self.debugwarp:
                print("df:", ci, theta_c, warp)

            if ci == 0:
                dfw = warp.df(x, theta_c)
            else:
                dfw = warp.df(dfw, theta_c)

        return dfw