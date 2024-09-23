import numpy as np
from abc import ABC
from abc import abstractmethod
from scipy.stats import norm


class WarpBase(ABC):
    """
    Base class for likelihood warping following:
        Rios and Torab (2019) Compositionally-warped Gaussian processes
        https://www.sciencedirect.com/science/article/pii/S0893608019301856

        All Warps must define the following methods::

            Warp.get_n_params() - return number of parameters
            Warp.f() - warping function (Non-Gaussian field -> Gaussian)
            Warp.invf() - inverse warp
            Warp.df() - derivatives
            Warp.warp_predictions() - compute predictive distribution

    """

    def __init__(self):
        self.n_params = np.nan

    def get_n_params(self):
        """Report the number of parameters required"""

        assert not np.isnan(self.n_params), "Warp function not initialised"

        return self.n_params

    def warp_predictions(self, mu, s2, param, percentiles=[0.025, 0.975]):
        """
        Compute the warped predictions from a gaussian predictive
            distribution, specifed by a mean (mu) and variance (s2)

            :param mu: Gassian predictive mean
            :param s2: Predictive variance
            :param param: warping parameters
            :param percentiles: Desired percentiles of the warped likelihood

            :returns: * median - median of the predictive distribution
                      * pred_interval - predictive interval(s)

        """

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
    def f(self, x, param):
        """Evaluate the warping function (mapping non-Gaussian respone
        variables to Gaussian variables)
        """

    @abstractmethod
    def invf(self, y, param):
        """Evaluate the warping function (mapping Gaussian latent variables
        to non-Gaussian response variables)
        """

    @abstractmethod
    def df(self, x, param):
        """Return the derivative of the warp, dw(x)/dx"""


class WarpLog(WarpBase):
    """Affine warp
    y = a + b*x
    """

    def __init__(self):
        self.n_params = 0

    def f(self, x, params=None):

        y = np.log(x)

        return y

    def invf(self, y, params=None):

        x = np.exp(y)

        return x

    def df(self, x, params):

        df = 1 / x

        return df


class WarpAffine(WarpBase):
    """Affine warp
    y = a + b*x
    """

    def __init__(self):
        self.n_params = 2

    def _get_params(self, param):
        if len(param) != self.n_params:
            raise ValueError("number of parameters must be " + str(self.n_params))
        return param[0], np.exp(param[1])

    def f(self, x, params):
        a, b = self._get_params(params)

        y = a + b * x
        return y

    def invf(self, y, params):
        a, b = self._get_params(params)

        x = (y - a) / b

        return x

    def df(self, x, params):
        a, b = self._get_params(params)

        df = np.ones(x.shape) * b
        return df


class WarpBoxCox(WarpBase):
    """Box cox transform having a single parameter (lambda), i.e.

    y = (sign(x) * abs(x) ** lamda - 1) / lambda

    This follows the generalization in Bicken and Doksum (1981) JASA 76
    and allows x to assume negative values.
    """

    def __init__(self):
        self.n_params = 1

    def _get_params(self, param):

        return np.exp(param)

    def f(self, x, params):
        lam = self._get_params(params)

        if lam == 0:
            y = np.log(x)
        else:
            y = (np.sign(x) * np.abs(x) ** lam - 1) / lam
        return y

    def invf(self, y, params):
        lam = self._get_params(params)

        if lam == 0:
            x = np.exp(y)
        else:
            x = np.sign(lam * y + 1) * np.abs(lam * y + 1) ** (1 / lam)

        return x

    def df(self, x, params):
        lam = self._get_params(params)

        dx = np.abs(x) ** (lam - 1)

        return dx


class WarpSinArcsinh(WarpBase):
    """Sin-hyperbolic arcsin warp having two parameters (a, b) and defined by

    y = sinh(b *  arcsinh(x) - a)

    Using the parametrisation of Rios et al, Neural Networks 118 (2017)
    where a controls skew and b controls kurtosis, such that:

    * a = 0 : symmetric
    * a > 0 : positive skew
    * a < 0 : negative skew
    * b = 1 : mesokurtic
    * b > 1 : leptokurtic
    * b < 1 : platykurtic

    where b > 0. However, it is more convenentent to use an alternative
    parameterisation, given in Jones and Pewsey 2019 JRSS Significance 16
    https://doi.org/10.1111/j.1740-9713.2019.01245.x

    where:

    y = sinh(b * arcsinh(x) + epsilon * b)

    and a = -epsilon*b

    see also Jones and Pewsey 2009 Biometrika, 96 (4) for more details
    about the SHASH distribution
    https://www.jstor.org/stable/27798865
    """

    def __init__(self):
        self.n_params = 2

    def _get_params(self, param):
        if len(param) != self.n_params:
            raise ValueError("number of parameters must be " + str(self.n_params))

        epsilon = param[0]
        b = np.exp(param[1])
        a = -epsilon * b

        return a, b

    def f(self, x, params):
        a, b = self._get_params(params)

        y = np.sinh(b * np.arcsinh(x) - a)
        return y

    def invf(self, y, params):
        a, b = self._get_params(params)

        x = np.sinh((np.arcsinh(y) + a) / b)

        return x

    def df(self, x, params):
        a, b = self._get_params(params)

        dx = (b * np.cosh(b * np.arcsinh(x) - a)) / np.sqrt(1 + x**2)

        return dx


class WarpCompose(WarpBase):
    """Composition of warps. These are passed in as an array and
    intialised automatically. For example::

        W = WarpCompose(('WarpBoxCox', 'WarpAffine'))

    where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, warpnames=None, debugwarp=False):

        if warpnames is None:
            raise ValueError("A list of warp functions is required")
        self.debugwarp = debugwarp
        self.warps = []
        self.n_params = 0
        for wname in warpnames:
            warp = eval(wname + "()")
            self.n_params += warp.get_n_params()
            self.warps.append(warp)

    def f(self, x, theta):
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

    def invf(self, x, theta):
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
                finvw = warp.invf(x, theta_c)
            else:
                finvw = warp.invf(finvw, theta_c)

        return finvw

    def df(self, x, theta):
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
