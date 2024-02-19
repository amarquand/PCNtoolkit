# from abc import ABC, ABCMeta, abstractmethod
# import numpy as np


# class CovBase(ABC):
#     """ Base class for covariance functions.

#         All covariance functions must define the following methods::

#             CovFunction.get_n_params()
#             CovFunction.cov()
#             CovFunction.xcov()
#             CovFunction.dcov()
#     """

#     def __init__(self, x=None):
#         self.n_params = np.nan

#     def get_n_params(self):
#         """ Report the number of parameters required """

#         assert not np.isnan(self.n_params), \
#             "Covariance function not initialised"

#         return self.n_params

#     @abstractmethod
#     def cov(self, theta, x, z=None):
#         """ Return the full covariance (or cross-covariance if z is given) """

#     @abstractmethod
#     def dcov(self, theta, x, i):
#         """ Return the derivative of the covariance function with respect to
#             the i-th hyperparameter """


# class CovLin(CovBase):
#     """ Linear covariance function (no hyperparameters)
#     """

#     def __init__(self, x=None):
#         self.n_params = 0
#         self.first_call = False

#     def cov(self, theta, x, z=None):
#         if not self.first_call and not theta and theta is not None:
#             self.first_call = True
#             if len(theta) > 0 and theta[0] is not None:
#                 print("CovLin: ignoring unnecessary hyperparameter ...")

#         if z is None:
#             z = x

#         K = x.dot(z.T)
#         return K

#     def dcov(self, theta, x, i):
#         raise ValueError("Invalid covariance function parameter")


# class CovSqExp(CovBase):
#     """ Ordinary squared exponential covariance function.
#         The hyperparameters are::

#             theta = ( log(ell), log(sf) )

#         where ell is a lengthscale parameter and sf2 is the signal variance
#     """

#     def __init__(self, x=None):
#         self.n_params = 2

#     def cov(self, theta, x, z=None):
#         self.ell = np.exp(theta[0])
#         self.sf2 = np.exp(2*theta[1])

#         if z is None:
#             z = x

#         R = squared_dist(x/self.ell, z/self.ell)
#         K = self.sf2 * np.exp(-R/2)
#         return K

#     def dcov(self, theta, x, i):
#         self.ell = np.exp(theta[0])
#         self.sf2 = np.exp(2*theta[1])

#         R = squared_dist(x/self.ell, x/self.ell)

#         if i == 0:   # return derivative of lengthscale parameter
#             dK = self.sf2 * np.exp(-R/2) * R
#             return dK
#         elif i == 1:   # return derivative of signal variance parameter
#             dK = 2*self.sf2 * np.exp(-R/2)
#             return dK
#         else:
#             raise ValueError("Invalid covariance function parameter")


# class CovSqExpARD(CovBase):
#     """ Squared exponential covariance function with ARD
#         The hyperparameters are::

#             theta = (log(ell_1, ..., log_ell_D), log(sf))

#         where ell_i are lengthscale parameters and sf2 is the signal variance
#     """

#     def __init__(self, x=None):
#         if x is None:
#             raise ValueError("N x D data matrix must be supplied as input")
#         if len(x.shape) == 1:
#             self.D = 1
#         else:
#             self.D = x.shape[1]
#         self.n_params = self.D + 1

#     def cov(self, theta, x, z=None):
#         self.ell = np.exp(theta[0:self.D])
#         self.sf2 = np.exp(2*theta[self.D])

#         if z is None:
#             z = x

#         R = squared_dist(x.dot(np.diag(1./self.ell)),
#                          z.dot(np.diag(1./self.ell)))
#         K = self.sf2*np.exp(-R/2)
#         return K

#     def dcov(self, theta, x, i):
#         K = self.cov(theta, x)
#         if i < self.D:    # return derivative of lengthscale parameter
#             dK = K * squared_dist(x[:, i]/self.ell[i], x[:, i]/self.ell[i])
#             return dK
#         elif i == self.D:   # return derivative of signal variance parameter
#             dK = 2*K
#             return dK
#         else:
#             raise ValueError("Invalid covariance function parameter")


# class CovSum(CovBase):
#     """ Sum of covariance functions. These are passed in as a cell array and
#         intialised automatically. For example::

#             C = CovSum(x,(CovLin, CovSqExpARD))
#             C = CovSum.cov(x, )

#         The hyperparameters are::

#             theta = ( log(ell_1, ..., log_ell_D), log(sf2) )

#         where ell_i are lengthscale parameters and sf2 is the signal variance
#     """

#     def __init__(self, x=None, covfuncnames=None):
#         if x is None:
#             raise ValueError("N x D data matrix must be supplied as input")
#         if covfuncnames is None:
#             raise ValueError("A list of covariance functions is required")
#         self.covfuncs = []
#         self.n_params = 0
#         for cname in covfuncnames:
#             covfunc = eval(cname + '(x)')
#             self.n_params += covfunc.get_n_params()
#             self.covfuncs.append(covfunc)

#         if len(x.shape) == 1:
#             self.N = len(x)
#             self.D = 1
#         else:
#             self.N, self.D = x.shape

#     def cov(self, theta, x, z=None):
#         theta_offset = 0
#         for ci, covfunc in enumerate(self.covfuncs):
#             try:
#                 n_params_c = covfunc.get_n_params()
#                 theta_c = [theta[c] for c in
#                            range(theta_offset, theta_offset + n_params_c)]
#                 theta_offset += n_params_c
#             except Exception as e:
#                 print(e)

#             if ci == 0:
#                 K = covfunc.cov(theta_c, x, z)
#             else:
#                 K += covfunc.cov(theta_c, x, z)
#         return K

#     def dcov(self, theta, x, i):
#         theta_offset = 0
#         for covfunc in self.covfuncs:
#             n_params_c = covfunc.get_n_params()
#             theta_c = [theta[c] for c in
#                        range(theta_offset, theta_offset + n_params_c)]
#             theta_offset += n_params_c

#             if theta_c:  # does the variable have any hyperparameters?
#                 if 'dK' not in locals():
#                     dK = covfunc.dcov(theta_c, x, i)
#                 else:
#                     dK += covfunc.dcov(theta_c, x, i)
#         return dK


# def squared_dist(x, z=None):
#     """
#     Compute sum((x-z) ** 2) for all vectors in a 2d array.

#     """

#     # do some basic checks
#     if z is None:
#         z = x
#     if len(x.shape) == 1:
#         x = x[:, np.newaxis]
#     if len(z.shape) == 1:
#         z = z[:, np.newaxis]

#     nx, dx = x.shape
#     nz, dz = z.shape
#     if dx != dz:
#         raise ValueError("""
#                 Cannot compute distance: vectors have different length""")

#     # mean centre for numerical stability
#     m = np.mean(np.vstack((np.mean(x, axis=0), np.mean(z, axis=0))), axis=0)
#     x = x - m
#     z = z - m

#     xx = np.tile(np.sum((x*x), axis=1)[:, np.newaxis], (1, nz))
#     zz = np.tile(np.sum((z*z), axis=1), (nx, 1))

#     dist = (xx - 2*x.dot(z.T) + zz)

#     return dist
