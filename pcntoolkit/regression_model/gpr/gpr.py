# import numpy as np

# from pcntoolkit.regression_model.gpr.gpr_conf import GPRConf
# from pcntoolkit.regression_model.regression_model import RegressionModel

# # from cov import CovBase


# class GPR(RegressionModel):

#     def __init__(
#         self, name: str, reg_conf: GPRConf, is_fitted=False, is_from_dict=False
#     ):
#         super().__init__(name, reg_conf, is_fitted, is_from_dict)
#         self.hyp = np.nan  # hyperparameters

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         """
#         Fits the model.
#         """
#         # some fitting logic
#         # ...
#         raise NotImplementedError(
#             f"Fit method not implemented for {self.__class__.__name__}"
#         )

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Predicts on new data.
#         """
#         # some prediction logic
#         # ...
#         raise NotImplementedError(
#             f"Predict method not implemented for {self.__class__.__name__}"
#         )

#     def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test) -> np.ndarray:
#         """
#         Fits and predicts the model.
#         """
#         """ Function to make predictions from the model
#         """
#         if len(hyp.shape) > 1:  # force 1d hyperparameter array
#             hyp = hyp.flatten()

#         # ensure X and Xs are multi-dimensional arrays
#         if len(Xs.shape) == 1:
#             Xs = Xs[:, np.newaxis]
#         if len(X.shape) == 1:
#             X = X[:, np.newaxis]

#         # parameters for warping the likelhood function
#         if self.warp is not None:
#             gamma = hyp[1 : (self.n_warp_param + 1)]
#             y = self.warp.f(y, gamma)

#         # reestimate posterior (avoids numerical problems with optimizer)
#         self.post(hyp, self.covfunc, X, y)

#         # hyperparameters
#         sn2 = np.exp(2 * hyp[0])  # noise variance
#         # (generic) covariance hyperparameters
#         theta = hyp[(self.n_warp_param + 1) :]

#         Ks = self.covfunc.cov(theta, Xs, X)
#         kss = self.covfunc.cov(theta, Xs)

#         # predictive mean
#         ymu = Ks.dot(self.alpha)

#         # predictive variance (for a noisy test input)
#         # v = solve(self.L, Ks.T)
#         v = 1
#         ys2 = kss - v.T.dot(v) + sn2

#         return ymu, ys2
#         # ...
#         raise NotImplementedError(
#             f"Fit-predict method not implemented for {self.__class__.__name__}"
#         )

#     @classmethod
#     def from_dict(cls, dict):
#         """
#         Creates a configuration from a dictionary.
#         """
#         name = dict["name"]
#         conf = GPRConf.from_dict(dict["reg_conf"])
#         is_fitted = dict["is_fitted"]
#         is_from_dict = True
#         self = cls(name, conf, is_fitted, is_from_dict)
#         return self

#     @classmethod
#     def from_args(cls, name, args):
#         """
#         Creates a configuration from command line arguments
#         """
#         conf = GPRConf.from_args(args)
#         is_fitted = args.get("is_fitted", False)
#         is_from_dict = True
#         self = cls(name, conf, is_fitted, is_from_dict)
#         return self
