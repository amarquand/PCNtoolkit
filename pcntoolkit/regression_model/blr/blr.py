from __future__ import annotations

import numpy as np
from scipy import linalg, optimize, stats
from scipy.linalg import LinAlgError

from pcntoolkit.regression_model.blr.blr_data import BLRData
from pcntoolkit.regression_model.regression_model import RegressionModel

from .blr_conf import BLRConf


class BLR(RegressionModel):
    def __init__(
        self, name: str, reg_conf: BLRConf, is_fitted=False, is_from_dict=False
    ):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        super().__init__(name, reg_conf, is_fitted, is_from_dict)

        self.hyp = None
        self.nlZ = np.nan
        self.N = None  # Number of samples
        self.D = None  # Number of features
        self.lambda_n_vec = None  # precision matrix
        self.Sigma_a = None  # prior covariance
        self.Lambda_a = None  # prior precision
        self.warp = None
        # self.gamma = None # Not used if warp is not used

    def fit(self, data: BLRData, hyp0=None):
        if hyp0 is None:
            if self.ard:
                hyp0 = np.zeros(data.X.shape[1] + 1)
            else:
                hyp0 = np.zeros(2)

        match self.reg_conf.optimizer.lower():
            case "cg":
                out = optimize.fmin_cg(
                    f=self.loglik,
                    x0=hyp0,
                    fprime=self.dloglik,
                    args=(data.X, data.y),
                    gtol=self.tol,
                    maxiter=self.n_iter,
                    full_output=1,
                )
            case "powell":
                out = optimize.fmin_powell(
                    func=self.loglik, x0=hyp0, args=(data.X, data.y), full_output=1
                )
            case "nelder-mead":
                out = optimize.fmin(
                    func=self.loglik, x0=hyp0, args=(data.X, data.y), full_output=1
                )
            case "l-bfgs-b":
                all_hyp_i = [hyp0]

                def store(X):
                    hyp = X
                    all_hyp_i.append(hyp)

                try:
                    out = optimize.fmin_l_bfgs_b(
                        func=self.penalized_loglik,
                        x0=hyp0,
                        args=(data.X, data.y, self.l, self.norm),
                        approx_grad=True,
                        epsilon=self.epsilon,
                        callback=store,
                    )
                except np.linalg.LinAlgError as e:
                    print(
                        f"Restarting estimation at hyp = {all_hyp_i[-1]}, due to *** numpy.linalg.LinAlgError: Matrix is singular.\n{e}"
                    )
                    out = optimize.fmin_l_bfgs_b(
                        func=self.penalized_loglik,
                        x0=all_hyp_i[-1],
                        args=(data.X, data.y, self.l, self.norm),
                        approx_grad=True,
                        epsilon=self.epsilon,
                    )

            case _:
                raise ValueError(f"Optimizer {self.reg_conf.optimizer} not recognized.")
        self.hyp = out[0]
        self.nlZ = out[1]
        self.beta, _ = self.parse_hyps(self.hyp, data.X)
        self.is_fitted = True

    def predict(self, data: BLRData):
        """Function to make predictions from the model
        :param X: covariates for test data
        This always returns Gaussian predictions, i.e.

        :returns: * ys - predictive mean
                  * s2 - predictive variance
        """
        beta = self.beta
        self.ys = data.X.dot(self.m)
        s2n = 1 / beta
        self.s2 = s2n + np.sum(data.X * linalg.solve(self.A, data.X.T).T, axis=1)
        return self.ys, self.s2

    def parse_hyps(self, hyp, X):
        N = X.shape[0]

        # Precision for the noise
        beta = np.asarray([np.exp(hyp[0])])
        n_lik_param = len(beta)

        # Precision for the coefficients
        if isinstance(beta, list) or type(beta) is np.ndarray:
            alpha = np.exp(hyp[n_lik_param:])
        else:
            alpha = np.exp(hyp[1:])

        # Precision matrix
        self.lambda_n_vec = np.ones(N) * beta

        return alpha, beta

    def post(self, hyp, X, y):
        # Store the number of samples and features
        self.N = X.shape[0]
        if len(X.shape) == 1:
            self.D = 1
        else:
            self.D = X.shape[1]

        # Check if hyperparameters have changed
        if (hyp == self.hyp).all() and hasattr(self, "N"):
            print("hyperparameters have not changed, exiting")
            return
        else:
            self.hyp = hyp

        # Parse hyperparameters
        alpha, beta = self.parse_hyps(self.hyp, X)

        # prior variance
        if len(alpha) == 1 or len(alpha) == self.D:
            self.Sigma_a = np.diag(np.ones(self.D)) / alpha
            self.Lambda_a = np.diag(np.ones(self.D)) * alpha
        else:
            raise ValueError("hyperparameter vector has invalid length")

        # Compute the posterior precision and mean
        XtLambda_n = X.T * self.lambda_n_vec
        self.A = XtLambda_n.dot(X) + self.Lambda_a
        invAXt = linalg.solve(self.A, X.T, check_finite=False)
        self.m = (invAXt * self.lambda_n_vec).dot(y)

    def loglik(self, hyp, X, y):
        alpha, beta = self.parse_hyps(hyp, X)

        something_big = 1 / np.finfo(float).eps

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not (hasattr(self, "A")):
            try:
                self.post(hyp, X, y)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                nlZ = something_big
                return nlZ

        try:
            # compute the log determinants in a numerically stable way
            logdetA = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(self.A))))
        except (ValueError, LinAlgError):
            print("Warning: Estimation of posterior distribution failed")
            nlZ = something_big
            return nlZ

        logdetSigma_a = np.sum(np.log(np.diag(self.Sigma_a)))  # diagonal
        logdetSigma_n = -np.sum(np.log(self.lambda_n_vec))

        # compute negative marginal log likelihood
        X_y_t_sLambda_n = (y - X.dot(self.m)) * np.sqrt(self.lambda_n_vec)
        nlZ = -0.5 * (
            -self.N * np.log(2 * np.pi)
            - logdetSigma_n
            - logdetSigma_a
            - X_y_t_sLambda_n.T.dot(X_y_t_sLambda_n)
            - self.m.T.dot(self.Lambda_a).dot(self.m)
            - logdetA
        )

        # make sure the output is finite to stop the minimizer getting upset
        if not np.isfinite(nlZ):
            nlZ = something_big

        self.nlZ = nlZ
        return nlZ

    def penalized_loglik(self, hyp, X, y, l=0.1, norm="L1"):
        """Function to compute the penalized log (marginal) likelihood

        :param hyp: hyperparameter vector
        :param X: covariates
        :param y: responses
        :param l: regularisation penalty
        :param norm: type of regulariser (L1 or L2)
        """

        if norm.lower() == "l1":
            L = self.loglik(hyp, X, y) + l * sum(abs(hyp))
        elif norm.lower() == "l2":
            L = self.loglik(hyp, X, y) + l * sum(np.sqrt(hyp**2))
        else:
            print("Requested penalty not recognized, choose between 'L1' or 'L2'.")
        return L

    def dloglik(self, hyp, X, y):
        """Function to compute derivatives"""

        # hyperparameters
        beta, alpha = self.parse_hyps(hyp, X)

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not (hasattr(self, "A")):
            try:
                self.post(hyp, X, y)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ

        # precompute re-used quantities to maximise speed
        # todo: revise implementation to use Cholesky throughout
        #       that would remove the need to explicitly compute the inverse
        S = np.linalg.inv(self.A)  # posterior covariance
        SX = S.dot(X.T)
        XLn = X.T * self.lambda_n_vec  # = X.T.dot(self.Lambda_n)
        XLny = XLn.dot(y)
        SXLny = S.dot(XLny)
        XLnXm = XLn.dot(X).dot(self.m)

        # initialise derivatives
        dnlZ = np.zeros(hyp.shape)
        dnl2 = np.zeros(hyp.shape)

        # noise precision parameter(s)
        for i in range(0, len(beta)):
            # first compute derivative of Lambda_n with respect to beta
            dL_n_vec = np.ones(self.N)
            dLambda_n = np.diag(dL_n_vec)

            # compute quantities used multiple times
            XdLnX = X.T.dot(dLambda_n).dot(X)
            dA = XdLnX

            # derivative of posterior parameters with respect to beta
            b = -S.dot(dA).dot(SXLny) + SX.dot(dLambda_n).dot(y)

            # compute np.trace(self.Sigma_n.dot(dLambda_n)) efficiently
            trSigma_ndLambda_n = sum((1 / self.lambda_n_vec) * np.diag(dLambda_n))

            # compute  y.T.dot(Lambda_n) efficiently
            ytLn = (y * self.lambda_n_vec).T

            # compute derivatives
            dnlZ[i] = (
                -(
                    0.5 * trSigma_ndLambda_n
                    - 0.5 * y.dot(dLambda_n).dot(y)
                    + y.dot(dLambda_n).dot(X).dot(self.m)
                    + ytLn.dot(X).dot(b)
                    - 0.5 * self.m.T.dot(XdLnX).dot(self.m)
                    - b.T.dot(XLnXm)
                    - b.T.dot(self.Lambda_a).dot(self.m)
                    - 0.5 * np.trace(S.dot(dA))
                )
                * beta[i]
            )

        # scaling parameter(s)
        for i in range(0, len(alpha)):
            # first compute derivatives with respect to alpha
            if len(alpha) == self.D:  # are we using ARD?
                dLambda_a = np.zeros((self.D, self.D))
                dLambda_a[i, i] = 1
            else:
                dLambda_a = np.eye(self.D)

            F = dLambda_a
            c = -S.dot(F).dot(SXLny)

            # compute np.trace(self.Sigma_a.dot(dLambda_a)) efficiently
            trSigma_adLambda_a = sum(np.diag(self.Sigma_a) * np.diag(dLambda_a))

            dnlZ[i + len(beta)] = (
                -(
                    0.5 * trSigma_adLambda_a
                    + XLny.T.dot(c)
                    - c.T.dot(XLnXm)
                    - c.T.dot(self.Lambda_a).dot(self.m)
                    - 0.5 * self.m.T.dot(F).dot(self.m)
                    - 0.5 * np.trace(linalg.solve(self.A, F))
                )
                * alpha[i]
            )

        # make sure the gradient is finite to stop the minimizer getting upset
        if not all(np.isfinite(dnlZ)):
            bad = np.where(np.logical_not(np.isfinite(dnlZ)))
            for b in bad:
                dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps

        self.dnlZ = dnlZ
        return dnlZ

    def centiles(self, data: BLRData, cummulative_densities, resample=True):
        if resample:
            self.predict(data)
        centiles = np.zeros((len(cummulative_densities), data.X.shape[0]))
        for i, cdf in enumerate(cummulative_densities):
            centiles[i, :] = self.ys + stats.norm.ppf(cdf) * np.sqrt(self.s2)
        return centiles

    def zscores(self, data: BLRData, resample=True):
        if resample:
            self.predict(data)
        return (data.y - self.ys) / np.sqrt(self.s2)

    def to_dict(self, path=None):
        my_dict = super().to_dict()
        my_dict["hyp"] = self.hyp.tolist()
        my_dict["nlZ"] = self.nlZ
        my_dict["N"] = self.N
        my_dict["D"] = self.D
        my_dict["lambda_n_vec"] = self.lambda_n_vec.tolist()
        my_dict["Sigma_a"] = self.Sigma_a.tolist()
        my_dict["Lambda_a"] = self.Lambda_a.tolist()
        my_dict["beta"] = self.beta.tolist()
        my_dict["m"] = self.m.tolist()
        my_dict["A"] = self.A.tolist()
        return my_dict

    @classmethod
    def from_dict(cls, dict, path=None):
        """
        Creates a configuration from a dictionary.
        """
        name = dict["name"]
        conf = BLRConf.from_dict(dict["reg_conf"])
        is_fitted = dict["is_fitted"]
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        self.hyp = np.array(dict["hyp"])
        self.nlZ = dict["nlZ"]
        self.N = dict["N"]
        self.D = dict["D"]
        self.lambda_n_vec = np.array(dict["lambda_n_vec"])
        self.Sigma_a = np.array(dict["Sigma_a"])
        self.Lambda_a = np.array(dict["Lambda_a"])
        self.beta = np.array(dict["beta"])
        self.m = np.array(dict["m"])
        self.A = np.array(dict["A"])
        return self

    @classmethod
    def from_args(cls, name, args):
        """
        Creates a configuration from command line arguments
        """
        conf = BLRConf.from_args(args)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        self.hyp = np.array(args.get("hyp", None))
        self.nlZ = args.get("nlZ", None)
        self.N = args.get("N", None)
        self.D = args.get("D", None)
        self.lambda_n_vec = np.array(args.get("lambda_n_vec", None))
        self.Sigma_a = np.array(args.get("Sigma_a", None))
        self.Lambda_a = np.array(args.get("Lambda_a", None))
        self.beta = np.array(args.get("beta", None))
        self.m = np.array(args.get("m", None))
        self.A = np.array(args.get("A", None))
        return self

    @property
    def tol(self):
        return self.reg_conf.tol

    @property
    def n_iter(self):
        return self.reg_conf.n_iter

    @property
    def optimizer(self):
        return self.reg_conf.optimizer

    @property
    def ard(self):
        return self.reg_conf.ard

    @property
    def l(self):
        return self.reg_conf.l_bfgs_b_l

    @property
    def epsilon(self):
        return self.reg_conf.l_bfgs_b_epsilon

    @property
    def norm(self):
        return self.reg_conf.l_bfgs_b_norm
