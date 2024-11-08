"""
Bayesian Linear Regression (BLR) implementation.

This module implements Bayesian Linear Regression with support for:
- L1/L2 regularization
- Automatic Relevance Determination (ARD)
- Heteroskedastic noise modeling
- Multiple optimization methods (CG, Powell, Nelder-Mead, L-BFGS-B)

The implementation follows standard Bayesian formulation with Gaussian priors
and supports both homoskedastic and heteroskedastic noise models.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from scipy import linalg, optimize, stats  # type: ignore
from scipy.linalg import LinAlgError  # type: ignore

from pcntoolkit.regression_model.blr.blr_data import BLRData
from pcntoolkit.regression_model.regression_model import RegressionModel

from .blr_conf import BLRConf


class BLR(RegressionModel):
    """
    Bayesian Linear Regression model implementation.

    This class implements Bayesian Linear Regression with various features including
    automatic relevance determination (ARD), heteroskedastic noise modeling, and
    multiple optimization methods.

    Attributes
    ----------
    hyp : np.ndarray
        Model hyperparameters
    nlZ : float
        Negative log marginal likelihood
    N : int
        Number of samples
    D : int
        Number of features
    lambda_n_vec : np.ndarray
        Precision matrix
    Sigma_a : np.ndarray
        Prior covariance
    Lambda_a : np.ndarray
        Prior precision
    warp : bool
        Whether to use warping
    hyp0 : np.ndarray
        Initial hyperparameters
    n_hyp : int
        Number of hyperparameters
    """

    @property
    def blr_conf(self) -> BLRConf:
        """Rewturn the configuration object for the BLR model.

        Returns
        -------
        BLRConf
            BLRConf
        """
        return cast(BLRConf, self.reg_conf)

    def __init__(
        self,
        name: str,
        reg_conf: BLRConf,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ) -> None:
        """
        Initialize the BLR model.

        Parameters
        ----------
        name : str
            Model name identifier
        reg_conf : BLRConf
            Model configuration object
        is_fitted : bool, optional
            Whether the model is already fitted, by default False
        is_from_dict : bool, optional
            Whether the model is being loaded from dictionary, by default False
        """
        super().__init__(name, reg_conf, is_fitted, is_from_dict)

        self.hyp: np.ndarray = None  # type: ignore
        self.nlZ: np.ndarray = np.nan  # type: ignore
        self.N: int = None  # type: ignore  # Number of samples
        self.D: int = None  # type: ignore # Number of features
        self.lambda_n_vec: np.ndarray = None  # type: ignore  # precision matrix
        self.Sigma_a: np.ndarray = None  # type: ignore  # prior covariance
        self.Lambda_a: np.ndarray = None  # type: ignore # prior precision
        self.warp: bool = None  # type: ignore
        self.hyp0: np.ndarray = None  # type: ignore
        self.n_hyp: int = 0
        self.var_D: int = 0
        self.alpha: np.ndarray = None  # type: ignore
        self.beta: np.ndarray = None  # type: ignore
        self.gamma: np.ndarray = None  # type: ignore
        self.m: np.ndarray = None  # type: ignore
        self.A: np.ndarray = None  # type: ignore
        self.dnlZ: np.ndarray = None  # type: ignore
        # ? Do we need ys and s2?
        self.ys: np.ndarray = None  # type: ignore
        self.s2: np.ndarray = None  # type: ignore

        # self.gamma = None # Not used if warp is not used

    def init_hyp(self, data: BLRData) -> np.ndarray:  # type:ignore
        """
        Initialize model hyperparameters.

        Parameters
        ----------
        data : BLRData
            Training data containing features and targets

        Returns
        -------
        np.ndarray
            Initialized hyperparameter vector
        """
        # TODO check if this is correct
        # Model order
        if self.hyp0:
            return self.hyp0

        if self.models_variance:
            n_beta = self.var_D
        else:
            n_beta = 1

        n_alpha = self.D
        n_gamma = 0
        self.n_hyp = n_beta + n_alpha + n_gamma  # type: ignore
        return np.zeros(self.n_hyp)

    def fit(self, data: BLRData) -> None:
        """
        Fit the Bayesian Linear Regression model to the data.

        Parameters
        ----------
        data : BLRData
            Data object containing features and target.
        """
        self.D = data.X.shape[1]
        self.var_D = data.var_X.shape[1]

        # Initialize hyperparameters if not provided
        hyp0 = self.init_hyp(data)

        args = (data.X, data.y, data.var_X)

        match self.blr_conf.optimizer.lower():
            case "cg":
                out = optimize.fmin_cg(
                    f=self.loglik,
                    x0=hyp0,
                    fprime=self.dloglik,
                    args=args,
                    gtol=self.tol,
                    maxiter=self.n_iter,
                    full_output=1,
                )
            case "powell":
                out = optimize.fmin_powell(
                    func=self.loglik, x0=hyp0, args=args, full_output=1
                )
            case "nelder-mead":
                out = optimize.fmin(func=self.loglik, x0=hyp0, args=args, full_output=1)
            case "l-bfgs-b":
                all_hyp_i = [hyp0]

                def store(X: np.ndarray) -> None:
                    hyp = X
                    all_hyp_i.append(hyp)

                try:
                    out = optimize.fmin_l_bfgs_b(
                        func=self.penalized_loglik,
                        x0=hyp0,
                        args=(*args, self.l_bfgs_b_l, self.norm),
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
                        args=(*args, self.l_bfgs_b_l, self.norm),
                        approx_grad=True,
                        epsilon=self.epsilon,
                    )

            case _:
                raise ValueError(f"Optimizer {self.blr_conf.optimizer} not recognized.")
        self.hyp = out[0]
        self.nlZ = out[1]
        _, self.beta = self.parse_hyps(self.hyp, data.X, data.var_X)
        self.is_fitted = True

    def predict(self, data: BLRData) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        data : BLRData
            Data object containing features for prediction.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Predictive mean and variance.
        """
        _, beta = self.parse_hyps(self.hyp, data.X, data.var_X)
        ys = data.X.dot(self.m)
        s2n = 1 / beta
        s2 = s2n + np.sum(data.X * linalg.solve(self.A, data.X.T).T, axis=1)
        # ! These need to be stored for the centiles and zscores methods
        self.ys = ys
        self.s2 = s2

        return ys, s2

    def parse_hyps(
        self, hyp: np.ndarray, X: np.ndarray, var_X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse hyperparameters into model parameters.

        Parameters
        ----------
        hyp : np.ndarray
            Hyperparameter vector.
        X : np.ndarray
            Covariates.
        var_X : np.ndarray
            Variance of covariates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Parsed alpha and beta parameters.
        """
        N = X.shape[0]
        beta: np.ndarray = None  # type: ignore
        # Noise precision
        if self.models_variance:
            Dv = var_X.shape[1]
            w_d = np.asarray(hyp[0:Dv])
            beta = np.exp(var_X.dot(w_d))
            n_lik_param = len(w_d)
            self.lambda_n_vec = beta
        else:
            beta = np.asarray([np.exp(hyp[0])])
            n_lik_param = len(beta)
            self.lambda_n_vec = np.ones(N) * beta

        # Coefficients precision
        if isinstance(beta, list) or isinstance(beta, np.ndarray):
            alpha = np.exp(hyp[n_lik_param:])
        else:
            alpha = np.exp(hyp[1:])

        return alpha, beta

    def post(
        self, hyp: np.ndarray, X: np.ndarray, y: np.ndarray, var_X: np.ndarray
    ) -> None:
        """
        Compute the posterior distribution.

        Parameters
        ----------
        hyp : np.ndarray
            Hyperparameter vector.
        X : np.ndarray
            Covariates.
        y : np.ndarray
            Responses.
        var_X : np.ndarray
            Variance of covariates.
        """
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
        alpha, _ = self.parse_hyps(self.hyp, X, var_X)

        # prior variance
        if len(alpha) == 1 or len(alpha) == self.D:
            self.Sigma_a = np.diag(np.ones(self.D)) / alpha
            self.Lambda_a = np.diag(np.ones(self.D)) * alpha
        else:
            raise ValueError("hyperparameter vector has invalid length")

        # Compute the posterior precision and mean
        XtLambda_n = X.T * self.lambda_n_vec
        self.A = XtLambda_n.dot(X) + self.Lambda_a
        invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
        self.m = (invAXt * self.lambda_n_vec).dot(y)

    def loglik(
        self, hyp: np.ndarray, X: np.ndarray, y: np.ndarray, var_X: np.ndarray
    ) -> float:
        """
        Compute the negative log likelihood.

        Parameters
        ----------
        hyp : np.ndarray
            Hyperparameter vector.
        X : np.ndarray
            Covariates.
        y : np.ndarray
            Responses.
        var_X : np.ndarray
            Variance of covariates.

        Returns
        -------
        float
            Negative log likelihood.
        """
        _, _ = self.parse_hyps(hyp, X, var_X)

        something_big: float = float(np.finfo(np.float64).max)

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not hasattr(self, "A"):
            try:
                self.post(hyp, X, y, var_X)
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

        self.nlZ = nlZ  # type: ignore
        return nlZ

    def penalized_loglik(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        var_X: np.ndarray,
        regularizer_strength: float = 0.1,
        norm: Literal["L1", "L2"] = "L1",
    ) -> float:
        """
        Compute the penalized log likelihood with L1 or L2 regularization.

        Parameters
        ----------
        hyp : np.ndarray
            Hyperparameter vector
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        var_X : np.ndarray
            Variance of features
        regularizer_strength : float, optional
            Regularization strength, by default 0.1
        norm : {"L1", "L2"}, optional
            Type of regularization norm, by default "L1"

        Returns
        -------
        float
            Penalized negative log likelihood value

        Raises
        ------
        ValueError
            If norm is not "L1" or "L2"
        """
        if norm.upper() == "L1":
            return self.loglik(hyp, X, y, var_X) + regularizer_strength * np.sum(
                np.abs(hyp)
            )
        elif norm.upper() == "L2":
            return self.loglik(hyp, X, y, var_X) + regularizer_strength * np.sum(
                np.square(hyp)
            )
        else:
            raise ValueError(
                "Requested penalty not recognized, choose between 'L1' or 'L2'."
            )

    def dloglik(
        self, hyp: np.ndarray, X: np.ndarray, y: np.ndarray, var_X: np.ndarray
    ) -> np.ndarray:
        """Function to compute derivatives"""

        # hyperparameters
        alpha, beta = self.parse_hyps(hyp, X, var_X)

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not hasattr(self, "A"):
            try:
                self.post(hyp, X, y, var_X)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                if self.dnlZ is not None:
                    dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                    return dnlZ
                return np.array(1 / np.finfo(float).eps)
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
        # dnl2 = np.zeros(hyp.shape)

        # noise precision parameter(s)
        for i, _ in enumerate(beta):
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
        for i, _ in enumerate(beta):
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
                dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps  # type: ignore

        self.dnlZ = dnlZ
        return dnlZ

    def centiles(
        self, data: BLRData, cdf: np.ndarray, resample: bool = True
    ) -> np.ndarray:
        """Calculate prediction centiles for given cumulative distribution function values.

        Parameters
        ----------
        data : BLRData
            Data object containing features for prediction
        cdf : np.ndarray
            Array of cumulative distribution function values to compute centiles for
        resample : bool, optional
            Whether to recompute predictions before calculating centiles, by default True

        Returns
        -------
        np.ndarray
            Array of shape (len(cdf), n_samples) containing the predicted centile 
            values for each CDF value and sample
        """
        if resample:
            self.predict(data)
        centiles = np.zeros((cdf.shape[0], data.X.shape[0]))
        for i, cdf in enumerate(cdf):
            centiles[i, :] = self.ys + stats.norm.ppf(cdf) * np.sqrt(self.s2)
        return centiles

    def zscores(self, data: BLRData, resample: bool = True) -> np.ndarray:
        """Calculate z-scores for observed values relative to predictions.

        Parameters
        ----------
        data : BLRData
            Data object containing features and observed values
        resample : bool, optional
            Whether to recompute predictions before calculating z-scores, by default True

        Returns
        -------
        np.ndarray
            Array of z-scores for each observation
        """
        if resample:
            self.predict(data)
        return (data.y - self.ys) / np.sqrt(self.s2)

    def to_dict(self, path: str | None = None) -> dict:
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
    def from_dict(cls, my_dict: dict, path: str | None = None) -> "BLR":
        """
        Creates a configuration from a dictionary.
        """
        name = my_dict["name"]
        conf = BLRConf.from_dict(my_dict["reg_conf"])
        is_fitted = my_dict["is_fitted"]
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        self.hyp = np.array(my_dict["hyp"])
        self.nlZ = my_dict["nlZ"]
        self.N = my_dict["N"]
        self.D = my_dict["D"]
        self.lambda_n_vec = np.array(my_dict["lambda_n_vec"])
        self.Sigma_a = np.array(my_dict["Sigma_a"])
        self.Lambda_a = np.array(my_dict["Lambda_a"])
        self.beta = np.array(my_dict["beta"])
        self.m = np.array(my_dict["m"])
        self.A = np.array(my_dict["A"])
        return self

    @classmethod
    def from_args(cls, name: str, args: dict) -> "BLR":
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
    def tol(self) -> float:
        """Optimization convergence tolerance.

        Returns
        -------
        float
            Tolerance value for optimization convergence
        """
        return self.blr_conf.tol

    @property
    def n_iter(self) -> int:
        """Maximum number of optimization iterations.

        Returns
        -------
        int
            Maximum number of iterations for optimization
        """
        return self.blr_conf.n_iter

    @property
    def optimizer(self) -> str:
        """Optimization method to use.

        Returns
        -------
        str
            Name of optimization method ('cg', 'powell', 'nelder-mead', or 'l-bfgs-b')
        """
        return self.blr_conf.optimizer

    @property
    def ard(self) -> bool:
        """Whether to use Automatic Relevance Determination.

        Returns
        -------
        bool
            True if using ARD, False otherwise
        """
        return self.blr_conf.ard

    @property
    def l_bfgs_b_l(self) -> float:
        """L-BFGS-B regularization strength.

        Returns
        -------
        float
            Regularization strength parameter for L-BFGS-B optimizer
        """
        return self.blr_conf.l_bfgs_b_l

    @property
    def epsilon(self) -> float:
        """Step size for gradient approximation in L-BFGS-B.

        Returns
        -------
        float
            Step size for finite difference gradient approximation
        """
        return self.blr_conf.l_bfgs_b_epsilon

    @property
    def norm(self) -> str:
        """Type of regularization norm for L-BFGS-B.

        Returns
        -------
        str
            Regularization norm type ('L1' or 'L2')
        """
        return self.blr_conf.l_bfgs_b_norm

    @property
    def intercept(self) -> bool:
        """Whether to include an intercept term.

        Returns
        -------
        bool
            True if model includes intercept, False otherwise
        """
        return self.blr_conf.intercept

    @property
    def random_intercept(self) -> bool:
        """Whether to include a random intercept.

        Returns
        -------
        bool
            True if model includes random intercept, False otherwise
        """
        return self.blr_conf.random_intercept

    @property
    def heteroskedastic(self) -> bool:
        """Whether to model heteroskedastic noise.

        Returns
        -------
        bool
            True if modeling heteroskedastic noise, False otherwise
        """
        return self.blr_conf.heteroskedastic

    @property
    def random_intercept_var(self) -> bool:
        """Whether to model random intercept variance.

        Returns
        -------
        bool
            True if modeling random intercept variance, False otherwise
        """
        return self.blr_conf.random_intercept_var

    @property
    def intercept_var(self) -> bool:
        """Whether to model intercept variance.

        Returns
        -------
        bool
            True if modeling intercept variance, False otherwise
        """
        return self.blr_conf.intercept_var

    @property
    def models_variance(self) -> bool:
        """Whether the model includes any variance components.

        Returns
        -------
        bool
            True if model includes heteroskedastic noise, random intercept variance,
            or intercept variance components
        """
        return (
            self.blr_conf.heteroskedastic
            or self.blr_conf.random_intercept_var
            or self.blr_conf.intercept_var
        )
