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

import copy
from typing import Literal, Optional

import numpy as np
import xarray as xr
from scipy import linalg, optimize  # type: ignore
from scipy.linalg import LinAlgError  # type: ignore

from pcntoolkit.math_functions.basis_function import BasisFunction, create_basis_function
from pcntoolkit.math_functions.warp import *
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.output import Errors, Messages, Output, Warnings


class BLR(RegressionModel):
    """
    Bayesian Linear Regression model implementation.

    This class implements Bayesian Linear Regression with various features including
    automatic relevance determination (ARD), heteroskedastic noise modeling, and
    multiple optimization methods.
    """

    def __init__(
        self,
        name: str = "template",
        fixed_effect: bool = False,
        heteroskedastic: bool = False,
        fixed_effect_var: bool = False,
        warp_name: Optional[str] = None,
        warp_reparam: bool = False,
        basis_function_mean: BasisFunction = None,  # type: ignore
        basis_function_var: BasisFunction = None,  # type: ignore
        n_iter: int = 300,
        tol: float = 1e-5,
        ard: bool = False,
        optimizer: str = "l-bfgs-b",
        l_bfgs_b_l: float = 0.1,
        l_bfgs_b_epsilon: float = 0.1,
        l_bfgs_b_norm: str = "l2",
        hyp0: np.ndarray | None = None,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ) -> None:
        """
        This class implements Bayesian Linear Regression with various features including
        automatic relevance determination (ARD), heteroskedastic noise modeling, and
        multiple optimization methods.

        Parameters
        ----------
        name : str
            Unique identifier for the model instance
        fixed_effect : bool, optional
            Whether to model a fixed effect in the mean, by default False
        heteroskedastic : bool, optional
            Whether to use heteroskedastic noise modeling, by default False
        fixed_effect_var : bool, optional
            Whether to model a fixed effect in the variance, by default False
        warp_name : str, optional
            Name of the warp function to use, by default None
        warp_reparam : bool, optional
            Whether to use a reparameterized warp function, by default False
        basis_function_mean : BasisFunction, optional
            Basis function for the mean, by default None
        basis_function_var : BasisFunction, optional
            Basis function for the variance, by default None
        n_iter : int, optional
            Number of iterations for the optimization, by default 300
        tol : float, optional
            Tolerance for the optimization, by default 1e-5
        ard : bool, optional
            Whether to use automatic relevance determination, by default False
        optimizer : str, optional
            Optimizer to use for the optimization, by default "l-bfgs-b"
        l_bfgs_b_l : float, optional
            L-BFGS-B parameter, by default 0.1
        l_bfgs_b_epsilon : float, optional
            L-BFGS-B parameter, by default 0.1
        l_bfgs_b_norm : str, optional
            L-BFGS-B parameter, by default "l2"
        hyp0 : np.ndarray, optional
            Initial hyperparameters, by default None
        is_fitted : bool, optional
            Whether the model has been fitted, by default False
        is_from_dict : bool, optional
            Whether the model was created from a dictionary, by default False

        """
        super().__init__(name, is_fitted, is_from_dict)
        self.n_iter = n_iter
        self.tol = tol
        self.ard = ard
        self.optimizer = optimizer
        self.l_bfgs_b_l = l_bfgs_b_l
        self.l_bfgs_b_epsilon = l_bfgs_b_epsilon
        self.l_bfgs_b_norm = l_bfgs_b_norm
        self.fixed_effect = fixed_effect
        self.heteroskedastic = heteroskedastic
        self.fixed_effect_var = fixed_effect_var
        self.warp_name = warp_name
        self.initialize_warp()
        self.warp_reparam = warp_reparam
        self.gamma: np.ndarray = None  # type: ignore
        self.basis_function_mean = (
            copy.deepcopy(basis_function_mean) if basis_function_mean else create_basis_function(basis_function_mean)
        )
        self.basis_function_var = (
            copy.deepcopy(basis_function_var) if basis_function_var else create_basis_function(basis_function_var)
        )
        self.models_variance = self.heteroskedastic or self.fixed_effect_var
        self.hyp0 = hyp0
        self.hyp: np.ndarray = None  # type: ignore

    def initialize_warp(self) -> None:
        if self.warp_name:
            self.warp = self.get_warp(self.warp_name)
            self.warp_params = self.warp.get_n_params()  # type: ignore
            self.n_gamma = self.warp_params
        else:
            self.n_gamma = 0
            self.warp_params = None
            self.warp = None

    def fit(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> None:
        """
        Fit the Bayesian Linear Regression model to the data.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        None
        """
        np_X = X.values
        np_be = be.values
        np_Y = Y.values
        # Phi self.expand_basis
        Phi, Phi_var = self.Phi_Phi_var(np_X, np_be, be_maps)
        self.D = Phi.shape[1]
        self.var_D = Phi_var.shape[1]

        # Initialize hyperparameters if not provided
        hyp0 = self.init_hyp()
        args = (Phi, np_Y, Phi_var)

        match self.optimizer.lower():
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
                out = optimize.fmin_powell(func=self.loglik, x0=hyp0, args=args, full_output=1)
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
                        args=(*args, self.l_bfgs_b_l, self.l_bfgs_b_norm),
                        approx_grad=True,
                        epsilon=self.l_bfgs_b_epsilon,
                        callback=store,
                    )
                except np.linalg.LinAlgError as e:
                    Output.print(Messages.BLR_RESTARTING_ESTIMATION_AT_HYP, hyp=all_hyp_i[-1], e=e)
                    out = optimize.fmin_l_bfgs_b(
                        func=self.penalized_loglik,
                        x0=all_hyp_i[-1],
                        args=(*args, self.l_bfgs_b_l, self.l_bfgs_b_norm),
                        approx_grad=True,
                        epsilon=self.l_bfgs_b_epsilon,
                    )
            case _:
                raise ValueError(Output.error(Errors.ERROR_UNKNOWN_CLASS, class_name=self.optimizer))
        self.hyp = out[0]
        self.nlZ = out[1]
        _, self.beta, self.gamma = self.parse_hyps(self.hyp, Phi, Phi_var)
        self.is_fitted = True

    def forward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> xr.DataArray:
        """Map Y values to Z space using BLR.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        xr.DataArray
            Z-values mapped to Y space
        """
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.BLR_MODEL_NOT_FITTED))
        np_X = X.values
        np_be = be.values
        np_Y = Y.values
        self.ys_s2(np_X, np_be, be_maps)
        if self.warp:
            warped_y = self.warp.f(np_Y, self.gamma)
            toreturn = (warped_y - self.ys) / np.sqrt(self.s2)
        else:
            toreturn = (np_Y - self.ys) / np.sqrt(self.s2)
        return xr.DataArray(toreturn, dims=("observations",))

    def backward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Z: xr.DataArray) -> xr.DataArray:
        """
        Map Z values to Y space using BLR.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Z : xr.DataArray
            Z-score data

        Returns
        -------
        xr.DataArray
            Z-values mapped to Y space
        """
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.BLR_MODEL_NOT_FITTED))
        np_X = X.values
        np_be = be.values
        np_Z = Z.values
        self.ys_s2(np_X, np_be, be_maps)
        centiles = np_Z * np.sqrt(self.s2) + self.ys
        if self.warp:
            centiles = self.warp.invf(centiles, self.gamma)
        return xr.DataArray(centiles, dims=("observations",))

    def elemwise_logp(
        self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ) -> xr.DataArray:
        """

        Compute log-probabilities for each observation in the data.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        xr.DataArray
            Log-probabilities of the data"""
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.BLR_MODEL_NOT_FITTED))

        np_X = X.values
        np_be = be.values
        np_Y = Y.values
        # Get predictions (mean and variance)
        ys, s2 = self.ys_s2(np_X, np_be, be_maps)

        if self.warp:
            # For warped models, transform the observations
            y_warped = self.warp.f(np_Y, self.gamma)
            y = y_warped
        else:
            y = np_Y

        # Compute log probabilities using Gaussian PDF
        # log p(y|x) = -0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        logp = -0.5 * (np.log(2 * np.pi) + np.log(s2) + ((y - ys) ** 2) / s2)
        if self.warp:
            # Add log determinant of Jacobian for warped models
            logp += np.log(self.warp.df(y, self.gamma))

        return xr.DataArray(logp, dims=("observations",))

    def model_specific_evaluation(self, path: str) -> None:
        """
        Save model-specific evaluation metrics.
        """
        pass

    def transfer(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray, **kwargs) -> BLR:
        raise ValueError(Output.error(Errors.ERROR_BLR_TRANSFER_NOT_IMPLEMENTED))

    def init_hyp(self) -> np.ndarray:  # type:ignore
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

        if self.hyp0:
            return self.hyp0

        if self.models_variance:
            n_beta = self.var_D
        else:
            n_beta = 1

        n_alpha = self.D
        n_gamma = self.n_gamma
        self.n_hyp = n_beta + n_alpha + n_gamma  # type: ignore
        return np.ones(self.n_hyp)

    def parse_hyps(
        self, hyp: np.ndarray, Phi: np.ndarray, Phi_var: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse hyperparameters into model parameters.

        Parameters
        ----------
        hyp : np.ndarray
            Hyperparameter vector.
        Phi : np.ndarray
            Covariates.
        Phi_var : np.ndarray (Optional)
            Variance of covariates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Parsed alpha, beta and gamma parameters.
        """
        N = Phi.shape[0]
        beta: np.ndarray = None  # type: ignore
        # Noise precision
        if self.models_variance:
            if Phi_var is None or (Phi_var == 0).all():
                raise ValueError(Output.error(Errors.ERROR_BLR_VAR_X_NOT_PROVIDED))
            Dv = Phi_var.shape[1]
            w_d = np.asarray(hyp[0:Dv])
            beta = np.exp(Phi_var.dot(w_d))
            n_lik_param = len(w_d)
            self.lambda_n_vec = beta
        else:
            beta = np.asarray([np.exp(hyp[0])])
            n_lik_param = len(beta)
            self.lambda_n_vec = np.ones(N) * beta

        if self.warp:
            gamma = hyp[n_lik_param : (n_lik_param + self.n_gamma)]
            n_lik_param += self.n_gamma
            if self.warp_reparam:
                # ! This is bogus if WarpCompose is used, works only for single SinhArcsinh warps
                # TODO: Pass warp_reparam to the warps themselves
                delta = np.exp(gamma[1])
                beta = beta / (delta**2)
        else:
            gamma = None  # type: ignore

        # Coefficients precision
        if isinstance(beta, list) or isinstance(beta, np.ndarray):
            alpha = np.exp(hyp[n_lik_param:])
        else:
            alpha = np.exp(hyp[1:])

        return alpha, beta, gamma  # type: ignore

    def post(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        var_X: Optional[np.ndarray] = None,
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
            Output.print(Messages.BLR_HYPERPARAMETERS_HAVE_NOT_CHANGED)
            return
        else:
            self.hyp = hyp

        # Parse hyperparameters
        alpha, _, _ = self.parse_hyps(self.hyp, X, var_X)

        # prior variance
        if len(alpha) == 1 or len(alpha) == self.D:
            self.Sigma_a = np.diag(np.ones(self.D)) / alpha
            self.Lambda_a = np.diag(np.ones(self.D)) * alpha
        else:
            raise ValueError(Output.error(Errors.BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH))

        # Compute the posterior precision and mean
        XtLambda_n = X.T * self.lambda_n_vec
        self.A = XtLambda_n.dot(X) + self.Lambda_a
        invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
        self.m = (invAXt * self.lambda_n_vec).dot(y)

    def loglik(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        var_X: Optional[np.ndarray] = None,
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
        _, _, gamma = self.parse_hyps(hyp, X, var_X)

        if self.warp:
            y_unwarped = copy.deepcopy(y)
            y = self.warp.f(y, gamma)

        something_big: float = float(np.finfo(np.float64).max)

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not hasattr(self, "A"):
            try:
                self.post(hyp, X, y, var_X)
            except ValueError as error:
                Output.warning(Warnings.BLR_ESTIMATION_OF_POSTERIOR_DISTRIBUTION_FAILED, error=error)
                nlZ = something_big
                return nlZ

        try:
            # compute the log determinants in a numerically stable way
            logdetA = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(self.A))))
        except (ValueError, LinAlgError) as error:
            Output.warning(Warnings.BLR_ESTIMATION_OF_POSTERIOR_DISTRIBUTION_FAILED, error=error)
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

        if self.warp:
            nlZ = nlZ - np.sum(np.log(self.warp.df(y_unwarped, gamma)))

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
        var_X: Optional[np.ndarray] = None,
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
            return self.loglik(hyp, X, y, var_X) + regularizer_strength * np.sum(np.abs(hyp))
        elif norm.upper() == "L2":
            return self.loglik(hyp, X, y, var_X) + regularizer_strength * np.sqrt(np.sum(np.square(hyp)))
        else:
            raise ValueError(Output.error(Errors.ERROR_BLR_PENALTY_NOT_RECOGNIZED, penalty=norm))

    def dloglik(self, hyp: np.ndarray, X: np.ndarray, y: np.ndarray, var_X: np.ndarray) -> np.ndarray:
        """Function to compute derivatives"""

        # hyperparameters
        alpha, beta, gamma = self.parse_hyps(hyp, X, var_X)

        if self.warp:
            raise ValueError(
                Output.error(Errors.ERROR_UNKNOWN_FUNCTION_FOR_CLASS, func="dloglik", class_name=self.__class__.__name__)
            )

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not hasattr(self, "A"):
            try:
                self.post(hyp, X, y, var_X)
            except ValueError as error:
                Output.warning(Warnings.BLR_ESTIMATION_OF_POSTERIOR_DISTRIBUTION_FAILED, error=error)
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

    def Phi_Phi_var(self, X: np.ndarray, be: np.ndarray, be_maps: dict[str, dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.basis_function_mean.is_fitted:
            self.basis_function_mean.fit(X)
        Phi = create_design_matrix(
            self.basis_function_mean.transform(X),
            be,
            be_maps,
            linear=True,
            intercept=True,
            fixed_effect=self.fixed_effect,
        )

        if self.models_variance:
            if not self.basis_function_var.is_fitted:
                self.basis_function_var.fit(X)
            Phi_var = create_design_matrix(
                self.basis_function_var.transform(X),
                be,
                be_maps,
                linear=self.heteroskedastic,
                intercept=True,
                fixed_effect=self.fixed_effect_var,
            )
        else:
            Phi_var = np.zeros((Phi.shape[0], 1))
        return Phi, Phi_var

    def ys_s2(self, X: np.ndarray, be: np.ndarray, be_maps: dict[str, dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
        Phi, Phi_var = self.Phi_Phi_var(X, be, be_maps)
        _, beta, self.gamma = self.parse_hyps(self.hyp, Phi, Phi_var)
        ys = Phi.dot(self.m)
        s2n = 1 / beta
        s2 = s2n + np.sum(Phi * linalg.solve(self.A, Phi.T).T, axis=1)
        self.ys = ys
        self.s2 = s2
        return ys, s2

    def to_dict(self, path: str | None = None) -> dict:
        my_dict = self.regmodel_dict
        for key, value in self.__dict__.items():
            if not key == "warp":
                if isinstance(value, np.ndarray):
                    my_dict[key] = value.tolist()
                elif key in ["basis_function_mean", "basis_function_var"]:
                    my_dict[key] = value.to_dict()
                else:
                    my_dict[key] = value
        return my_dict

    @classmethod
    def from_dict(cls, my_dict: dict, path: str | None = None) -> "BLR":
        """
        Creates a configuration from a dictionary.
        """
        self = cls(my_dict["name"])
        for key, value in my_dict.items():
            if isinstance(value, list):
                object.__setattr__(self, key, np.array(value))
            else:
                object.__setattr__(self, key, value)
        self.initialize_warp()
        self.is_from_dict = True
        self.basis_function_mean = BasisFunction.from_dict(my_dict["basis_function_mean"])
        self.basis_function_var = BasisFunction.from_dict(my_dict["basis_function_var"])
        return self

    @classmethod
    def from_args(cls, name: str, args: dict) -> "BLR":
        """
        Creates a configuration from command line arguments
        """
        _default_instance = BLR("default")
        is_from_dict = True
        is_fitted = args.get("is_fitted", False)
        self = cls(name, is_fitted, is_from_dict)
        name = args.get("name", "BLR")
        n_iter = args.get("n_iter", _default_instance.n_iter)
        tol = args.get("tol", _default_instance.tol)
        ard = args.get("ard", _default_instance.ard)
        optimizer = args.get("optimizer", _default_instance.optimizer)
        l_bfgs_b_l = args.get("l_bfgs_b_l", _default_instance.l_bfgs_b_l)
        l_bfgs_b_epsilon = args.get("l_bfgs_b_epsilon", _default_instance.l_bfgs_b_epsilon)
        l_bfgs_b_norm = args.get("l_bfgs_b_norm", _default_instance.l_bfgs_b_norm)
        fixed_effect = args.get("fixed_effect", _default_instance.fixed_effect)
        heteroskedastic = args.get("heteroskedastic", _default_instance.heteroskedastic)
        fixed_effect_var = args.get("fixed_effect_var", _default_instance.fixed_effect_var)
        warp = args.get("warp", _default_instance.warp)
        warp_reparam = args.get("warp_reparam", _default_instance.warp_reparam)
        try:
            basis_function_mean = create_basis_function(args.get("basis_function_mean"))
        except ValueError:
            basis_function_mean = _default_instance.basis_function_mean
        try:
            basis_function_var = create_basis_function(args.get("basis_function_var"))
        except ValueError:
            basis_function_var = _default_instance.basis_function_var
        hyp0 = args.get("hyp0", _default_instance.hyp0)
        self = cls(
            name=name,
            is_fitted=is_fitted,
            is_from_dict=is_from_dict,
            n_iter=n_iter,
            tol=tol,
            ard=ard,
            optimizer=optimizer,
            l_bfgs_b_l=l_bfgs_b_l,
            l_bfgs_b_epsilon=l_bfgs_b_epsilon,
            l_bfgs_b_norm=l_bfgs_b_norm,
            fixed_effect=fixed_effect,
            heteroskedastic=heteroskedastic,
            fixed_effect_var=fixed_effect_var,
            warp_name=warp,
            warp_reparam=warp_reparam,
            basis_function_mean=basis_function_mean,
            basis_function_var=basis_function_var,
            hyp0=hyp0,
        )

        return self

    def get_warp(self, warp: str | None) -> Optional[WarpBase]:
        if warp is None:
            return None
        if warp.lower() == "warpboxcox":
            return WarpBoxCox()
        elif warp.lower() == "warpaffine":
            return WarpAffine()
        elif warp.lower() == "warpsinharcsinh":
            return WarpSinhArcsinh()
        elif warp.lower() == "warplog":
            return WarpLog()
        elif warp.lower().startswith("warpcompose"):
            # Expect a comma separated list of warp names: e.g. warpcompose(warpboxcox,warpaffine,warpsinharcsinh)
            warps = []
            for warp_name in warp.lower().split("warpcompose")[1].strip("[]").split(","):
                warps.append(self.get_warp(warp_name.strip()))
            return WarpCompose(warps)
        else:
            raise ValueError(Output.error(Errors.ERROR_UNKNOWN_CLASS, class_name=warp))

    @property
    def has_batch_effect(self) -> bool:
        return self.fixed_effect or self.fixed_effect_var


def create_design_matrix(
    X: np.ndarray,
    be: np.ndarray,
    be_maps: dict[str, dict[str, int]],
    linear: bool = False,
    intercept: bool = False,
    fixed_effect: bool = False,
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
    fixed_effect : bool, default=False
        Include fixed effect for batch effects.

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
        acc.append(X)

    if intercept:
        acc.append(np.ones((X.shape[0], 1)))

    # Create one-hot encoding for fixed effect
    if fixed_effect:
        for i, v in enumerate(be_maps.values()):
            acc.append(
                np.eye(len(v))[be[:, i]],
            )

    if len(acc) == 0:
        raise ValueError(Output.error(Errors.BLR_ERROR_NO_DESIGN_MATRIX_CREATED))

    return np.concatenate(acc, axis=1)
