from typing import List

import numpy as np
import xarray as xr
from scipy import stats  # type: ignore
from sklearn.metrics import explained_variance_score  # type: ignore

from pcntoolkit.dataio.norm_data import NormData


class Evaluator:
    """
    A class for evaluating normative model predictions.

    This class implements various statistical measures to assess the quality of
    normative model predictions, including correlation coefficients, error metrics,
    and normality tests.

    Attributes
    ----------
    response_vars : List[str]
        List of response variables to evaluate
    """

    def __init__(self) -> None:
        """Initialize the Evaluator."""
        self.response_vars: List[str] = []

    def evaluate(self, data: NormData) -> NormData:
        """
        Evaluate model predictions using multiple statistical measures.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        NormData
            Data container updated with evaluation measures
        """
        data["Yhat"] = data.centiles.sel(cdf=0.5, method="nearest")
        # ! For asymmetric distributions, this is not correct.
        data["S2"] = (
            data.centiles.sel(cdf=0.1587, method="nearest") - data["Yhat"]
        ) ** 2
        self.create_measures_group(data)

        self.evaluate_shapiro_w(data)
        # self.evaluate_bic(data)
        self.evaluate_rho(data)
        self.evaluate_rmse(data)
        self.evaluate_smse(data)
        self.evaluate_expv(data)
        # self.evaluate_msll(data)
        self.evaluate_nll(data)
        return data

    def create_measures_group(self, data: NormData) -> None:
        """
        Create a measures group in the data container.

        Parameters
        ----------
        data : NormData
            Data container to add measures group to
        """
        data["measures"] = xr.DataArray(
            np.nan * np.ones((len(data.response_var_list), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": data.response_var_list,
                "statistics": ["Rho", "RMSE", "SMSE", "ExpV", "NLL", "ShapiroW"],
            },
        )

    def evaluate_rho(self, data: NormData) -> None:
        """
        Evaluate Spearman's rank correlation coefficient.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            rho = self._evaluate_rho(resp_predict_data)
            data.measures.loc[{"response_vars": responsevar, "statistics": "Rho"}] = float(rho)

    # Similar docstrings should be added for other evaluation methods...

    def evaluate_rmse(self, data: NormData) -> None:
        """
        Evaluate Root Mean Square Error (RMSE) for model predictions.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            rmse = self._evaluate_rmse(resp_predict_data)
            data.measures.loc[{"response_vars": responsevar, "statistics": "RMSE"}] = rmse

    def evaluate_smse(self, data: NormData) -> None:
        """
        Evaluate Standardized Mean Square Error (SMSE) for model predictions.

        SMSE normalizes the mean squared error by the variance of the target variable,
        making it scale-independent.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            smse = self._evaluate_smse(resp_predict_data)
            data.measures.loc[{"response_vars": responsevar, "statistics": "SMSE"}] = smse

    def evaluate_expv(self, data: NormData) -> None:
        """
        Evaluate Explained Variance score for model predictions.

        The explained variance score measures the proportion of variance in the target variable
        that is predictable from the input features.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            expv = self._evaluate_expv(resp_predict_data)
            data.measures.loc[{"response_vars": responsevar, "statistics": "ExpV"}] = expv

    def evaluate_msll(self, data: NormData) -> None:
        """
        Evaluate Mean Standardized Log Loss (MSLL) for model predictions.

        MSLL compares the log loss of the model to that of a simple baseline predictor
        that always predicts the mean of the training data.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y', 'Yhat',
            and standard deviation predictions.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            msll = self._evaluate_msll(resp_predict_data)
            data.measures.loc[{"response_vars": responsevar, "statistics": "MSLL"}] = msll

    def evaluate_nll(self, data: NormData) -> None:
        """
        Evaluate Negative Log Likelihood (NLL) for model predictions.

        NLL measures the probabilistic accuracy of the model's predictions, assuming
        binary classification targets.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
            'y' should contain binary values (0 or 1).
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            nll = self._evaluate_nll(resp_predict_data)
            data.measures.loc[{"response_vars": responsevar, "statistics": "NLL"}] = nll

    def evaluate_bic(self, data: NormData) -> None:
        """
        Evaluate Bayesian Information Criterion (BIC) for model predictions.

        BIC is a criterion for model selection that measures the trade-off between
        model fit and complexity.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            bic = self._evaluate_bic(resp_predict_data)
            self.prepare(responsevar)
            data.measures.loc[{"response_vars": responsevar, "statistics": "BIC"}] = bic
            self.reset()

    def evaluate_shapiro_w(self, data: NormData) -> None:
        """
        Evaluate Shapiro-Wilk test statistic for normality of residuals.

        The Shapiro-Wilk test assesses whether the z-scores follow a normal distribution.
        A higher W statistic (closer to 1) indicates stronger normality.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'zscores' variable.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            shapiro_w = self._evaluate_shapiro_w(resp_predict_data)
            data.measures.loc[
                {"response_vars": responsevar, "statistics": "ShapiroW"}
            ] = shapiro_w

    def _evaluate_rho(self, data: NormData) -> float:
        """
        Calculate Spearman's rank correlation coefficient.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Spearman's rank correlation coefficient between actual and predicted values
        """
        y = data["y"].values
        yhat = data["Yhat"].values
        rho, _ = stats.spearmanr(y, yhat)
        return float(rho)

    def _evaluate_rmse(self, data: NormData) -> float:
        """
        Calculate Root Mean Square Error.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Root mean square error between actual and predicted values
        """
        y = data["y"].values
        yhat = data["Yhat"].values
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return float(rmse)

    def _evaluate_smse(self, data: NormData) -> float:
        """
        Calculate Standardized Mean Square Error.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Standardized mean square error between actual and predicted values
        """
        y = data["y"].values
        yhat = data["Yhat"].values

        mse = np.mean((y - yhat) ** 2)
        variance = np.var(y)
        smse = float(mse / variance if variance != 0 else 0)

        return smse

    def _evaluate_expv(self, data: NormData) -> float:
        """
        Calculate Explained Variance score.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Explained variance score between actual and predicted values
        """
        y = data["y"].values
        yhat = data["Yhat"].values

        expv = explained_variance_score(y, yhat)
        return float(expv)  # Explicitly cast to float

    def _evaluate_msll(self, data: NormData) -> float:
        """
        Calculate Mean Standardized Log Loss.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Mean standardized log loss between actual and predicted values
        """
        y = data["y"].values
        yhat = data["Yhat"].values
        yhat_std = data["Yhat"]

        log_loss = np.mean((y - yhat) ** 2 / (2 * yhat_std**2) + np.log(yhat_std))
        naive_std = np.std(y)
        naive_log_loss = np.mean(
            (y - np.mean(y)) ** 2 / (2 * naive_std**2) + np.log(naive_std)
        )

        msll = float(log_loss - naive_log_loss)  # Explicitly cast to float
        return msll

    def _evaluate_nll(self, data: NormData) -> float:
        """
        Calculate Negative Log Likelihood.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Assumes binary targets (0 or 1).

        Returns
        -------
        float
            Negative log likelihood of predictions
        """
        y = data["y"].values
        yhat = data["Yhat"].values

        nll = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return float(nll)  # Explicitly cast to float

    def _evaluate_bic(self, data: NormData) -> float:
        """
        Calculate Bayesian Information Criterion.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Bayesian Information Criterion value
        """
        n_params = self.n_params()
        y = data["y"].values
        yhat = data["Yhat"].values

        rss = np.sum((y - yhat) ** 2)
        n = len(y)
        bic = float(n * np.log(rss / n) + n_params * np.log(n))  # Explicitly cast to float
        return bic

    def _evaluate_shapiro_w(self, data: NormData) -> float:
        """
        Calculate Shapiro-Wilk test statistic.

        Parameters
        ----------
        data : NormData
            Data container with z-scores

        Returns
        -------
        float
            Shapiro-Wilk W statistic
        """
        y = data["zscores"].values
        shapiro_w, _ = stats.shapiro(y)
        return float(shapiro_w)  # Explicitly cast to float

    def empty_measure(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros(len(self.response_vars)),
            dims=("response_vars"),
            coords={"response_vars": self.response_vars},
        )

    def prepare(self, responsevar: str) -> None:
        """Prepare the evaluator for a specific response variable."""
        pass

    def reset(self) -> None:
        """Reset the evaluator state."""
        pass

    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return 0  # Override in subclasses
