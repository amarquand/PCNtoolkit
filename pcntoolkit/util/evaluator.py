import numpy as np
from scipy import stats
from sklearn.metrics import explained_variance_score
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData


class Evaluator:
    def __init__(self):
        pass

    def compute_measures(self, data: NormData) -> NormData:
        data["Yhat"] = data.centiles.sel(cummulative_densities=0.5, method="nearest")
        data["S2"] = (
            data.centiles.sel(cummulative_densities=0.1587, method="nearest")
            - data["Yhat"]
        ) ** 2
        self.response_vars = data.response_vars.to_numpy().copy().tolist()

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

    def create_measures_group(self, data):
        data["measures"] = xr.DataArray(
            np.nan * np.ones((len(self.response_vars), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": self.response_vars,
                "statistics": ["Rho", "RMSE", "SMSE", "ExpV", "NLL", "ShapiroW"],
            },
        )

    def evaluate_rho(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            rho = self._evaluate_rho(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "Rho"}] = rho

    def evaluate_rmse(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            rmse = self._evaluate_rmse(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "RMSE"}] = (
                rmse
            )

    def evaluate_smse(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            smse = self._evaluate_smse(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "SMSE"}] = (
                smse
            )

    def evaluate_expv(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            expv = self._evaluate_expv(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "ExpV"}] = (
                expv
            )

    def evaluate_msll(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            msll = self._evaluate_msll(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "MSLL"}] = (
                msll
            )

    def evaluate_nll(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            nll = self._evaluate_nll(resp_predict_data)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "NLL"}] = nll

    def evaluate_bic(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            bic = self._evaluate_bic(resp_predict_data)

            self.prepare(responsevar)

            # Store the measure
            data.measures.loc[{"response_vars": responsevar, "statistics": "BIC"}] = bic

            self.reset()

    def evaluate_shapiro_w(self, data: NormData):
        for responsevar in self.response_vars:
            # Select the data for the current response variable
            resp_predict_data = data.sel(response_vars=responsevar)

            # Compute the measure
            shapiro_w = self._evaluate_shapiro_w(resp_predict_data)

            # Store the measure
            data.measures.loc[
                {"response_vars": responsevar, "statistics": "ShapiroW"}
            ] = shapiro_w

    def _evaluate_rho(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        rho, _ = stats.spearmanr(y, yhat)
        return rho

    def _evaluate_rmse(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return rmse

    def _evaluate_smse(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        mse = np.mean((y - yhat) ** 2)
        variance = np.var(y)
        smse = mse / variance if variance != 0 else 0

        return smse

    def _evaluate_expv(self, data: NormData) -> float:
        y = data["y"].values
        yhat = data["Yhat"].values

        expv = explained_variance_score(y, yhat)
        return expv

    def _evaluate_msll(self, data: NormData) -> float:
        # TODO check if this is correct

        y = data["y"].values
        yhat = data["Yhat"].values
        yhat_std = data["Yhat"]

        # Calculate the log loss of the model's predictions
        log_loss = np.mean((y - yhat) ** 2 / (2 * yhat_std**2) + np.log(yhat_std))

        # Calculate the log loss of the naive model
        naive_std = np.std(y)
        naive_log_loss = np.mean(
            (y - np.mean(y)) ** 2 / (2 * naive_std**2) + np.log(naive_std)
        )

        # Calculate MSLL
        msll = log_loss - naive_log_loss

        return msll

    def _evaluate_nll(self, data: NormData) -> float:
        # TODO check if this is correct

        # assume 'Y' is binary (0 or 1)
        y = data["y"].values
        yhat = data["Yhat"].values

        # Calculate the NLL
        nll = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return nll

    def _evaluate_bic(self, data: NormData) -> float:
        n_params = self.n_params()

        # Assuming 'data' is a NormData object with 'Y' and 'Yhat' DataArrays
        y = data["y"].values
        yhat = data["Yhat"].values

        # Calculate the residual sum of squares
        rss = np.sum((y - yhat) ** 2)

        # Calculate the number of observations
        n = len(y)

        # Calculate the BIC
        bic = n * np.log(rss / n) + n_params * np.log(n)

        return bic

    def _evaluate_shapiro_w(self, data: NormData) -> float:
        y = data["zscores"].values
        shapiro_w, _ = stats.shapiro(y)
        return shapiro_w

    def empty_measure(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros(len(self.response_vars)),
            dims=("response_vars"),
            coords={"response_vars": self.response_vars},
        )
