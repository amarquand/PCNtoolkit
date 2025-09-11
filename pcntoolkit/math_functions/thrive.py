from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression

from pcntoolkit.dataio.norm_data import NormData


def design_matrix(bandwidth: int, Sigma: np.ndarray) -> pd.DataFrame:
    """Constructs a design matrix according to: Buuren, S. Evaluation and prediction of individual growth trajectories. Ann. Hum. Biol. 50, 247–257 (2023).

    Args:
        bandwidth (int): The bandwidth for which the covariance has been computed
        Sigma np.ndarray: Covariate matrix with possibly missing values. The 0'th column represents an age of 0.
    Returns:
        pd.DataFrame: A design matrix with regressors and predictors. The matrix may have missing values in the 'y' column.
    """
    max_age = Sigma.shape[0] - 1
    Ages = np.arange(max_age)
    dfs = []
    for offset in range(1, bandwidth + 1):
        ages_i = Ages[: max_age - offset + 1]
        df_i = pd.DataFrame(index=ages_i)
        df_i["v0"] = 1
        df_i["V1"] = np.log(ages_i + (offset / 2))
        df_i["V2"] = np.log(offset)
        df_i["V3"] = 1 / offset
        df_i["V4"] = df_i["V1"] * df_i["V2"]
        df_i["V5"] = df_i["V1"] ** 2
        df_i["y"] = np.diagonal(Sigma, offset)
        dfs.append(df_i)
    df = pd.concat(dfs, axis=0)
    return df


def fill_missing(bandwidth: int, cors: np.ndarray) -> np.ndarray:
    """Fills in missing correlation values according to:

    Args:
        bandwidth (int): the bandwidth within which the indices are filled in
        cors (np.ndarray): possibly incomplete correlation matrix of shape [n_responsevars, n_ages, n_ages]

    Returns:
        np.ndarray: New matrix completed with predicted values
    """
    f_cors = fisher_transform(cors)
    max_age = f_cors.shape[1] - 1
    newcors = np.zeros_like(f_cors)
    # Loop over response variables
    for rv in range(f_cors.shape[0]):
        # Create design matrix
        Phi = design_matrix(bandwidth, f_cors[rv])
        # Drop rows with NaN
        Xy = Phi.dropna(axis=0, inplace=False)
        # Fit regressionmodel to cleaned data
        regmodel = LinearRegression(fit_intercept=False).fit(Xy.drop(columns="y", inplace=False), y=Xy[["y"]])
        # Use that to infer all rows including the rows with NaN
        y_pred = regmodel.predict(Phi.drop(columns="y"))
        # Fill in the predicted correlations
        for i, (age1, age2) in enumerate(offset_indices(max_age, bandwidth)):
            newcors[rv, age1, age2] = newcors[rv, age2, age1] = y_pred[i].item()
    # Inverse Fisher transform (tanh)
    newcors = np.tanh(newcors)
    # Take only the predicted values where there were missing values
    # newcors = np.where(np.isnan(f_cors), newcors, f_cors)
    return newcors


def offset_indices(max_age: int, bandwidth: int):
    """Generate pairs of indices that iterate over all the cells in the upper triangular region specified by the parameters.

    E.g:
    Offset_indices(3, 2) will yield (0,1) -> (0,2) -> (1,2) -> (1,3) -> (2,3)
    Which index these positions:
    _,0,1,_
    _,_,2,3
    _,_,_,4
    _,_,_,_

    Args:
        max_age (int): max age for which indices are generated (includes 0)
        bandwidth (int): the bandwidth within which the indices are computed

    Yields:
        (int, int): pairs of indices
    """
    acc = np.zeros((max_age + 1, max_age + 1))
    acc[np.triu_indices(max_age + 1, 1)] = 1
    acc[np.triu_indices(max_age + 1, bandwidth + 1)] = 0
    for pair in zip(*np.where(acc)):
        yield pair


def fisher_transform(cor):
    epsilon = 1e-13
    cor = np.clip(cor, -1 + epsilon, 1 + epsilon)
    return 0.5 * np.log((1 + cor) / (1 - cor))


def get_correlation_matrix(data: NormData, bandwidth: int, covariate_name="age"):
    """Compute correlations of Z scores between pairs of observations of the same subject at different ages

    Args:
        data (NormData): Data containing covariates, predicted Z-scores, batch effects and subject indices
        bandwidth (int): The age offset range within which correlations are computed
        covariate_name (str, optional): Covariate to use for grouping subjects. Defaults to "age".

    Returns:
        xr.DataArray: Correlations of shape [n_response_vars, n_ages, n_ages]
    """

    df = data.to_dataframe()[["X", "Z", "batch_effects", "subjects"]].droplevel(level=0, axis=1)
    # create dictionary of (age:indices)
    grps = df.groupby(covariate_name).indices | defaultdict(list)
    # get the max age in the dataset
    max_age = int(max(list(grps.keys())))  # type:ignore
    # the number of response variable for which to compute correlations
    n_responsevars = len(data.response_vars.to_numpy())
    # create empty correlation matrix
    cors = np.tile(np.eye(max_age + 1), (n_responsevars, 1, 1))
    for age1, age2 in offset_indices(max_age, bandwidth):
        # merge two ages on subjects
        merged = pd.merge(df.iloc[grps[age1]], df.iloc[grps[age2]], how="inner", on="subjects")
        if len(merged) >= 4:
            # Compute correlations if there are enough samples
            for i, rv in enumerate(data.response_vars.to_numpy()):
                cors[i, age2, age1] = cors[i, age1, age2] = merged[f"{rv}_x"].corr(merged[f"{rv}_y"])
        elif age1 != age2:
            # Otherwise, set all response variables to NaN for these ages
            cors[:, age2, age1] = cors[:, age1, age2] = np.NaN
    # Fill in missing correlation values
    newcors = fill_missing(bandwidth, cors)
    newcors = xr.DataArray(
        newcors,
        dims=("response_vars", f"{covariate_name}_1", f"{covariate_name}_2"),
        coords={
            "response_vars": data.response_vars.to_numpy(),
            f"{covariate_name}_1": np.arange(cors.shape[1]),
            f"{covariate_name}_2": np.arange(cors.shape[1]),
        },
    )
    return newcors


def get_thrive_Z_X(cors: xr.DataArray, start_x: xr.DataArray, start_z: xr.DataArray, span: int, z_thrive=1.96):
    assert start_x.shape == start_z.shape
    assert cors.shape[0] == cors.shape[1]
    padded_cors = np.pad(cors, ((0, span), (0, span)), mode="edge")
    thrive_Z = np.zeros((start_x.shape[0], span + 1))
    thrive_X = np.zeros_like(thrive_Z).astype(int)
    thrive_X[:, 0] = start_x
    thrive_Z[:, 0] = start_z
    for i in range(span):
        thrive_X[:, i + 1] = thrive_X[:, i] + 1
        this_cors = padded_cors[thrive_X[:, i], thrive_X[:, i + 1]]
        thrive_Z[:, i + 1] = thrive_Z[:, i] * this_cors + np.sqrt(1 - this_cors**2) * z_thrive
    thrive_Z = xr.DataArray(thrive_Z, dims=("observations", "offset"))
    thrive_X = xr.DataArray(thrive_X.astype(float), dims=("observations", "offset"))
    return thrive_Z, thrive_X
