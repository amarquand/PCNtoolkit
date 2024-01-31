import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pytest_tests.fixtures.data import *


def test_create_data_from_arrays(
    train_arrays, n_covariates, n_train_datapoints, n_response_vars
):
    X, y, batch_effects = train_arrays
    norm_data = NormData.from_ndarrays("test", X, y, batch_effects)
    assert norm_data.X.shape == (n_train_datapoints, n_covariates)
    assert norm_data.y.shape == (n_train_datapoints, n_response_vars)
    assert norm_data.batch_effects.shape == (n_train_datapoints, n_covariates)
    assert norm_data.covariates.shape == (n_covariates,)
    assert norm_data.batch_effect_dims.shape == (2,)
    assert norm_data.attrs["batch_effects_maps"]["batch_effect_0"] == {0: 0, 1: 1}
    assert norm_data.attrs["batch_effects_maps"]["batch_effect_1"] == {0: 0, 1: 1, 2: 2}
    assert norm_data.coords["datapoints"].shape == (n_train_datapoints,)
    assert norm_data.coords["covariates"].shape == (n_covariates,)
    assert norm_data.coords["batch_effect_dims"].shape == (2,)


def test_create_data_from_dataframe(
    train_dataframe, n_covariates, n_train_datapoints, n_response_vars
):
    norm_data = NormData.from_dataframe(
        "test",
        train_dataframe,
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
    assert norm_data.X.shape == (n_train_datapoints, n_covariates)
    assert norm_data.y.shape == (n_train_datapoints, n_response_vars)
    assert norm_data.batch_effects.shape == (n_train_datapoints, n_covariates)
    assert norm_data.covariates.shape == (n_covariates,)
    assert norm_data.batch_effect_dims.shape == (n_covariates,)
    assert norm_data.attrs["batch_effects_maps"]["batch1"] == {0: 0, 1: 1}
    assert norm_data.attrs["batch_effects_maps"]["batch2"] == {0: 0, 1: 1, 2: 2}
    assert norm_data.coords["datapoints"].shape == (n_train_datapoints,)
    assert norm_data.coords["covariates"].to_numpy().tolist() == ["X1", "X2"]
    assert norm_data.coords["batch_effect_dims"].to_numpy().tolist() == [
        "batch1",
        "batch2",
    ]


def test_split_with_stratify(
    train_norm_data, n_covariates, n_train_datapoints, n_response_vars
):
    splits = train_norm_data.train_test_split(
        splits=(0.5, 0.5), split_names=("train", "val")
    )
    assert len(splits) == 2
    assert splits[0].attrs["name"] == "train"
    assert splits[1].attrs["name"] == "val"
    assert splits[0].X.shape == (n_train_datapoints // 2, n_covariates)
    assert splits[0].y.shape == (n_train_datapoints // 2, n_response_vars)
    assert splits[0].batch_effects.shape == (n_train_datapoints // 2, n_covariates)
    assert splits[0].covariates.shape == (n_covariates,)
    assert splits[0].batch_effect_dims.shape == (n_covariates,)
    assert splits[0].batch_effects_maps["batch1"] == {0: 0, 1: 1}
    assert splits[0].batch_effects_maps["batch2"] == {0: 0, 1: 1, 2: 2}
    assert splits[0].coords["datapoints"].shape == (n_train_datapoints // 2,)
    assert splits[0].coords["covariates"].to_numpy().tolist() == ["X1", "X2"]
    assert splits[0].coords["batch_effect_dims"].to_numpy().tolist() == [
        "batch1",
        "batch2",
    ]
    assert splits[1].X.shape == (n_train_datapoints // 2, n_covariates)
    assert splits[1].y.shape == (n_train_datapoints // 2, n_response_vars)
    assert splits[1].batch_effects.shape == (n_train_datapoints // 2, n_covariates)
    assert splits[1].covariates.shape == (n_covariates,)
    assert splits[1].batch_effect_dims.shape == (n_covariates,)
    assert splits[1].batch_effects_maps["batch1"] == {0: 0, 1: 1}
    assert splits[1].batch_effects_maps["batch2"] == {0: 0, 1: 1, 2: 2}
    assert splits[1].coords["datapoints"].shape == (n_train_datapoints // 2,)
    assert splits[1].coords["covariates"].to_numpy().tolist() == ["X1", "X2"]
    assert splits[1].coords["batch_effect_dims"].to_numpy().tolist() == [
        "batch1",
        "batch2",
    ]

    original_unique_batch_effects = np.unique(
        train_norm_data.batch_effects.data, axis=0
    )
    split1_unique_batch_effects = np.unique(splits[0].batch_effects.data, axis=0)
    split2_unique_batch_effects = np.unique(splits[1].batch_effects.data, axis=0)
    assert np.all(original_unique_batch_effects == split1_unique_batch_effects)
    assert np.all(original_unique_batch_effects == split2_unique_batch_effects)
