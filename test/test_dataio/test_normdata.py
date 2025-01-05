import numpy as np
import pytest

from test.fixtures.norm_data_fixtures import *

"""
This file contains tests for the NormData class in the PCNtoolkit.

The tests cover the following aspects:
1. Creating NormData objects from numpy arrays and pandas DataFrames
2. Verifying the structure and properties of created NormData objects
3. Splitting NormData into training and validation sets
4. Checking the stratification of splits

These tests ensure that the NormData class correctly handles data input,
maintains data integrity, and performs proper data splitting and stratification.
"""


@pytest.mark.parametrize(
    "norm_data_fixture, n_batch_effect_dims",
    [("norm_data_from_arrays", 2), ("norm_data_from_dataframe", 2)],
)
def test_norm_data_creation(
    request,
    norm_data_fixture,
    n_batch_effect_dims,
    n_covariates,
    n_train_datapoints,
    batch_effect_values,
    n_response_vars,
):
    norm_data = request.getfixturevalue(norm_data_fixture)

    assert norm_data.X.shape == (n_train_datapoints, n_covariates)
    assert norm_data.y.shape == (n_train_datapoints, n_response_vars)
    assert norm_data.batch_effects.shape == (n_train_datapoints, len(batch_effect_values))
    assert norm_data.covariates.shape == (n_covariates,)
    assert norm_data.batch_effect_dims.shape == (n_batch_effect_dims,)
    assert norm_data.coords["datapoints"].shape == (n_train_datapoints,)
    assert norm_data.coords["covariates"].shape == (n_covariates,)
    assert norm_data.coords["batch_effect_dims"].shape == (n_batch_effect_dims,)


@pytest.mark.parametrize("split_ratio", [(0.5, 0.5), (0.7, 0.3), (0.8, 0.2)])
def test_split_with_stratify(
    norm_data_from_arrays,
    n_covariates,
    n_train_datapoints,
    n_response_vars,
    split_ratio,
    batch_effect_values,
):
    splits = norm_data_from_arrays.train_test_split(
        splits=split_ratio, split_names=("train", "val")
    )

    # Check basic split properties
    assert len(splits) == 2
    assert splits[0].name == "train"
    assert splits[1].name == "val"

    # Check if total samples in splits equal original samples
    assert splits[0].X.shape[0] + splits[1].X.shape[0] == n_train_datapoints

    for i, split in enumerate(splits):
        expected_samples = int(n_train_datapoints * split_ratio[i])
        assert split.X.shape == (expected_samples, n_covariates)
        assert split.y.shape == (expected_samples, n_response_vars)
        assert split.batch_effects.shape == (expected_samples, len(batch_effect_values))
        assert split.covariates.shape == (n_covariates,)
        assert split.batch_effect_dims.shape == (len(batch_effect_values),)
        assert split.coords["datapoints"].shape == (expected_samples,)
        assert split.coords["covariates"].to_numpy().tolist() == [f"covariate_{i}" for i in range(n_covariates)]
        assert split.coords["batch_effect_dims"].to_numpy().tolist() == [f"batch_effect_{i}" for i in range(len(batch_effect_values))]

    # Check if stratification worked
    original_batch_effects = norm_data_from_arrays.batch_effects.data

    # Check column-wise value distributions
    for col_idx in range(original_batch_effects.shape[1]):
        original_values, original_counts = np.unique(original_batch_effects[:, col_idx], return_counts=True)
        original_frequencies = original_counts / len(original_batch_effects)
        
        for split in splits:
            split_values, split_counts = np.unique(split.batch_effects.data[:, col_idx], return_counts=True)
            split_frequencies = split_counts / len(split.batch_effects.data)
            
            # Check if all unique values from original exist in split
            assert np.all(np.isin(original_values, split_values)), \
                f"Split missing some values from column {col_idx}"
            
            # Check if frequencies are approximately equal (within 10% tolerance)
            for val, orig_freq in zip(original_values, original_frequencies):
                split_freq = split_frequencies[split_values == val][0]
                np.testing.assert_allclose(
                    split_freq, 
                    orig_freq, 
                    atol=0.1,
                    err_msg=f"Frequency mismatch for value {val} in column {col_idx}"
                )

    # Check if all unique batch effects are present in both splits
    original_unique_batch_effects = np.unique(original_batch_effects, axis=0)
    for split in splits:
        split_unique_batch_effects = np.unique(split.batch_effects.data, axis=0)
        assert np.all(
            np.isin(original_unique_batch_effects, split_unique_batch_effects)
        )

        # Check if the data in splits is a subset of the original data
        original_data = np.hstack(
            (
                norm_data_from_arrays.X.data,
                norm_data_from_arrays.y.data,
                norm_data_from_arrays.batch_effects.data,
            )
        )
        for split in splits:
            split_data = np.hstack(
                (split.X.data, split.y.data, split.batch_effects.data)
            )
            assert np.all(np.isin(split_data, original_data))

        # Check if the attributes are preserved in the splits
        for split in splits:
            assert split.name in ["train", "val"]
            for key in norm_data_from_arrays.attrs:
                if key != "name":
                    assert split.attrs[key] == norm_data_from_arrays.attrs[key]

        # Check if the coordinate names are preserved
        for split in splits:
            assert list(split.coords) == list(norm_data_from_arrays.coords)

        # Check if the data variable names are preserved
        for split in splits:
            assert list(split.data_vars) == list(norm_data_from_arrays.data_vars)

        # Check if the dimensions are preserved
        for split in splits:
            assert list(split.dims) == list(norm_data_from_arrays.dims)


def test_chunk(norm_data_from_arrays):
    chunks = norm_data_from_arrays.chunk(n_chunks=2)
    for i, chunk in enumerate(chunks):
        assert chunk.response_vars == ['response_var_{}'.format(i)]

