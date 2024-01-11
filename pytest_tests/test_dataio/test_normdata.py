import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData


@pytest.fixture
def np_arrays():
    X = np.random.randn(1000, 2)
    y = np.random.randn(1000)
    batch_effects = np.concatenate([np.random.choice(
        [0, 1], (1000, 1)), np.random.choice([0, 2, 1], (1000, 1))], axis=1)
    return X, y, batch_effects


@pytest.fixture
def dataframe(np_arrays):
    X, y, batch_effects = np_arrays
    return pd.DataFrame(np.concatenate([X, y.reshape(-1, 1), batch_effects], axis=1), columns=['X1', 'X2', 'y', 'batch1', 'batch2'])


def test_create_data_from_arrays(np_arrays):
    X, y, batch_effects = np_arrays
    norm_data = NormData.from_ndarrays('test', X, y, batch_effects)
    assert norm_data.X.shape == (1000, 2)
    assert norm_data.y.shape == (1000,)
    assert norm_data.batch_effects.shape == (1000, 2)
    assert norm_data.covariates.shape == (2,)
    assert norm_data.batch_effect_dims.shape == (2,)
    assert norm_data.batch_effects_maps['batch_effect_0'] == {0: 0, 1: 1}
    assert norm_data.batch_effects_maps['batch_effect_1'] == {0: 0, 1: 1, 2: 2}
    assert norm_data.coords['datapoints'].shape == (1000,)
    assert norm_data.coords['covariates'].shape == (2,)
    assert norm_data.coords['batch_effect_dims'].shape == (2,)
    assert norm_data.attrs == {}


def test_create_data_from_dataframe(dataframe):
    norm_data = NormData.from_dataframe('test', dataframe, covariates=[
                                        'X1', 'X2'], batch_effects=['batch1', 'batch2'], y='y')
    assert norm_data.X.shape == (1000, 2)
    assert norm_data.y.shape == (1000,)
    assert norm_data.batch_effects.shape == (1000, 2)
    assert norm_data.covariates.shape == (2,)
    assert norm_data.batch_effect_dims.shape == (2,)
    assert norm_data.batch_effects_maps['batch1'] == {0: 0, 1: 1}
    assert norm_data.batch_effects_maps['batch2'] == {0: 0, 1: 1, 2: 2}
    assert norm_data.coords['datapoints'].shape == (1000,)
    assert norm_data.coords['covariates'].to_numpy().tolist() == ['X1', 'X2']
    assert norm_data.coords['batch_effect_dims'].to_numpy().tolist() == [
        'batch1', 'batch2']
    assert norm_data.attrs == {}


def test_split(dataframe):
    norm_data = NormData.from_dataframe('test', dataframe, covariates=[
                                        'X1', 'X2'], batch_effects=['batch1', 'batch2'], y='y')
    splits = norm_data.split(splits=(0.5, 0.5), split_names=('train', 'val'))
    assert len(splits) == 2
    assert splits[0].name == 'train'
    assert splits[1].name == 'val'
    assert splits[0].X.shape == (500, 2)
    assert splits[0].y.shape == (500,)
    assert splits[0].batch_effects.shape == (500, 2)
    assert splits[0].covariates.shape == (2,)
    assert splits[0].batch_effect_dims.shape == (2,)
    assert splits[0].batch_effects_maps['batch1'] == {0: 0, 1: 1}
    assert splits[0].batch_effects_maps['batch2'] == {0: 0, 1: 1, 2: 2}
    assert splits[0].coords['datapoints'].shape == (500,)
    assert splits[0].coords['covariates'].to_numpy().tolist() == ['X1', 'X2']
    assert splits[0].coords['batch_effect_dims'].to_numpy().tolist() == [
        'batch1', 'batch2']
    assert splits[0].attrs == {}
    assert splits[1].X.shape == (500, 2)
    assert splits[1].y.shape == (500,)
    assert splits[1].batch_effects.shape == (500, 2)
    assert splits[1].covariates.shape == (2,)
    assert splits[1].batch_effect_dims.shape == (2,)
    assert splits[1].batch_effects_maps['batch1'] == {0: 0, 1: 1}
    assert splits[1].batch_effects_maps['batch2'] == {0: 0, 1: 1, 2: 2}
    assert splits[1].coords['datapoints'].shape == (500,)
    assert splits[1].coords['covariates'].to_numpy().tolist() == ['X1', 'X2']
    assert splits[1].coords['batch_effect_dims'].to_numpy().tolist() == [
        'batch1', 'batch2']
    assert splits[1].attrs == {}

    original_unique_batch_effects = np.unique(
        norm_data.batch_effects.data, axis=0)
    split1_unique_batch_effects = np.unique(
        splits[0].batch_effects.data, axis=0)
    split2_unique_batch_effects = np.unique(
        splits[1].batch_effects.data, axis=0)
    assert np.all(original_unique_batch_effects == split1_unique_batch_effects)
    assert np.all(original_unique_batch_effects == split2_unique_batch_effects)
