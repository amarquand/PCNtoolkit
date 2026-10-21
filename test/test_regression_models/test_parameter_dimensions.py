"""Small tests for fitted-model structural parameter dimensions.

These dimensions are not automatically suitable for a BIC penalty.
"""

import numpy as np
import pymc as pm
import pytest

from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


def test_blr_parameter_dimensions_and_roundtrip():
    model = BLR("demo")
    with pytest.raises(ValueError, match="fitted BLR"):
        model.n_parameters()

    # m has one coefficient per column of the expanded mean design matrix.
    # hyp holds the optimized hyperparameters, including warp/noise/ARD terms.
    model.m = np.zeros(5)
    model.hyp = np.zeros(8)
    model.D = 5
    model.is_fitted = True
    assert model.parameter_dimensions() == {"coefficients": 5, "hyperparameters": 8}
    assert model.n_parameters() == 13

    loaded = BLR.from_dict(model.to_dict())
    assert loaded.parameter_dimensions() == model.parameter_dimensions()


def test_blr_fitted_dimensions_match_stored_arrays(fitted_blr_model):
    """A fitted model already stores the actual expanded dimensionality."""
    counts = fitted_blr_model.parameter_dimensions()
    assert counts["coefficients"] == fitted_blr_model.D
    assert counts["hyperparameters"] == len(fitted_blr_model.hyp)
    assert fitted_blr_model.n_parameters() == sum(counts.values())


def test_hbr_count_includes_only_independent_free_parameters():
    with pm.Model(coords={"site": ["a", "b", "c"]}) as pymc_model:
        pm.Normal("coefficients", shape=4)
        pm.HalfNormal("sigma")
        offsets = pm.ZeroSumNormal("site_offsets", dims="site")
        pm.Deterministic("derived_offsets", 2 * offsets)

    model = HBR("demo")
    with pytest.raises(ValueError, match="fitted HBR"):
        model.n_parameters()
    model.is_fitted = True
    model.pymc_model = pymc_model

    # Four coefficients, one positive sigma and two free site effects:
    # the third site effect is fixed by the zero-sum constraint.
    assert model.n_parameters() == 7
    assert model.n_parameters() == sum(model.parameter_dimensions().values())
    assert not any("derived_offsets" in name for name in model.parameter_dimensions())

    # A real saved HBR has no compiled PyMC model when loaded. The count is
    # retained in its JSON model metadata.
    loaded = HBR.from_dict(model.to_dict())
    assert loaded.pymc_model is None
    assert loaded.parameter_dimensions() == model.parameter_dimensions()
    assert loaded.n_parameters() == 7
