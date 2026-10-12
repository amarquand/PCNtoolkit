"""Tests for CorrelationMatrix."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import xarray as xr

from pcntoolkit import CorrelationMatrix
from test.test_longitudinal.conftest import make_cross_sectional_norm_data


def _make_matrix(
    n_ages: int = 8,
    response_vars: list[str] | None = None,
    covariate: str = "age",
) -> xr.DataArray:
    """Build a correlation matrix that decays away from the diagonal."""
    response_vars = response_vars or ["roi_a"]
    gap = np.abs(np.arange(n_ages)[:, None] - np.arange(n_ages)[None, :])
    single = np.exp(-gap / 3.0)
    np.fill_diagonal(single, 1.0)

    return xr.DataArray(
        np.stack([single] * len(response_vars)),
        dims=("response_vars", f"{covariate}_1", f"{covariate}_2"),
        coords={
            "response_vars": response_vars,
            f"{covariate}_1": np.arange(n_ages),
            f"{covariate}_2": np.arange(n_ages),
        },
    )


# ------------------------------------------------------------------ #
# CorrelationMatrix: save and load
# ------------------------------------------------------------------ #


def test_save_writes_one_csv_per_response_var(tmp_path):
    corr = CorrelationMatrix(_make_matrix(response_vars=["roi_a", "roi_b"]))
    corr.save(str(tmp_path / "corr"))

    written = sorted(p.name for p in (tmp_path / "corr").iterdir())
    assert written == ["correlation_matrix.json", "roi_a.csv", "roi_b.csv"]


def test_save_records_metadata(tmp_path):
    corr = CorrelationMatrix(
        _make_matrix(),
        bandwidth=3,
        max_correlation=0.95,
        n_subjects=67,
        estimated_range=(2, 5),
    )
    corr.save(str(tmp_path / "corr"))

    with open(tmp_path / "corr" / "correlation_matrix.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["covariate"] == "age"
    assert metadata["response_vars"] == ["roi_a"]
    assert metadata["bandwidth"] == 3
    assert metadata["max_correlation"] == 0.95
    assert metadata["n_subjects"] == 67
    assert metadata["estimated_range"] == [2, 5]


def test_round_trip_preserves_values_exactly(tmp_path):
    """A loaded matrix must score identically to the one that was saved."""
    corr = CorrelationMatrix(_make_matrix(response_vars=["roi_a", "roi_b"]))
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr"))

    assert np.array_equal(corr.matrix.values, loaded.matrix.values)
    assert corr.get("roi_a", 5, 3) == loaded.get("roi_a", 5, 3)


def test_round_trip_preserves_metadata(tmp_path):
    corr = CorrelationMatrix(
        _make_matrix(),
        bandwidth=3,
        max_correlation=0.95,
        n_subjects=67,
        estimated_range=(2, 5),
    )
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr"))

    assert loaded.bandwidth == 3
    assert loaded.max_correlation == 0.95
    assert loaded.n_subjects == 67
    assert loaded.estimated_range == (2, 5)


def test_round_trip_preserves_missing_metadata(tmp_path):
    """Matrices without cohort provenance must not gain any on a round-trip."""
    corr = CorrelationMatrix(_make_matrix(), bandwidth=3)
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr"))

    assert loaded.n_subjects is None
    assert loaded.estimated_range is None


def test_round_trip_preserves_non_age_covariate(tmp_path):
    corr = CorrelationMatrix(
        _make_matrix(covariate="timepoint"), covariate="timepoint"
    )
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr"))

    assert loaded.covariate == "timepoint"
    assert loaded.matrix.dims == ("response_vars", "timepoint_1", "timepoint_2")
    assert np.array_equal(corr.matrix.values, loaded.matrix.values)


def test_save_then_load_is_stable(tmp_path):
    """Saving a loaded matrix again must not drift."""
    corr = CorrelationMatrix(_make_matrix())
    corr.save(str(tmp_path / "one"))
    once = CorrelationMatrix.load(str(tmp_path / "one"))
    once.save(str(tmp_path / "two"))
    twice = CorrelationMatrix.load(str(tmp_path / "two"))

    assert np.array_equal(once.matrix.values, twice.matrix.values)


def test_load_without_metadata_raises(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="not a saved correlation matrix"):
        CorrelationMatrix.load(str(tmp_path / "empty"))


def test_load_with_missing_csv_raises(tmp_path):
    corr = CorrelationMatrix(_make_matrix(response_vars=["roi_a", "roi_b"]))
    corr.save(str(tmp_path / "corr"))
    (tmp_path / "corr" / "roi_b.csv").unlink()

    with pytest.raises(FileNotFoundError, match="roi_b"):
        CorrelationMatrix.load(str(tmp_path / "corr"))


def test_saved_csv_is_readable_without_pcntoolkit(tmp_path):
    """The point of CSV is that the correlations open in any tool."""
    corr = CorrelationMatrix(_make_matrix(n_ages=4))
    corr.save(str(tmp_path / "corr"))

    lines = (tmp_path / "corr" / "roi_a.csv").read_text().splitlines()
    assert lines[0] == "age,0,1,2,3"
    # Diagonal first, then decaying off-diagonal entries.
    assert lines[1].split(",")[1] == "1.0"


# ------------------------------------------------------------------ #
# CorrelationMatrix: compute and get
# ------------------------------------------------------------------ #


def test_correlation_matrix_compute_from_longitudinal_cohort(
    longitudinal_cohort_norm_data,
):
    corr = CorrelationMatrix.compute(
        longitudinal_cohort_norm_data,
        bandwidth=1,
        covariate="age",
    )

    assert corr.n_subjects == 30
    assert corr.estimated_range == (10, 11)
    assert corr.matrix.dims == ("response_vars", "age_1", "age_2")
    assert np.allclose(np.diagonal(corr.matrix.values[0]), 1.0)


def test_correlation_matrix_compute_rejects_cross_sectional(
    predicted_norm_data_factory,
):
    data = make_cross_sectional_norm_data(predicted_norm_data_factory)
    with pytest.raises(
        ValueError, match="single-visit|cross-sectional|multiple visits"
    ):
        CorrelationMatrix.compute(data, bandwidth=1)


def test_correlation_matrix_compute_rejects_missing_z(longitudinal_cohort_norm_data):
    data = longitudinal_cohort_norm_data.drop_vars("Z")
    with pytest.raises(ValueError, match="'Z'"):
        CorrelationMatrix.compute(data, bandwidth=1)


def test_correlation_matrix_invalid_max_correlation(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory()
    with pytest.raises(ValueError, match="max_correlation"):
        CorrelationMatrix(matrix, max_correlation=1.0)


@pytest.mark.parametrize(
    ("age_1", "age_2", "expected_r"),
    [
        (0, 1, 0.8),
        (1, 0, 0.8),
        (-5, 1, 0.8),
        (99, 4, 0.8),
    ],
    ids=["forward", "symmetric", "clamp_low", "clamp_high"],
)
def test_correlation_matrix_get_clamps_ages(
    correlation_matrix,
    age_1,
    age_2,
    expected_r,
):
    assert correlation_matrix.get("metric_a", age_1, age_2) == pytest.approx(expected_r)


def test_correlation_matrix_get_respects_max_correlation(
    correlation_matrix_array_factory,
):
    matrix = correlation_matrix_array_factory(offset_correlations={1: 0.999})
    corr = CorrelationMatrix(matrix, max_correlation=0.5)
    assert corr.get("metric_a", 0, 1) == pytest.approx(0.5)


def test_correlation_matrix_get_warns_outside_estimated_range(correlation_matrix):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlation_matrix.get("metric_a", 99, 100)

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)


def test_correlation_matrix_get_unknown_response_var_raises(correlation_matrix):
    with pytest.raises(KeyError, match="metric_b"):
        correlation_matrix.get("metric_b", 0, 1)
