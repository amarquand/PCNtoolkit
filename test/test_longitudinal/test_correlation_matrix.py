"""Tests for CorrelationMatrix."""

from __future__ import annotations

import warnings
from pathlib import Path

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


def test_round_trip_preserves_values_exactly(tmp_path):
    """A loaded matrix must score identically to the one that was saved."""
    corr = CorrelationMatrix(_make_matrix(response_vars=["roi_a", "roi_b"]))
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr"))

    assert np.array_equal(corr.matrix.values, loaded.matrix.values)
    assert corr.get("roi_a", 5, 3) == loaded.get("roi_a", 5, 3)


def test_round_trip_preserves_non_age_covariate(tmp_path: Path) -> None:
    corr = CorrelationMatrix(_make_matrix(covariate="timepoint"), covariate="timepoint")
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr"), covariate="timepoint")

    assert loaded.covariate == "timepoint"
    assert loaded.matrix.dims == ("response_vars", "timepoint_1", "timepoint_2")


def test_load_single_file_loads_one_response_var(tmp_path: Path) -> None:
    corr = CorrelationMatrix(_make_matrix(response_vars=["roi_a", "roi_b"]))
    corr.save(str(tmp_path / "corr"))
    loaded = CorrelationMatrix.load(str(tmp_path / "corr" / "roi_b.csv"))

    assert list(loaded.matrix.coords["response_vars"].values) == ["roi_b"]
    assert np.array_equal(
        loaded.matrix.values[0], corr.matrix.sel(response_vars="roi_b").values
    )


def test_save_reproduces_shared_csv_byte_for_byte(tmp_path: Path) -> None:
    """A matrix shared as CSV must survive a load and save unchanged, down to
    the last digit and line ending."""
    original = tmp_path / "roi_a.csv"
    original.write_bytes(
        b",0,1,2\n"
        b"0,1.0,0.9764900020268343,0.0\n"
        b"1,0.9764900020268343,1.0,0.9457637571045612\n"
        b"2,0.0,0.9457637571045612,1.0\n"
    )

    CorrelationMatrix.load(str(original)).save(str(tmp_path / "saved"))

    assert (tmp_path / "saved" / "roi_a.csv").read_bytes() == original.read_bytes()


def test_load_empty_folder_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="no .csv files"):
        CorrelationMatrix.load(str(tmp_path / "empty"))


def test_load_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CorrelationMatrix.load(str(tmp_path / "missing"))


def test_load_rejects_csv_that_is_not_a_matrix(tmp_path: Path) -> None:
    """Ages must run 0, 1, ..., N-1, because get() reads cells by position."""
    (tmp_path / "corr").mkdir()
    (tmp_path / "corr" / "roi_a.csv").write_bytes(
        b",5,6,7\n5,1.0,0.5,0.0\n6,0.5,1.0,0.5\n7,0.0,0.5,1.0\n"
    )

    with pytest.raises(ValueError, match="roi_a.csv"):
        CorrelationMatrix.load(str(tmp_path / "corr"))


def test_load_rejects_matrices_of_different_sizes(tmp_path: Path) -> None:
    folder = str(tmp_path / "corr")
    CorrelationMatrix(_make_matrix(n_ages=4, response_vars=["roi_a"])).save(folder)
    CorrelationMatrix(_make_matrix(n_ages=5, response_vars=["roi_b"])).save(folder)

    with pytest.raises(ValueError, match="same covariate values"):
        CorrelationMatrix.load(folder)


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
