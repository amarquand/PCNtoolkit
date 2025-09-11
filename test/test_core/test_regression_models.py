"""Tests for regression models."""

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.likelihood import NormalLikelihood
from pcntoolkit.math_functions.prior import (
    LinearPrior,
    Prior,
    RandomPrior,
    make_prior,
)
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR  # noqa: F401, F403
from test.fixtures.data_fixtures import *  # noqa: F401, F403
from test.fixtures.hbr_model_fixtures import *  # noqa: F401, F403
from test.fixtures.norm_data_fixtures import *  # noqa: F401, F403
from test.fixtures.path_fixtures import *


class TestBLR:
    """Test Bayesian Linear Regression model."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data):
        """Setup test environment."""
        self.data = synthetic_data

        # Create NormData object
        self.norm_data = NormData.from_ndarrays(
            "test_data", self.data["covariates"], self.data["responses"], self.data["batch_effects"]
        )
        self.unique_batch_effects = {
            k: np.unique(self.norm_data.batch_effects.sel(batch_effect_dims=k).to_numpy())
            for k in self.norm_data.batch_effect_dims.to_numpy()
        }
        self.unique_batch_effects = {k: list(v) for k, v in self.unique_batch_effects.items()}
        self.be_maps = {
            be: {k: i for i, k in enumerate(self.unique_batch_effects[be])} for be in self.unique_batch_effects.keys()
        }
        self.mapped_batch_effects = self.norm_data.batch_effects.copy()
        for i in range(self.norm_data.batch_effects.shape[1]):
            for j in range(self.norm_data.batch_effects.shape[0]):
                self.mapped_batch_effects[j, i] = self.be_maps[f"batch_effect_{i}"][
                    str(self.norm_data.batch_effects[j, i].to_numpy().item())
                ]

        # Create BLR model
        self.model = BLR("test_blr")

    @pytest.mark.parametrize("n_iter,tol,ard", [(100, 1e-3, False), (1, 1e-6, True)])
    def test_model_creation(self, n_iter, tol, ard):
        """Test creating BLR model from arguments."""
        args = {"n_iter": n_iter, "tol": tol, "ard": ard}
        model = BLR.from_args("test_blr", args)

        # Check model parameters
        assert model.n_iter == n_iter
        assert model.tol == tol
        assert model.ard == ard
        assert model.optimizer == "l-bfgs-b"
        assert model.l_bfgs_b_l == 0.1
        assert model.l_bfgs_b_epsilon == 0.1
        assert model.l_bfgs_b_norm == "l2"

        # Test serialization
        model_dict = model.to_dict()
        assert model_dict["n_iter"] == n_iter
        assert model_dict["tol"] == tol
        assert model_dict["ard"] == ard
        assert model_dict["optimizer"] == "l-bfgs-b"
        assert model_dict["l_bfgs_b_l"] == 0.1
        assert model_dict["l_bfgs_b_epsilon"] == 0.1
        assert model_dict["l_bfgs_b_norm"] == "l2"

        # Test deserialization
        loaded_model = BLR.from_dict(model_dict)
        assert loaded_model.n_iter == n_iter
        assert loaded_model.tol == tol
        assert loaded_model.ard == ard
        assert loaded_model.optimizer == "l-bfgs-b"
        assert loaded_model.l_bfgs_b_l == 0.1
        assert loaded_model.l_bfgs_b_epsilon == 0.1
        assert loaded_model.l_bfgs_b_norm == "l2"

    def test_model_fit(self):
        """Test model fitting."""
        # Extract data for first response variable
        response_var = self.norm_data.response_vars[0]
        data = self.norm_data.sel(response_vars=response_var)
        X = data.X
        Y = data.Y
        be = data.batch_effects
        # Fit model
        self.model.fit(X, self.mapped_batch_effects, self.be_maps, Y)
        assert self.model.is_fitted

    def test_forward_backward(self):
        """Test forward and backward passes."""
        # Extract data for first response variable
        response_var = self.norm_data.response_vars[0]
        data = self.norm_data.sel(response_vars=response_var)
        X = data.X
        Y = data.Y
        be = data.batch_effects
        # Fit model
        self.model.fit(X, self.mapped_batch_effects, self.be_maps, Y)

        # Test forward pass
        Z = self.model.forward(X, be, Y)
        assert Z.shape == Y.shape

        # Test backward pass
        Y_prime = self.model.backward(X, be, Z)
        assert Y_prime.shape == Y.shape
        assert np.allclose(Y_prime, Y)


class TestHBR:
    """Test Hierarchical Bayesian Regression model."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data):
        """Setup test environment."""
        self.data = synthetic_data

        # Create NormData object
        self.norm_data = NormData.from_ndarrays(
            "test_data", self.data["covariates"], self.data["responses"], self.data["batch_effects"]
        )

        # Create HBR model
        self.model = HBR("test_hbr")

    @pytest.mark.parametrize(
        "args",
        [
            {"likelihood": "Normal", "linear_mu": False, "random_mu": False},
            {"likelihood": "Normal", "linear_mu": False, "random_mu": True},
            {"likelihood": "Normal", "linear_mu": True, "random_slope_mu": False, "random_intercept_mu": False},
            {"likelihood": "Normal", "linear_mu": True, "random_slope_mu": True, "random_intercept_mu": False},
            {"likelihood": "Normal", "linear_mu": True, "random_slope_mu": True, "random_intercept_mu": True},
        ],
    )
    def test_model_creation(self, args):
        """Test creating HBR model from arguments."""
        # Create model
        model = HBR.from_args("test_hbr", args)

        # Check model parameters
        assert isinstance(model.likelihood, NormalLikelihood)
        if args.get("linear_mu", False):
            assert isinstance(model.likelihood.mu, LinearPrior)
            if args.get("random_slope_mu", False):
                assert isinstance(model.likelihood.mu.slope, RandomPrior)
            if args.get("random_intercept_mu", False):
                assert isinstance(model.likelihood.mu.intercept, RandomPrior)

        # Test serialization
        model_dict = model.to_dict()
        assert model_dict["likelihood"]["name"] == "Normal"
        if args.get("linear_mu", False):
            assert model_dict["likelihood"]["mu"]["type"] == "LinearPrior"
            if args.get("random_slope_mu", False):
                assert model_dict["likelihood"]["mu"]["slope"]["type"] == "RandomPrior"
            if args.get("random_intercept_mu", False):
                assert model_dict["likelihood"]["mu"]["intercept"]["type"] == "RandomPrior"

        # Test deserialization
        loaded_model = HBR.from_dict(model_dict)
        assert isinstance(loaded_model.likelihood, NormalLikelihood)
        if args.get("linear_mu", False):
            assert isinstance(loaded_model.likelihood.mu, LinearPrior)
            if args.get("random_slope_mu", False):
                assert isinstance(loaded_model.likelihood.mu.slope, RandomPrior)
            if args.get("random_intercept_mu", False):
                assert isinstance(loaded_model.likelihood.mu.intercept, RandomPrior)

    def test_prior_creation(self):
        """Test creating different types of priors."""
        # Test fixed prior
        prior = make_prior("fixed", dist_name="Normal", dims=("mu_covariates",))
        assert isinstance(prior, Prior)
        assert prior.name == "fixed"
        assert prior.dims == ("mu_covariates",)
        assert prior.dist_name == "Normal"
        assert prior.dist_params == (0, 10)

        # Test random prior
        prior = make_prior("random", random=True)
        assert isinstance(prior, RandomPrior)
        assert prior.name == "random"
        assert prior.dims is None
        assert isinstance(prior.mu, Prior)
        assert isinstance(prior.sigma, Prior)

        # Test linear prior
        prior = make_prior("linear", linear=True)
        assert isinstance(prior, LinearPrior)
        assert prior.name == "linear"
        assert prior.dims is None
        assert isinstance(prior.intercept, Prior)
        assert isinstance(prior.slope, Prior)

        # Test linear prior with random components
        prior = make_prior(
            "linear_random", linear=True, intercept=make_prior("random", random=True), slope=make_prior("random", random=True)
        )
        assert isinstance(prior, LinearPrior)
        assert prior.name == "linear_random"
        assert prior.dims is None
        assert isinstance(prior.intercept, RandomPrior)
        assert isinstance(prior.slope, RandomPrior)
