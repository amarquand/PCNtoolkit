import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pcntoolkit.model.SHASH import SHASH, C, S, S_inv, SHASHb, SHASHo, SHASHo2


def test_shash_transformations():
    """Test the basic SHASH transformations (S, S_inv, C)"""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    epsilon = 0.5
    delta = 1.5

    # Test S transformation
    s_result = S(x, epsilon, delta)
    # Test inverse relationship
    s_inv_result = S_inv(s_result, epsilon, delta)
    assert_array_almost_equal(x, s_inv_result, decimal=6)

    # Test C transformation
    c_result = C(x, epsilon, delta)
    # C should always be >= 1 since it's sqrt(1 + S^2)
    assert np.all(c_result >= 1.0)
    # Test relationship between S and C
    assert_array_almost_equal(c_result, np.sqrt(1 + s_result**2), decimal=6)


def test_moment_calculations():
    """Test moment calculation functions"""
    epsilon = 0.5
    delta = 1.5

    # Test compute_moments function
    mean, var = SHASH.m1m2(epsilon, delta)
    assert_almost_equal(mean.eval(), SHASH.m1(epsilon, delta).eval(), decimal=6)
    assert_almost_equal(
        var.eval(), SHASH.m2(epsilon, delta).eval() - mean.eval() ** 2, decimal=6
    )


def test_shash_random_generation():
    """Test random number generation for SHASH distributions"""
    rng = np.random.RandomState(42)
    n_samples = 1000

    # Test base SHASH
    epsilon, delta = 0.5, 1.5
    samples = SHASH.rv_op.rng_fn(rng, epsilon, delta, size=n_samples)
    assert samples.shape == (n_samples,)

    # Test SHASHo
    mu, sigma = 1.0, 2.0
    samples_o = SHASHo.rv_op.rng_fn(rng, mu, sigma, epsilon, delta, size=n_samples)
    assert samples_o.shape == (n_samples,)

    # Test SHASHo2
    samples_o2 = SHASHo2.rv_op.rng_fn(rng, mu, sigma, epsilon, delta, size=n_samples)
    assert samples_o2.shape == (n_samples,)

    # Test SHASHb
    samples_b = SHASHb.rv_op.rng_fn(rng, mu, sigma, epsilon, delta, size=n_samples)
    assert samples_b.shape == (n_samples,)


def test_shash_distribution_properties():
    """Test statistical properties of SHASH distributions"""
    rng = np.random.RandomState(42)
    n_samples = 10000
    mu, sigma = 1.0, 2.0
    epsilon, delta = 0.5, 1.5

    # Generate samples from SHASHb
    samples = SHASHb.rv_op.rng_fn(rng, mu, sigma, epsilon, delta, size=n_samples)

    # Check mean and standard deviation are close to specified values
    assert_almost_equal(np.mean(samples), mu, decimal=1)
    assert_almost_equal(np.std(samples), sigma, decimal=1)


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    x = np.array([-1e10, 0.0, 1e10])  # Test very large/small values
    epsilon = 0.0  # Test zero skewness
    delta = 1.0  # Test unit tail weight

    # S transformation should handle extreme values
    s_result = S(x, epsilon, delta)
    assert not np.any(np.isnan(s_result))
    assert not np.any(np.isinf(s_result))

    # C transformation should handle extreme values
    c_result = C(x, epsilon, delta)
    assert not np.any(np.isnan(c_result))
    assert not np.any(np.isinf(c_result))
    assert np.all(c_result >= 1.0)

    # Test moment calculations with edge case parameters
    mean, var = SHASH.m1m2(0.0, 1.0)  # Standard case
    assert not np.isnan(mean.eval())
    assert not np.isnan(var.eval())
    assert var.eval() > 0  # Variance should always be positive
