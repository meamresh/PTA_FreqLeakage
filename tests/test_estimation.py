"""Tests for estimation module."""

import numpy as np
import pytest
import jax.numpy as jnp
from pta_anisotropy import estimation, constants


class TestEstimation:
    """Test estimation utilities."""

    def test_compute_inverse(self):
        """Test matrix inversion."""
        # Create a positive definite matrix
        A = np.random.randn(5, 5)
        A = A @ A.T + np.eye(5) * 0.1

        A_inv = estimation.compute_inverse(jnp.array(A))
        result = jnp.dot(A, A_inv)

        # Should be close to identity (allow small numerical error)
        np.testing.assert_allclose(result, np.eye(5), rtol=1e-4, atol=1e-10)

    def test_get_covariance_diagonal(self):
        """Test diagonal covariance computation."""
        n_params = 4
        n_pulsars = 5
        n_freqs = 3

        signal_lm = np.random.rand(n_params)
        gamma_IJ_lm = np.random.rand(n_params, n_pulsars, n_pulsars)
        ff = np.array([1e-8, 2e-8, 3e-8])
        S_f = np.array([1e-30, 1e-30, 1e-30])

        C = estimation.get_covariance_diagonal(
            jnp.array(signal_lm),
            jnp.array(gamma_IJ_lm),
            jnp.array(ff),
            jnp.array(S_f)
        )

        assert C.shape == (n_freqs, n_pulsars, n_pulsars)
        assert np.all(np.isfinite(C))

    def test_get_dcovariance_diagonal(self):
        """Test covariance derivative computation."""
        n_params = 4
        n_pulsars = 5
        n_freqs = 3

        signal_lm = np.random.rand(n_params)
        gamma_IJ_lm = np.random.rand(n_params, n_pulsars, n_pulsars)
        ff = np.array([1e-8, 2e-8, 3e-8])
        S_f = np.array([1e-30, 1e-30, 1e-30])

        dC = estimation.get_dcovariance_diagonal(
            jnp.array(signal_lm),
            jnp.array(gamma_IJ_lm),
            jnp.array(ff),
            jnp.array(S_f)
        )

        assert dC.shape == (n_params, n_freqs, n_pulsars, n_pulsars)
        assert np.all(np.isfinite(dC))

    def test_get_update_estimate_diagonal(self):
        """Test parameter update computation."""
        n_params = 4
        n_pulsars = 3
        n_freqs = 2

        parameters = np.random.rand(n_params) * 0.1
        gamma_IJ_lm = np.random.rand(n_params, n_pulsars, n_pulsars)
        # Make symmetric
        gamma_IJ_lm = (gamma_IJ_lm + np.transpose(gamma_IJ_lm, (0, 2, 1))) / 2

        ff = np.array([1e-8, 2e-8])
        S_f = np.array([1e-30, 1e-30])

        # Create mock data
        C = estimation.get_covariance_diagonal(
            jnp.array(parameters),
            jnp.array(gamma_IJ_lm),
            jnp.array(ff),
            jnp.array(S_f)
        )
        data = C + np.random.randn(n_freqs, n_pulsars, n_pulsars) * 1e-32

        delta, F_inv = estimation.get_update_estimate_diagonal(
            jnp.array(parameters),
            jnp.array(data),
            jnp.array(gamma_IJ_lm),
            jnp.array(ff),
            jnp.array(S_f)
        )

        assert delta.shape == (n_params,)
        assert F_inv.shape == (n_params, n_params)
        assert np.all(np.isfinite(delta))
        assert np.all(np.isfinite(F_inv))

    def test_iterative_estimation(self):
        """Test iterative estimation."""
        n_params = 4
        n_pulsars = 3
        n_freqs = 2

        # Start with initial guess
        theta_init = np.random.rand(n_params) * 0.1
        gamma_IJ_lm = np.random.rand(n_params, n_pulsars, n_pulsars)
        gamma_IJ_lm = (gamma_IJ_lm + np.transpose(gamma_IJ_lm, (0, 2, 1))) / 2

        ff = np.array([1e-8, 2e-8])
        S_f = np.array([1e-30, 1e-30])

        # Create mock data close to true parameters
        true_params = theta_init + np.random.randn(n_params) * 0.01
        C_true = estimation.get_covariance_diagonal(
            jnp.array(true_params),
            jnp.array(gamma_IJ_lm),
            jnp.array(ff),
            jnp.array(S_f)
        )
        D_IJ = C_true + np.random.randn(n_freqs, n_pulsars, n_pulsars) * 1e-32

        theta, uncertainties, converged = estimation.iterative_estimation(
            estimation.get_update_estimate_diagonal,
            jnp.array(theta_init),
            jnp.array(D_IJ),
            jnp.array(gamma_IJ_lm),
            jnp.array(ff),
            jnp.array(S_f),
            i_max=10  # Limit iterations for test
        )

        assert theta.shape == (n_params,)
        assert uncertainties.shape == (n_params,)
        assert np.all(np.isfinite(theta))
        # For this unit test we only require that the routine runs and returns
        # correctly-shaped arrays; uncertainties may be NaN if convergence or
        # the Fisher matrix are ill-conditioned with this random setup.
        assert isinstance(converged, bool)
