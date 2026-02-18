"""Tests for data model module."""

import numpy as np
import pytest
import jax.numpy as jnp
from pta_anisotropy import data_model, constants, geometry


class TestDataModel:
    """Test data model functions."""

    def test_get_s_I(self):
        """Test signal projection."""
        n_pulsars = 3
        n_pixels = 5
        n_freqs = 2

        h_p = np.random.randn(n_pixels, n_freqs) + 1j * np.random.randn(n_pixels, n_freqs)
        h_c = np.random.randn(n_pixels, n_freqs) + 1j * np.random.randn(n_pixels, n_freqs)

        R_p = np.random.randn(n_pulsars, n_pixels, n_freqs) + 1j * np.random.randn(n_pulsars, n_pixels, n_freqs)
        R_c = np.random.randn(n_pulsars, n_pixels, n_freqs) + 1j * np.random.randn(n_pulsars, n_pixels, n_freqs)

        s_I = data_model.get_s_I(
            jnp.array(h_p), jnp.array(h_c),
            jnp.array(R_p), jnp.array(R_c)
        )

        assert s_I.shape == (n_pulsars, n_freqs)
        assert np.all(np.isfinite(s_I))

    def test_get_s_I_fi_baseline(self):
        """Test baseline frequency integration."""
        Tspan = 16.0 * constants.yr
        n_pulsars = 3
        n_freqs_fi = 3
        n_freqs_ff = 5

        fi = jnp.arange(1, n_freqs_fi + 1) / Tspan
        ff = jnp.arange(0.5, n_freqs_fi + 1, step=0.1) / Tspan

        h_p = np.random.randn(10, n_freqs_ff) + 1j * np.random.randn(10, n_freqs_ff)
        h_c = np.random.randn(10, n_freqs_ff) + 1j * np.random.randn(10, n_freqs_ff)

        R_p = np.random.randn(n_pulsars, 10, n_freqs_ff) + 1j * np.random.randn(n_pulsars, 10, n_freqs_ff)
        R_c = np.random.randn(n_pulsars, 10, n_freqs_ff) + 1j * np.random.randn(n_pulsars, 10, n_freqs_ff)

        s_I = data_model.get_s_I_fi_baseline(
            Tspan, fi, ff,
            jnp.array(h_p), jnp.array(h_c),
            jnp.array(R_p), jnp.array(R_c)
        )

        assert s_I.shape == (n_pulsars, n_freqs_fi)
        assert np.all(np.isfinite(s_I))

    def test_get_s_I_fi_windowed(self):
        """Test windowed frequency integration."""
        Tspan = 16.0 * constants.yr
        n_pulsars = 3
        n_freqs_fi = 3
        n_freqs_ff = 5

        fi = jnp.arange(1, n_freqs_fi + 1) / Tspan
        ff = jnp.arange(0.5, n_freqs_fi + 1, step=0.1) / Tspan

        h_p = np.random.randn(10, n_freqs_ff) + 1j * np.random.randn(10, n_freqs_ff)
        h_c = np.random.randn(10, n_freqs_ff) + 1j * np.random.randn(10, n_freqs_ff)

        R_p = np.random.randn(n_pulsars, 10, n_freqs_ff) + 1j * np.random.randn(n_pulsars, 10, n_freqs_ff)
        R_c = np.random.randn(n_pulsars, 10, n_freqs_ff) + 1j * np.random.randn(n_pulsars, 10, n_freqs_ff)

        s_I = data_model.get_s_I_fi_windowed(
            Tspan, fi, ff,
            jnp.array(h_p), jnp.array(h_c),
            jnp.array(R_p), jnp.array(R_c)
        )

        assert s_I.shape == (n_pulsars, n_freqs_fi)
        assert np.all(np.isfinite(s_I))

    def test_get_D_IJ_fifj_baseline(self):
        """Test baseline covariance matrix computation."""
        Tspan = 16.0 * constants.yr
        n_pulsars = 3
        n_freqs_fi = 3
        n_freqs_ff = 5

        fi = jnp.arange(1, n_freqs_fi + 1) / Tspan
        ff = jnp.arange(0.5, n_freqs_fi + 1, step=0.1) / Tspan

        h_tilde = np.random.randn(2, 10, n_freqs_ff) + 1j * np.random.randn(2, 10, n_freqs_ff)
        distances = np.random.rand(n_pulsars) * 1e18

        theta_p = np.random.rand(n_pulsars) * np.pi
        phi_p = np.random.rand(n_pulsars) * 2 * np.pi
        p_vec = geometry.unit_vector(theta_p, phi_p)

        theta_k = np.random.rand(10) * np.pi
        phi_k = np.random.rand(10) * 2 * np.pi

        D_IJ = data_model.get_D_IJ_fifj_baseline(
            Tspan, fi, ff,
            jnp.array(h_tilde),
            jnp.array(distances),
            p_vec,
            theta_k,
            phi_k
        )

        assert D_IJ.shape == (n_freqs_fi, n_freqs_fi, n_pulsars, n_pulsars)
        assert np.all(np.isfinite(D_IJ))
        assert np.all(np.isreal(D_IJ))  # Should be real

    def test_get_D_IJ_fifj_windowed(self):
        """Test windowed covariance matrix computation."""
        Tspan = 16.0 * constants.yr
        n_pulsars = 3
        n_freqs_fi = 3
        n_freqs_ff = 5

        fi = jnp.arange(1, n_freqs_fi + 1) / Tspan
        ff = jnp.arange(0.5, n_freqs_fi + 1, step=0.1) / Tspan

        h_tilde = np.random.randn(2, 10, n_freqs_ff) + 1j * np.random.randn(2, 10, n_freqs_ff)
        distances = np.random.rand(n_pulsars) * 1e18

        theta_p = np.random.rand(n_pulsars) * np.pi
        phi_p = np.random.rand(n_pulsars) * 2 * np.pi
        p_vec = geometry.unit_vector(theta_p, phi_p)

        theta_k = np.random.rand(10) * np.pi
        phi_k = np.random.rand(10) * 2 * np.pi

        D_IJ = data_model.get_D_IJ_fifj_windowed(
            Tspan, fi, ff,
            jnp.array(h_tilde),
            jnp.array(distances),
            p_vec,
            theta_k,
            phi_k
        )

        assert D_IJ.shape == (n_freqs_fi, n_freqs_fi, n_pulsars, n_pulsars)
        assert np.all(np.isfinite(D_IJ))
        assert np.all(np.isreal(D_IJ))  # Should be real

    def test_normalization_baseline(self):
        """Test baseline normalization computation."""
        Tspan = 16.0 * constants.yr
        n_freqs_fi = 3
        n_freqs_ff = 5
        Nside = 8

        fi = jnp.arange(1, n_freqs_fi + 1) / Tspan
        ff = jnp.arange(0.5, n_freqs_fi + 1, step=0.1) / Tspan

        H_p_ff = np.random.rand(12 * Nside * Nside, n_freqs_ff)

        C_ff = data_model.get_D_IJ_fifj_normalization_baseline(
            Tspan, fi, ff, jnp.array(H_p_ff)
        )

        assert C_ff.shape == (n_freqs_fi, n_freqs_fi)
        assert np.all(np.isfinite(C_ff))
        assert np.all(np.isreal(C_ff))

    def test_normalization_windowed(self):
        """Test windowed normalization computation."""
        Tspan = 16.0 * constants.yr
        n_freqs_fi = 3
        n_freqs_ff = 5
        Nside = 8

        fi = jnp.arange(1, n_freqs_fi + 1) / Tspan
        ff = jnp.arange(0.5, n_freqs_fi + 1, step=0.1) / Tspan

        H_p_ff = np.random.rand(12 * Nside * Nside, n_freqs_ff)

        C_ff = data_model.get_D_IJ_fifj_normalization_windowed(
            Tspan, fi, ff, jnp.array(H_p_ff)
        )

        assert C_ff.shape == (n_freqs_fi, n_freqs_fi)
        assert np.all(np.isfinite(C_ff))
        assert np.all(np.isreal(C_ff))
