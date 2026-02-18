"""Tests for geometry module."""

import numpy as np
import pytest
from pta_anisotropy import geometry, constants


class TestGeometry:
    """Test geometry utilities."""

    def test_unit_vector(self):
        """Test unit vector computation."""
        theta = np.array([0.0, np.pi / 2, np.pi])
        phi = np.array([0.0, np.pi / 2, np.pi])

        k_vec = geometry.unit_vector(theta, phi)

        assert k_vec.shape == (len(theta), 3)

        # Check normalization
        norms = np.linalg.norm(k_vec, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # Check specific values
        # theta=0, phi=0 should be [0, 0, 1]
        np.testing.assert_allclose(k_vec[0], [0, 0, 1], rtol=1e-10)

    def test_get_u(self):
        """Test u vector (theta derivative)."""
        theta = np.array([0.0, np.pi / 2])
        phi = np.array([0.0, np.pi / 2])

        u = geometry.get_u(theta, phi)
        assert u.shape == (len(theta), 3)
        assert np.all(np.isfinite(u))

    def test_get_v(self):
        """Test v vector (phi derivative)."""
        theta = np.array([0.0, np.pi / 2])
        phi = np.array([0.0, np.pi / 2])

        v = geometry.get_v(theta, phi)
        assert v.shape == (len(theta), 3)
        assert np.all(np.isfinite(v))

    def test_get_plus_cross(self):
        """Test polarization tensor computation."""
        theta = np.array([np.pi / 4])
        phi = np.array([np.pi / 4])

        plus, cross = geometry.get_plus_cross(theta, phi)

        assert plus.shape == (len(theta), 3, 3)
        assert cross.shape == (len(theta), 3, 3)

        # Plus should be symmetric
        np.testing.assert_allclose(plus[0], plus[0].T, rtol=1e-10)

        # Cross is also symmetric for this construction
        # Use absolute tolerance for very small off-diagonal elements
        np.testing.assert_allclose(cross[0], cross[0].T, rtol=1e-8, atol=1e-15)

    def test_get_R_pc(self):
        """Test PTA response function computation."""
        n_pulsars = 5
        n_freqs = 3
        n_pixels = 10

        f_vec = np.array([1e-8, 2e-8, 3e-8])  # Frequencies in Hz
        distances = np.random.rand(n_pulsars) * 1e18  # Distances in meters

        # Generate pulsar vectors
        theta_p = np.random.rand(n_pulsars) * np.pi
        phi_p = np.random.rand(n_pulsars) * 2 * np.pi
        p_vec = geometry.unit_vector(theta_p, phi_p)

        # Generate sky pixels
        theta_k = np.random.rand(n_pixels) * np.pi
        phi_k = np.random.rand(n_pixels) * 2 * np.pi

        R_p, R_c = geometry.get_R_pc(f_vec, distances, p_vec, theta_k, phi_k)

        # get_R_pc currently returns arrays shaped (n_pulsars, n_pixels, n_freqs)
        assert R_p.shape == (n_pulsars, n_pixels, n_freqs)
        assert R_c.shape == (n_pulsars, n_pixels, n_freqs)
        assert np.iscomplexobj(R_p)
        assert np.iscomplexobj(R_c)
        assert np.all(np.isfinite(R_p))
        assert np.all(np.isfinite(R_c))
