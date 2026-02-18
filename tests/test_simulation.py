"""Tests for simulation module."""

import numpy as np
import pytest
from pta_anisotropy import simulation, constants


class TestSimulation:
    """Test simulation utilities."""

    def test_generate_gaussian(self):
        """Test Gaussian random number generation."""
        size = (10, 20)
        result = simulation.generate_gaussian(0.0, 1.0, size=size)
        assert result.shape == size
        assert np.iscomplexobj(result)

    def test_generate_distance(self):
        """Test distance generation."""
        n_pulsars = 10
        distances = simulation.generate_distance(n_pulsars)
        assert len(distances) == n_pulsars
        assert np.all(distances > 0)

    def test_distances_to_meters(self):
        """Test distance conversion."""
        distances_pc = np.array([100.0, 1000.0, 10000.0])
        distances_m = simulation.distances_to_meters(distances_pc)
        expected = distances_pc * constants.parsec
        np.testing.assert_allclose(distances_m, expected)

    def test_generate_pulsar_sky_and_kpixels(self):
        """Test pulsar sky generation."""
        Np = 10
        Nside = 8

        p_vec, cos_IJ, distances_pc, theta_k, phi_k = simulation.generate_pulsar_sky_and_kpixels(
            Np, Nside
        )

        assert p_vec.shape == (Np, 3)
        assert cos_IJ.shape == (Np, Np)
        assert len(distances_pc) == Np
        assert len(theta_k) == 12 * Nside * Nside
        assert len(phi_k) == 12 * Nside * Nside

        # Check unit vectors
        norms = np.linalg.norm(p_vec, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # Check cosine matrix is symmetric
        np.testing.assert_allclose(cos_IJ, cos_IJ.T, rtol=1e-10)

        # Check cosine values are in [-1, 1]
        assert np.all(cos_IJ >= -1.0)
        assert np.all(cos_IJ <= 1.0)

    def test_generate_hpc_polarization_pixel_frequency(self):
        """Test GW signal generation."""
        Npix = 100
        nfreqs = 5
        H_p_ff = np.random.rand(Npix, nfreqs)

        h_tilde = simulation.generate_hpc_polarization_pixel_frequency(H_p_ff)

        assert h_tilde.shape == (2, Npix, nfreqs)  # Plus and cross polarizations
        assert np.iscomplexobj(h_tilde)
        assert np.all(np.isfinite(h_tilde))
