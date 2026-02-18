"""Tests for spherical harmonics module."""

import numpy as np
import pytest
from pta_anisotropy import spherical


class TestSphericalHarmonics:
    """Test spherical harmonics utilities."""

    def test_get_n_coefficients_real(self):
        """Test coefficient count calculation."""
        assert spherical.get_n_coefficients_real(0) == 1
        assert spherical.get_n_coefficients_real(1) == 4
        assert spherical.get_n_coefficients_real(2) == 9
        assert spherical.get_n_coefficients_real(3) == 16

    def test_get_l_max_real(self):
        """Test l_max extraction from real coefficients."""
        # l_max=0: 1 coefficient
        coeffs = np.zeros(1)
        assert spherical.get_l_max_real(coeffs) == 0

        # l_max=1: 4 coefficients
        coeffs = np.zeros(4)
        assert spherical.get_l_max_real(coeffs) == 1

        # l_max=2: 9 coefficients
        coeffs = np.zeros(9)
        assert spherical.get_l_max_real(coeffs) == 2

    def test_get_l_max_complex(self):
        """Test l_max extraction from complex coefficients."""
        # l_max=0: 1 coefficient
        coeffs = np.zeros(1, dtype=complex)
        assert spherical.get_l_max_complex(coeffs) == 0

        # l_max=1: 3 coefficients (m=-1,0,1)
        coeffs = np.zeros(3, dtype=complex)
        assert spherical.get_l_max_complex(coeffs) == 1

    def test_real_complex_conversion(self):
        """Test round-trip conversion between real and complex."""
        # Test with l_max=1
        l_max = 1
        n_real = spherical.get_n_coefficients_real(l_max)
        real_coeffs = np.random.randn(n_real)

        # Convert to complex and back
        complex_coeffs = spherical.real_to_complex_conversion(real_coeffs)
        real_coeffs_back = spherical.complex_to_real_conversion(complex_coeffs)

        # Should recover original (within numerical precision)
        np.testing.assert_allclose(real_coeffs, real_coeffs_back, rtol=1e-10)

    def test_get_map_from_real_clms(self):
        """Test HEALPix map generation."""
        Nside = 8
        l_max = 1
        n_coeffs = spherical.get_n_coefficients_real(l_max)
        clms_real = np.zeros(n_coeffs)
        clms_real[0] = 1.0 / np.sqrt(4 * np.pi)  # Monopole

        my_map = spherical.get_map_from_real_clms(clms_real, Nside, l_max=l_max)
        assert my_map.shape == (12 * Nside * Nside,)
        assert np.all(np.isfinite(my_map))

    def test_get_sort_indexes(self):
        """Test sorting index generation."""
        l_max = 1
        l_grid, m_grid, ll, mm, sort_indexes = spherical.get_sort_indexes(l_max)
        assert len(sort_indexes) == spherical.get_n_coefficients_real(l_max)
        assert len(ll) == len(mm)
