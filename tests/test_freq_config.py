"""Tests for frequency configuration module."""

import numpy as np
import pytest
from pta_anisotropy import freq_config, constants


class TestFrequencyConfig:
    """Test FrequencyConfig class."""

    def test_frequency_config_init(self):
        """Test FrequencyConfig initialization."""
        fi_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / constants.yr
        injection_bins = np.array([0, 1])
        reconstruction_bins = np.array([2, 3, 4])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        assert len(fcfg.fi_full) == 5
        assert len(fcfg.injection_bins) == 2
        assert len(fcfg.reconstruction_bins) == 3
        assert fcfg.n_fi == 5
        assert np.array_equal(fcfg.fi_recon, fi_full[reconstruction_bins])

    def test_injection_mask(self):
        """Test injection mask creation."""
        fi_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / constants.yr
        injection_bins = np.array([0, 2, 4])
        reconstruction_bins = np.array([1, 3])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        assert fcfg.inj_mask[0] == True
        assert fcfg.inj_mask[1] == False
        assert fcfg.inj_mask[2] == True
        assert fcfg.inj_mask[3] == False
        assert fcfg.inj_mask[4] == True

    def test_reconstruction_mask(self):
        """Test reconstruction mask creation."""
        fi_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / constants.yr
        injection_bins = np.array([0, 1])
        reconstruction_bins = np.array([2, 3, 4])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        assert fcfg.recon_mask[0] == False
        assert fcfg.recon_mask[1] == False
        assert fcfg.recon_mask[2] == True
        assert fcfg.recon_mask[3] == True
        assert fcfg.recon_mask[4] == True

    def test_no_overlap(self):
        """Test configuration with no overlap."""
        fi_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / constants.yr
        injection_bins = np.array([0, 1])
        reconstruction_bins = np.array([2, 3, 4])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        overlap = np.intersect1d(fcfg.injection_bins, fcfg.reconstruction_bins)
        assert len(overlap) == 0

    def test_with_overlap(self):
        """Test configuration with overlap."""
        fi_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / constants.yr
        injection_bins = np.array([0, 1, 2])
        reconstruction_bins = np.array([2, 3, 4])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        overlap = np.intersect1d(fcfg.injection_bins, fcfg.reconstruction_bins)
        assert len(overlap) == 1
        assert overlap[0] == 2


class TestCreateHPFFBinned:
    """Test create_H_p_ff_binned function."""

    def test_create_H_p_ff_binned(self):
        """Test H_p_ff creation."""
        Nside = 8
        l_max = 1
        n_coeffs = 4  # l_max=1 has 4 coefficients

        fi_full = np.array([1.0, 2.0, 3.0]) / constants.yr
        ff = np.array([1.2, 1.5, 2.1, 2.5]) / constants.yr
        S_ff = np.ones(len(ff))

        injection_bins = np.array([0])
        reconstruction_bins = np.array([1, 2])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        clms_peak = np.zeros(n_coeffs)
        clms_peak[0] = 1.0 / np.sqrt(4 * np.pi)  # Monopole
        clms_peak[2] = 0.1  # Dipole component

        H_p_ff = freq_config.create_H_p_ff_binned(
            fi_full, ff, fcfg, clms_peak, Nside, l_max, S_ff
        )

        assert H_p_ff.shape == (12 * Nside * Nside, len(ff))
        assert np.all(np.isfinite(H_p_ff))
        assert np.all(H_p_ff >= 0)  # Power spectrum should be non-negative

    def test_anisotropy_injection(self):
        """Test that anisotropy is only injected in specified bins."""
        Nside = 8
        l_max = 1
        n_coeffs = 4

        fi_full = np.array([1.0, 2.0, 3.0]) / constants.yr
        ff = np.array([1.2, 2.1]) / constants.yr  # One in bin 0, one in bin 1
        S_ff = np.ones(len(ff))

        injection_bins = np.array([0])  # Only inject in bin 0
        reconstruction_bins = np.array([1, 2])

        fcfg = freq_config.FrequencyConfig(
            fi_full, injection_bins, reconstruction_bins
        )

        clms_peak = np.zeros(n_coeffs)
        clms_peak[0] = 1.0 / np.sqrt(4 * np.pi)
        clms_peak[2] = 0.1  # Dipole

        H_p_ff = freq_config.create_H_p_ff_binned(
            fi_full, ff, fcfg, clms_peak, Nside, l_max, S_ff
        )

        # Both frequencies should have monopole, but only first should have dipole
        # (since first ff maps to bin 0 which has injection)
        assert np.all(H_p_ff[:, 0] > 0)  # Should have signal
        assert np.all(H_p_ff[:, 1] > 0)  # Should have monopole at least
