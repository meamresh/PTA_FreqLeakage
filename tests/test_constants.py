"""Tests for constants module."""

import os
import tempfile
import pytest
from pta_anisotropy import constants


class TestConstants:
    """Test physical constants."""

    def test_constants_exist(self):
        """Test that all constants are defined."""
        assert hasattr(constants, "Hubble_over_h")
        assert hasattr(constants, "hour")
        assert hasattr(constants, "day")
        assert hasattr(constants, "yr")
        assert hasattr(constants, "f_yr")
        assert hasattr(constants, "light_speed")
        assert hasattr(constants, "parsec")
        assert hasattr(constants, "Mpc")

    def test_time_constants(self):
        """Test time-related constants."""
        assert constants.hour == 3600
        assert constants.day == 24 * constants.hour
        assert constants.yr == 365.25 * constants.day
        assert abs(constants.f_yr - 1 / constants.yr) < 1e-10

    def test_physical_constants(self):
        """Test physical constants have reasonable values."""
        assert constants.light_speed > 0
        assert constants.parsec > 0
        assert constants.Mpc == 1e6 * constants.parsec
        assert constants.Hubble_over_h > 0

    def test_get_output_directory(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = "test_output"
            result = constants.get_output_directory(tmpdir, subdir)
            expected = os.path.join(tmpdir, subdir)
            assert result == expected
            assert os.path.exists(expected)
            assert os.path.isdir(expected)

    def test_get_output_directory_existing(self):
        """Test output directory with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = "test_output"
            # Create directory first
            os.makedirs(os.path.join(tmpdir, subdir))
            result = constants.get_output_directory(tmpdir, subdir)
            expected = os.path.join(tmpdir, subdir)
            assert result == expected
            assert os.path.exists(expected)
