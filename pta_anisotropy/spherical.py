"""
Spherical-harmonic utilities and real/complex conversions.
"""

import numpy as np
import scipy

from .constants import jnp  # type: ignore[attr-defined]


def compare_versions(version1, version2):
    """Compare two version strings."""
    v1_parts = [int(x) for x in version1.split(".")]
    v2_parts = [int(x) for x in version2.split(".")]
    for i in range(max(len(v1_parts), len(v2_parts))):
        v1 = v1_parts[i] if i < len(v1_parts) else 0
        v2 = v2_parts[i] if i < len(v2_parts) else 0
        if v1 > v2:
            return True
        elif v1 < v2:
            return False
    return True


if compare_versions(scipy.__version__, "1.15.0"):
    from scipy.special import sph_harm_y
else:
    from scipy.special import sph_harm

    def sph_harm_y(ell, m, theta, phi):
        return sph_harm(m, ell, phi, theta)


def get_l_max_real(real_spherical_harmonics):
    """Get maximum ell value from real spherical harmonics coefficients."""
    return int(np.sqrt(len(real_spherical_harmonics)) - 1)


def get_l_max_complex(complex_spherical_harmonics):
    """Get maximum ell value from complex spherical harmonics coefficients."""
    return int(np.sqrt(1.0 + 8.0 * len(complex_spherical_harmonics)) / 2 - 1.5)


def get_n_coefficients_real(l_max):
    """Get number of real spherical harmonic coefficients."""
    return int((l_max + 1) ** 2)


def get_sort_indexes(l_max):
    """Get sorting indexes for spherical harmonics."""
    l_values = np.arange(l_max + 1)
    m_values = np.arange(l_max + 1)
    l_grid, m_grid = np.meshgrid(l_values, m_values, indexing="xy")
    l_flat = l_grid.flatten()
    m_flat = m_grid.flatten()
    l_grid = l_flat[np.abs(m_flat) <= l_flat]
    m_grid = m_flat[np.abs(m_flat) <= l_flat]
    mm = np.append(-np.flip(m_grid[m_grid > 0]), m_grid)
    ll = np.append(np.flip(l_grid[m_grid > 0]), l_grid)
    return l_grid, m_grid, ll, mm, np.lexsort((mm, ll))


def complex_to_real_conversion(spherical_harmonics):
    """Convert complex to real spherical harmonics."""
    l_max = get_l_max_complex(spherical_harmonics)
    _, m_grid, _, _, sort_indexes = get_sort_indexes(l_max)
    zero_m = spherical_harmonics[m_grid == 0.0].real
    sign = (-1.0) ** m_grid[m_grid > 0]
    positive_spherical = np.sqrt(2.0) * spherical_harmonics[m_grid > 0.0]
    positive_m = np.einsum("i,i...->i...", sign, positive_spherical.real)
    negative_m = np.einsum("i,i...->i...", sign, positive_spherical.imag)
    all_spherical_harmonics = np.concatenate(
        (np.flip(negative_m, axis=0), zero_m, positive_m), axis=0
    )
    return all_spherical_harmonics[sort_indexes]


def real_to_complex_conversion(real_spherical_harmonics):
    """Convert real to complex spherical harmonics."""
    l_max = get_l_max_real(real_spherical_harmonics)
    _, _, _, mm, sort_indexes = get_sort_indexes(l_max)
    ordered_real_spherical_harmonics = np.zeros_like(real_spherical_harmonics)
    ordered_real_spherical_harmonics[sort_indexes] = real_spherical_harmonics
    zero_m = ordered_real_spherical_harmonics[mm == 0]
    positive_m = ordered_real_spherical_harmonics[mm > 0]
    negative_m = ordered_real_spherical_harmonics[mm < 0]
    m_positive = mm[mm > 0]
    complex_positive_m = (positive_m + 1j * negative_m[::-1]) / (
        np.sqrt(2.0) * (-1.0) ** m_positive
    )
    complex_spherical_harmonics = np.concatenate(
        (zero_m, complex_positive_m), axis=0
    )
    return complex_spherical_harmonics


def get_map_from_real_clms(clms_real, Nside, l_max=None):
    """Get HEALPix map from real spherical harmonic coefficients."""
    import healpy as hp

    l_max = get_l_max_real(clms_real) if l_max is None else l_max
    clms_complex = real_to_complex_conversion(clms_real)
    my_map = hp.alm2map(clms_complex, Nside, lmax=l_max)
    return my_map

