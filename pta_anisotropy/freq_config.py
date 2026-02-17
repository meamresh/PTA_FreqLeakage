"""
Frequency-bin configuration and anisotropy injection helpers.
"""

import numpy as np
import healpy as hp

from .spherical import get_map_from_real_clms


class FrequencyConfig:
    """Configuration for frequency-dependent injection and reconstruction."""

    def __init__(self, fi_full, injection_bins, reconstruction_bins):
        """
        Parameters
        ----------
        fi_full : array
            Full array of analysis frequencies.
        injection_bins : array of int
            Indices of fi_full where anisotropy is injected.
        reconstruction_bins : array of int
            Indices of fi_full used for parameter estimation.
        """

        self.fi_full = fi_full
        self.injection_bins = np.array(injection_bins)
        self.reconstruction_bins = np.array(reconstruction_bins)

        # Derived quantities
        self.n_fi = len(fi_full)
        self.inj_mask = np.zeros(self.n_fi, dtype=bool)
        self.inj_mask[injection_bins] = True

        self.recon_mask = np.zeros(self.n_fi, dtype=bool)
        self.recon_mask[reconstruction_bins] = True

        self.fi_recon = fi_full[reconstruction_bins]

    def print_summary(self, yr):
        print("Frequency Configuration:")
        print(f"  Total bins: {self.n_fi}")
        print(f"  Injection bins: {self.injection_bins}")
        print(
            f"    Frequencies: {self.fi_full[self.injection_bins][0]*yr:.3e} - "
            f"{self.fi_full[self.injection_bins][-1]*yr:.3e} Hz"
        )
        print(f"  Reconstruction bins: {self.reconstruction_bins}")
        print(
            f"    Frequencies: {self.fi_recon[0]*yr:.3e} - "
            f"{self.fi_recon[-1]*yr:.3e} Hz"
        )

        # Check for overlap
        overlap = np.intersect1d(self.injection_bins, self.reconstruction_bins)
        if len(overlap) > 0:
            print(f"  WARNING: {len(overlap)} overlapping bins: {overlap}")
        else:
            print("  ✓ No overlap (clean separation)")


def create_H_p_ff_binned(fi_full, ff, freq_config, clms_peak, Nside, l_max, S_ff):
    """
    Create H_p_ff with anisotropy only in specified fi bins.

    This uses proper mapping between fi (analysis) and ff (fine grid).
    """
    H_p_ff = np.zeros((hp.nside2npix(Nside), len(ff)))

    # For each ff frequency, determine which fi bin it belongs to
    fi_bins_for_ff = np.searchsorted(fi_full, ff, side="left")
    fi_bins_for_ff = np.clip(fi_bins_for_ff, 0, len(fi_full) - 1)

    # Inject anisotropy
    for i_freq, _f in enumerate(ff):
        corresponding_fi_bin = fi_bins_for_ff[i_freq]

        # Check if this fi bin has anisotropy
        has_anisotropy = freq_config.inj_mask[corresponding_fi_bin]

        clms_real_f = np.zeros(len(clms_peak))
        clms_real_f[0] = clms_peak[0]  # Monopole always present

        if has_anisotropy:
            # Copy all anisotropic components
            clms_real_f[1:] = clms_peak[1:]

        Pk_f = get_map_from_real_clms(clms_real_f, Nside, l_max=l_max)
        H_p_ff[:, i_freq] = Pk_f * S_ff[i_freq]

    return H_p_ff

