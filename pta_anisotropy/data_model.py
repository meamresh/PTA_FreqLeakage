"""
Data-space model: projection of the GWB onto PTA data and frequency windows.

This module provides both:
- A baseline sinc window implementation (matching the original oo.py), and
- A window-corrected implementation (matching the original oowindow.py).
"""

import jax
import jax.numpy as jnp

from .geometry import get_R_pc


@jax.jit
def get_s_I(h_p, h_c, R_p, R_c):
    """Project GWB realization onto response functions."""
    sp = jnp.einsum("p...,Ip...->I...", h_p, R_p)
    sc = jnp.einsum("p...,Ip...->I...", h_c, R_c)
    return sp + sc


# =============================================================================
# Baseline sinc window (oo.py)
# =============================================================================


@jax.jit
def get_s_I_fi_baseline(Tspan, fi, ff, h_p, h_c, R_p, R_c):
    """Project GWB realization with baseline sinc frequency integration."""
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)
    h_R_tot = get_s_I(h_p, h_c, R_p, R_c)
    to_int1 = jnp.einsum("If,gf->Igf", h_R_tot, sinc_minus)
    to_int2 = jnp.einsum("If,gf->Igf", jnp.conj(h_R_tot), sinc_plus)
    s_I = jnp.sum((to_int1 - to_int2), axis=-1)
    return s_I


@jax.jit
def get_D_IJ_fifj_baseline(Tspan, fi, ff, h_tilde, distances, p_vec, theta_k, phi_k):
    """Covariance matrix D_IJ(f_i, f_j) using baseline sinc window."""
    R_p, R_c = get_R_pc(ff, distances, p_vec, theta_k, phi_k)
    s_I = get_s_I_fi_baseline(Tspan, fi, ff, h_tilde[0], h_tilde[1], R_p, R_c)
    D_IJ = jnp.einsum("If,Jg->fgIJ", s_I, jnp.conjugate(s_I))
    return 2.0 * jnp.real(D_IJ)


@jax.jit
def get_D_IJ_fifj_normalization_baseline(Tspan, fi, ff, H_p_ff):
    """Normalization factor for D_IJ using the baseline sinc window."""
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)
    H_fj = 4 * jnp.pi * jnp.mean(H_p_ff, axis=0)
    spectrum = H_fj / (2.0 * jnp.pi * ff) ** 2
    normalization = (
        2.0
        * 2.0
        / 3.0
        * jnp.sum(
            (
                sinc_minus[:, None] * sinc_minus[None, :]
                + sinc_plus[:, None] * sinc_plus[None, :]
            )
            * spectrum[None, None, :],
            axis=-1,
        )
    )
    return normalization


# =============================================================================
# Window-corrected implementation (oowindow.py)
# =============================================================================


@jax.jit
def get_s_I_fi_windowed(Tspan, fi, ff, h_p, h_c, R_p, R_c):
    """
    Project GWB realization with modified frequency integration window.

    This matches the window implementation used in the original oowindow.py.
    """
    # Basic sinc terms
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)

    # Additional correction terms
    cos_term = jnp.cos(jnp.pi * fi[:, None] * Tspan)
    sinc_f = jnp.sinc(ff[None, :] * Tspan)

    correction = (ff[None, :] * cos_term / fi[:, None]) * sinc_f

    # Modified window functions
    window_minus = sinc_minus + correction
    window_plus = sinc_plus - correction

    h_R_tot = get_s_I(h_p, h_c, R_p, R_c)

    to_int1 = jnp.einsum("If,gf->Igf", h_R_tot, window_minus)
    to_int2 = jnp.einsum("If,gf->Igf", jnp.conj(h_R_tot), window_plus)
    s_I = jnp.sum((to_int1 - to_int2), axis=-1)
    return s_I


@jax.jit
def get_D_IJ_fifj_windowed(Tspan, fi, ff, h_tilde, distances, p_vec, theta_k, phi_k):
    """Covariance matrix D_IJ(f_i, f_j) using window-corrected implementation."""
    R_p, R_c = get_R_pc(ff, distances, p_vec, theta_k, phi_k)
    s_I = get_s_I_fi_windowed(Tspan, fi, ff, h_tilde[0], h_tilde[1], R_p, R_c)
    D_IJ = jnp.einsum("If,Jg->fgIJ", s_I, jnp.conjugate(s_I))
    return 2.0 * jnp.real(D_IJ)


@jax.jit
def get_D_IJ_fifj_normalization_windowed(Tspan, fi, ff, H_p_ff):
    """Normalization factor for D_IJ using the window-corrected implementation."""
    # Basic sinc terms
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)

    # Correction terms
    cos_term = jnp.cos(jnp.pi * fi[:, None] * Tspan)
    sinc_f = jnp.sinc(ff[None, :] * Tspan)
    correction = (ff[None, :] * cos_term / fi[:, None]) * sinc_f

    window_minus = sinc_minus + correction
    window_plus = sinc_plus - correction

    H_fj = 4 * jnp.pi * jnp.mean(H_p_ff, axis=0)
    spectrum = H_fj / (2.0 * jnp.pi * ff) ** 2

    normalization = (
        2.0
        * 2.0
        / 3.0
        * jnp.sum(
            (
                window_minus[:, None] * window_minus[None, :]
                + window_plus[:, None] * window_plus[None, :]
            )
            * spectrum[None, None, :],
            axis=-1,
        )
    )
    return normalization

