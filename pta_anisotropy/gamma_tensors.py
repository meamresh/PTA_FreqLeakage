"""
Overlap reduction tensor (Gamma_lm) utilities.
"""

import numpy as np
import jax.numpy as jnp
import healpy as hp

from .geometry import unit_vector
from .spherical import sph_harm_y


def get_correlations_lm_IJ(p_vec, l_max, Nside):
    """Get gamma_IJ_lm correlation tensor."""
    npix = hp.nside2npix(Nside)
    theta_k, phi_k = hp.pix2ang(Nside, jnp.arange(npix))
    theta_k = jnp.array(theta_k)
    phi_k = jnp.array(phi_k)

    hat_k = unit_vector(theta_k, phi_k)
    pIpJ = jnp.einsum("iv,jv->ij", p_vec, p_vec)
    pIdotk = jnp.einsum("iv,jv->ij", p_vec, hat_k)

    sum_term = 1 + pIdotk
    diff = 1 - pIdotk
    second_term = -diff[:, None, :] * diff[None, ...]
    pk_qk = pIdotk[:, None, :] * pIdotk[None, ...]
    numerator = 2 * (pIpJ[..., None] - pk_qk) ** 2
    denominator = sum_term[:, None, :] * sum_term[None, ...]

    first_term = jnp.where(denominator != 0.0, numerator / denominator, 0.0)
    first_term = jnp.where(
        ((denominator == 0.0) & (jnp.bool_(jnp.floor(pk_qk)))), -2.0 * second_term, first_term
    )

    gamma_pq = 3.0 / 8.0 * (first_term + second_term)

    n_pulsars = len(p_vec)
    n_lm = (l_max + 1) ** 2

    Y_lm_k = np.zeros((n_lm, npix))
    idx = 0
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            Y_complex = sph_harm_y(ell, m, theta_k, phi_k)
            if m == 0:
                Y_lm_k[idx] = Y_complex.real
            elif m > 0:
                Y_lm_k[idx] = np.sqrt(2) * Y_complex.real * (-1) ** m
            else:
                Y_lm_k[idx] = np.sqrt(2) * Y_complex.imag * (-1) ** m
            idx += 1

    dOmega = 4 * np.pi / npix
    gamma_lm = np.einsum("ijp,lp->lij", np.array(gamma_pq), Y_lm_k) * dOmega
    gamma_lm = gamma_lm * (1 + np.eye(n_pulsars))[None, ...]

    return jnp.array(gamma_lm)

