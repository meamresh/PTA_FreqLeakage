"""
Monte Carlo simulation utilities for PTA anisotropy studies.
"""

import numpy as np
import jax.numpy as jnp
import healpy as hp

from .constants import parsec


def generate_gaussian(mean, sigma, size=None):
    """Generate complex Gaussian data."""
    real = np.random.normal(loc=mean, scale=sigma, size=size)
    imaginary = np.random.normal(loc=mean, scale=sigma, size=size)
    return (real + 1j * imaginary) / np.sqrt(2)


def generate_distance(
    n_pulsars, loc=3.1356587021094077, scale=0.2579495260515389
):
    """Generate pulsar distances in parsecs."""
    return 10 ** np.random.normal(loc=loc, scale=scale, size=n_pulsars)


def generate_pulsar_sky_and_kpixels(
    Np, Nside, log10_loc_pc=3.1356587021094077, log10_scale_pc=0.2579495260515389
):
    """Generate pulsar sky positions and k-pixels."""
    Npix = hp.nside2npix(Nside)
    theta_k, phi_k = hp.pix2ang(Nside, np.arange(Npix))
    theta = jnp.arccos(jnp.array(np.random.uniform(-1.0, 1.0, Np)))
    phi = jnp.array(np.random.uniform(0.0, 2.0 * jnp.pi, Np))
    p_vec = jnp.array(hp.ang2vec(theta, phi))
    cos_IJ = jnp.dot(p_vec, p_vec.T)
    cos_IJ = jnp.clip(cos_IJ, -1.0, 1.0)
    distance = generate_distance(Np, loc=log10_loc_pc, scale=log10_scale_pc)
    return p_vec, cos_IJ, distance, theta_k, phi_k


def generate_hpc_polarization_pixel_frequency(H_p_ff):
    """Generate Gaussian data for GW signal in pixel and frequency space."""
    Npix = H_p_ff.shape[0]
    nfrequencies = H_p_ff.shape[1]
    dOmega = 4 * jnp.pi / Npix
    sigma = jnp.sqrt(H_p_ff * dOmega)
    h_tilde = generate_gaussian(0.0, sigma, size=(2, Npix, nfrequencies))
    return h_tilde


def distances_to_meters(distances_pc):
    """Convert distances in parsec to meters."""
    return distances_pc * parsec

