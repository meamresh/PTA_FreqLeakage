"""
Geometric helpers: unit vectors, polarization tensors, and PTA response.
"""

import jax
import jax.numpy as jnp

from .constants import light_speed


@jax.jit
def unit_vector(theta, phi):
    """Compute unit vector from spherical coordinates."""
    x_term = jnp.sin(theta) * jnp.cos(phi)
    y_term = jnp.sin(theta) * jnp.sin(phi)
    z_term = jnp.cos(theta)
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def get_u(theta, phi):
    """Compute derivative of unit vector with respect to theta."""
    x_term = jnp.cos(theta) * jnp.cos(phi)
    y_term = jnp.cos(theta) * jnp.sin(phi)
    z_term = -jnp.sin(theta)
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def get_v(theta, phi):
    """Compute derivative of unit vector with respect to phi."""
    x_term = -jnp.sin(phi)
    y_term = jnp.cos(phi)
    z_term = 0.0 * phi
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def get_plus_cross(theta, phi):
    """Compute plus and cross GW polarization tensors."""
    u = get_u(theta, phi)
    v = get_v(theta, phi)
    plus = jnp.einsum("...i,...j->...ij", u, u) - jnp.einsum(
        "...i,...j->...ij", v, v
    )
    cross = jnp.einsum("...i,...j->...ij", u, v) + jnp.einsum(
        "...i,...j->...ij", v, u
    )
    return plus, cross


@jax.jit
def get_F_pc(p_vec, k_vec, e_p_k, e_c_k):
    """Compute pattern function for pulsars and sky directions."""
    pp = jnp.einsum("...i,...j->...ij", p_vec, p_vec)
    e_p_pp = jnp.einsum("pij,...ij->...p", e_p_k, pp)
    e_c_pp = jnp.einsum("pij,...ij->...p", e_c_k, pp)
    den = 2.0 * (1.0 + jnp.einsum("pi,...i->...p", k_vec, p_vec))
    return e_p_pp / den, e_c_pp / den


@jax.jit
def get_R_pc(f_vec, distances, p_vec, theta_k, phi_k):
    """Compute linear response function for pulsars."""
    k_vec = unit_vector(theta_k, phi_k)
    e_p_k, e_c_k = get_plus_cross(theta_k, phi_k)
    F_p, F_c = get_F_pc(p_vec, k_vec, e_p_k, e_c_k)
    one_plus = 1.0 + jnp.einsum("pi,...i->...p", k_vec, p_vec)
    exponential = 1 - jnp.exp(
        -2.0j
        * jnp.pi
        * jnp.einsum("f,...,...p->...pf", f_vec, distances / light_speed, one_plus)
    )
    R_p_f = jnp.einsum("...,...f,f->...f", F_p, exponential, 1.0 / (2j * jnp.pi * f_vec))
    R_c_f = jnp.einsum("...,...f,f->...f", F_c, exponential, 1.0 / (2j * jnp.pi * f_vec))
    return R_p_f, R_c_f

