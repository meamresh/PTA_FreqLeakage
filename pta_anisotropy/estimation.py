"""
Covariance model and iterative parameter estimation utilities.
"""

import jax
import jax.numpy as jnp


@jax.jit
def compute_inverse(matrix):
    """Compute the inverse of a matrix using JAX."""
    return jnp.linalg.inv(matrix)


@jax.jit
def get_covariance_diagonal(signal_lm, gamma_IJ_lm, ff, S_f):
    """Compute diagonal covariance in frequency domain."""
    frequency_part = 4.0 / 3.0 * S_f / (2.0 * jnp.pi * ff) ** 2
    IJ_part = jnp.einsum("p,pIJ->IJ", signal_lm, gamma_IJ_lm)
    return jnp.einsum("f,IJ->fIJ", frequency_part, IJ_part)


@jax.jit
def get_dcovariance_diagonal(signal_lm, gamma_IJ_lm, ff, S_f):
    """Compute derivative of diagonal covariance."""
    identity = jnp.eye(len(signal_lm))
    frequency_part = 4.0 / 3.0 * S_f / (2.0 * jnp.pi * ff) ** 2
    d_IJ_part = jnp.einsum("pq,qIJ->pIJ", identity, gamma_IJ_lm)
    return jnp.einsum("f,pIJ->pfIJ", frequency_part, d_IJ_part)


@jax.jit
def get_update_estimate_diagonal(parameters, data, gamma_IJ_lm, frequencies, S_f):
    """Compute parameter updates using iterative estimation."""
    from .estimation import get_covariance_diagonal, get_dcovariance_diagonal, compute_inverse

    C = get_covariance_diagonal(parameters, gamma_IJ_lm, frequencies, S_f)
    dC = get_dcovariance_diagonal(parameters, gamma_IJ_lm, frequencies, S_f)
    C_inv = compute_inverse(C)
    C_inv_dC = jnp.einsum("fij,afjk->afik", C_inv, dC)
    F = jnp.einsum("afij,bfji->ab", C_inv_dC, C_inv_dC)
    delta = jnp.einsum("fij,fjk->fik", C_inv, data - C)
    d_term = jnp.einsum("afij,fji->a", C_inv_dC, delta)
    F_inv = compute_inverse(F)
    res = jnp.einsum("ab,b->a", F_inv, d_term).real
    return res, F_inv


def iterative_estimation(update_function, theta, D_IJ, gamma_IJ_lm, ff, S_f, i_max=100):
    """Perform iterative estimation of parameters."""
    i = 0
    delta_theta = 1000
    uncertainties = 1e-10

    while i < i_max and jnp.max(jnp.abs(delta_theta / uncertainties)) > 1e-2:
        delta_theta, F_inv = update_function(theta, D_IJ, gamma_IJ_lm, ff, S_f)
        uncertainties = jnp.sqrt(jnp.diag(F_inv))
        theta += delta_theta
        i += 1

    converged = True if i < i_max else False
    return theta, uncertainties, converged

