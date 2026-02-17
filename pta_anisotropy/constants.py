"""
Basic constants and JAX configuration shared by PTA anisotropy scripts.
"""

import os

import jax
import jax.numpy as jnp

# =============================================================================
# JAX CONFIGURATION
# =============================================================================

which_device = "cpu"
jax.config.update("jax_default_device", jax.devices(which_device)[0])
jax.config.update("jax_enable_x64", True)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

Hubble_over_h = 3.24e-18
hour = 3600
day = 24 * hour
yr = 365.25 * day
f_yr = 1 / yr
light_speed = 299792458.0
parsec = 3.085677581491367e16
Mpc = 1e6 * parsec


def get_output_directory(base_dir: str, subdir: str) -> str:
    """
    Ensure that a given output subdirectory exists under base_dir and return it.
    """
    full_path = os.path.join(base_dir, subdir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

