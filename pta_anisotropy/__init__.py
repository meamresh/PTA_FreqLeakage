"""
PTA anisotropy simulation package.

This package factors out the common functionality used by the original
`oo.py` and `oowindow.py` scripts into reusable modules so that different
window functions and experiment configurations can share the same core.
"""

from . import constants
from . import spherical
from . import geometry
from . import data_model
from . import simulation
from . import gamma_tensors
from . import estimation
from . import freq_config

__all__ = [
    "constants",
    "spherical",
    "geometry",
    "data_model",
    "simulation",
    "gamma_tensors",
    "estimation",
    "freq_config",
]

