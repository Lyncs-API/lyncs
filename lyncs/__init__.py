"""
Lyncs, a python API for LQCD applications
"""
__version__ = "0.0.0"

from .lattice import *

from . import field
from . import io

import lyncs_utils as utils

for pkg in [
    "mpi",
    "cppyy",
    "clime",
    "DDalphaAMG",
    "tmLQCD",
]:
    assert pkg not in globals(), f"{pkg} already defined"
    try:
        exec(f"import lyncs_{pkg} as {pkg}")
    except ModuleNotFoundError:
        pass
