"""
Lyncs, a python API for LQCD applications
"""
__version__ = "0.0.0"

from importlib import import_module

import lyncs_utils as utils
from .lattice import *
from .field import *

# Local sub-modules
from . import field
from . import io

# Importing available Lyncs packages
for pkg in [
    "mpi",
    "cppyy",
    "clime",
    "DDalphaAMG",
    "tmLQCD",
]:
    assert pkg not in globals(), f"{pkg} already defined"
    try:
        globals()[pkg] = import_module(f"lyncs_{pkg}")
    except ModuleNotFoundError:
        pass

del import_module
