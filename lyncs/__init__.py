"""
Lyncs, a python API for LQCD applications
"""
__version__ = "0.0.0"

from . import io

from .lattice import *
from .field import *
from .io import *
from . import utils

for extra in [
    "lyncs_mpi",
    "lyncs_cppyy",
    "lyncs_clime",
    "lyncs_DDalphaAMG",
]:
    try:
        exec("import %s as %s" % (extra, extra[6:]))
    except ModuleNotFoundError:
        pass
