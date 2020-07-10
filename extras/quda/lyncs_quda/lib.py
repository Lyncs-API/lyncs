"""
Loading the QUDA library
"""

__all__ = [
    "lib",
    "PATHS",
]

from lyncs_cppyy import Lib
from . import __path__
from .config import QUDA_MPI

if QUDA_MPI:
    from lyncs_mpi import lib as libmpi
else:
    libmpi = None

PATHS = list(__path__)

lib = Lib(
    path=PATHS, header="quda.h", library=["libquda.so", libmpi], check="initQuda",
)
