__all__ = [
    "lib",
    "PATHS",
]

from lyncs_cppyy import Lib
from lyncs_mpi import lib as libmpi
from . import __path__
from .config import WITH_CLIME

if WITH_CLIME:
    from lyncs_clime import lib as libclime
else:
    libclime = None

PATHS = list(__path__)

lib = Lib(
    path=PATHS,
    header="DDalphaAMG.h",
    library=["libDDalphaAMG.so", libmpi, libclime],
    c_include=True,
    check="DDalphaAMG_init",
)
