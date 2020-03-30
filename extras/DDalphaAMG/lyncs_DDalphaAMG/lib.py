__all__ = [
    "lib",
    "PATHS",
]

from lyncs_cppyy import Lib
from . import __path__
from .config import MPI_INCLUDE_DIRS

PATHS = list(__path__)
lib = Lib(
    path=PATHS,
    include=MPI_INCLUDE_DIRS.split(";"),
    header="DDalphaAMG.h",
    library="libDDalphaAMG.so",
    c_include=False,
    check="DDalphaAMG_init",
)
