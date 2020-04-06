__all__ = [
    "lib",
    "initialized",
    "finalized",
    "initialize",
    "finalize",
]

from ctypes import c_int
from cppyy import nullptr
from lyncs_cppyy import Lib
from .config import MPI_INCLUDE_DIRS, MPI_LIBRARIES

lib = Lib(
    include=MPI_INCLUDE_DIRS.split(";"),
    header="mpi.h",
    library=MPI_LIBRARIES.split(";"),
    c_include=False,
    check="MPI_Init",
)


def initialized():
    val = c_int(0)
    lib.MPI_Initialized(val)
    return bool(val)


def finalized():
    val = c_int(0)
    lib.MPI_Finalized(val)
    return bool(val)


def initialize():
    assert not initialized() and not finalized()
    lib.MPI_Init(nullptr, nullptr)


def finalize():
    assert initialized() and not finalized()
    lib.MPI_Finalize()
