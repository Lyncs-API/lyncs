from lyncs_config import mpi_enabled as enabled
assert enabled, "MPI not enabled. Cannot import it"
from .dask_mpi import Client

_lib = None
def get_lib():
    global _lib
    if not _lib:
        import cppyy
        import lyncs_config as config
        cppyy.add_include_path(config.mpi_include)
        cppyy.include("mpi.h")
        cppyy.load_library(config.mpi_libraries)
        _lib = cppyy.gbl
    return _lib

def initialized():
    from ctypes import c_int
    u = c_int(0)
    get_lib().MPI_Initialized(u)
    return bool(u)

def initialize():
    assert not initialized()
    from cppyy import nullptr
    get_lib().MPI_Init(nullptr, nullptr)
