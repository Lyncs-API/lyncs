from lyncs_config import ddalphaamg_enabled as enabled
assert enabled, "DDalphaAMG not enabled. Cannot import it"

_lib = None
def get_lib():
    global _lib
    if not _lib:
        import cppyy
        import lyncs_config as config
        cppyy.add_include_path(config.ddalphaamg_path+"/include")
        cppyy.c_include("DDalphaAMG.h")
        cppyy.load_library(config.ddalphaamg_path+"/lib/libDDalphaAMG.so")
        _lib = cppyy.gbl
    return _lib
