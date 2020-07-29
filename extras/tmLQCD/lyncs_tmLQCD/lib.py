"""
Here we load the library libtmLQCD.so and used headers
"""

__all__ = [
    "lib",
    "PATHS",
]

from lyncs_cppyy import Lib
from lyncs_clime import lib as libclime

from . import __path__

PATHS = list(__path__)

headers = [
    "global.h",
    "rational/elliptic.h",
    "measure_gauge_action.h",
]

redefined = {}
with open(__path__[0] + "/lib/redefine-syms.txt", "r") as fp:
    redefined.update((line.split() for line in fp.readlines()))


lib = Lib(
    path=PATHS,
    header=headers,
    library=["libtmLQCD.so", libclime],
    c_include=True,
    check="measure_plaquette",
    redefined=redefined,
)
