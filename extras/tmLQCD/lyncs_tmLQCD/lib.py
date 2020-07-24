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

header = [
    "measure_gauge_action.h",
]


lib = Lib(
    path=PATHS,
    library=["libtmLQCD.so", libclime],
    c_include=True,
    check="measure_plaquette",
)
