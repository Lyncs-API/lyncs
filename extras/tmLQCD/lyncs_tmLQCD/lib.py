"""
Here we load the library libtmLQCD.so and used headers
"""

__all__ = [
    "lib",
    "PATHS",
]

from lyncs_cppyy import Lib, nullptr
from lyncs_clime import lib as libclime

from . import __path__


class tmLQCDLib(Lib):
    __slots__ = [
        "_initialized",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    def initialize(self, x, y, z, t):
        if self.initialized:
            if not (x, y, z, t) == (self.LX, self.LY, self.LZ, self.T):
                raise RuntimeError(
                    f"""
                tmLQCD has been already initialized with
                (x,y,z,t) = {(self.LX, self.LY, self.LZ, self.T)}
                and cannot be initialized again.
                """
                )
            return
        self.LX, self.LY, self.LZ, self.T_global = x, y, z, t
        self.tmlqcd_mpi_init(0, nullptr)
        self.init_geometry_indices(self.VOLUMEPLUSRAND)
        self.geometry()
        self.init_gauge_field(self.VOLUMEPLUSRAND, 0)
        self._initialized = True


PATHS = list(__path__)

headers = [
    "global.h",
    "start.h",
    "mpi_init.h",
    "geometry_eo.h",
    "init/init_geometry_indices.h",
    "init/init_gauge_field.h",
    "rational/elliptic.h",
    "measure_gauge_action.h",
]

redefined = {}
with open(__path__[0] + "/lib/redefine-syms.txt", "r") as fp:
    redefined.update((line.split() for line in fp.readlines()))


lib = tmLQCDLib(
    path=PATHS,
    header=headers,
    library=["libtmLQCD.so", libclime],
    c_include=True,
    check="measure_plaquette",
    redefined=redefined,
)
