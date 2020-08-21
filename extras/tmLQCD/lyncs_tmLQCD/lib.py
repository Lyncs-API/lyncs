"""
Here we load the library libtmLQCD.so and used headers
"""

__all__ = [
    "lib",
    "PATHS",
]

from time import time
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
        "Whether the glibal structure of tmLQCD has been initialized"
        return self._initialized

    def initialize(self, x, y, z, t, seed=None):
        "Initializes the global structure of tmLQCD"

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
        if not x == y == z:
            raise ValueError(
                """
                tmLQCD supports only Lx == Ly == Lz.
                This limitation is due to the usage in the library
                of the variable L. TODO: fix it (good_first_issue).
                """
            )
        self.L = x
        self.LX, self.LY, self.LZ, self.T_global = x, y, z, t
        self.g_nproc_x, self.g_nproc_y, self.g_nproc_z, self.g_nproc_t = 1, 1, 1, 1
        self.g_proc_id = 0
        self.g_mu, self.g_kappa = 0, 1
        self.tmlqcd_mpi_init(0, nullptr)
        self.start_ranlux(1, seed or int(time()))
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
    "io/gauge.h",
    "io/params.h",
    "init/init_geometry_indices.h",
    "init/init_gauge_field.h",
    "rational/elliptic.h",
    "measure_gauge_action.h",
    "measure_rectangles.h",
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
