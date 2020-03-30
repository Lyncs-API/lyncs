__all__ = [
    "Solver",
]

from cppyy import nullptr
import cppyy.ll
from mpi4py import MPI
from . import lib


class Solver:
    "The DDalphaAMG solver class, remote implementation"

    def __init__(
            self,
            comm=MPI.COMM_WORLD,
            procs=None,
    ):
        self.init_params = lib.DDalphaAMG_init()
        self.run_params = lib.DDalphaAMG_parameters()
        self.status = lib.DDalphaAMG_status()

        self.init_params.comm_cart = cppyy.ll.cast["MPI_Comm*"](
            MPI._addressof(comm)
        )
        self.init_params.Cart_rank = nullptr
        self.init_params.Cart_coords = nullptr

        for i in range(4):
            self.init_params.global_lattice[i] = 8
            self.init_params.procs[i] = procs[i]

            self.init_params.block_lattice[i] = 4

            self.init_params.theta[i] = 0

        self.init_params.bc = 0

        self.init_params.number_of_levels = 2
        self.init_params.number_openmp_threads = 1

        self.init_params.kappa = 0
        self.init_params.mu = 0
        self.init_params.csw = 0

        # self.init_params.init_file = nullptr
        # self.init_params.rnd_seeds = nullptr

        lib.DDalphaAMG_initialize(
            self.init_params,
            self.run_params,
            self.status
        )

    def __del__(self):
        lib.DDalphaAMG_finalize()
