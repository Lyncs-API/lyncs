"""
Interface to the DDalphaAMG solver library.

Reference C header and documentation can be found in 
https://github.com/sbacchio/DDalphaAMG/blob/master/src/DDalphaAMG.h
"""

__all__ = [
    "Solver",
]

import logging
from os.path import isfile, realpath
import numpy
from cppyy import nullptr
from mpi4py import MPI
from lyncs_mpi import default_comm
from lyncs_cppyy import ll
from lyncs.utils import factors, prime_factors
from . import lib


class Solver:
    """
    The DDalphaAMG solver class.
    """

    initialized = False
    __slots__ = ["_init_params", "_run_params", "_status", "updated", "_setup"]

    def __init__(
        self,
        global_lattice=None,
        block_lattice=None,
        procs=None,
        comm=None,
        boundary_conditions=-1,
        number_of_levels=1,
        number_openmp_threads=1,
        **kwargs,
    ):
        """
        Initialize a new DDalphaAMG solver class.
        
        Parameters
        ----------
        global_lattice: int[4]
            Size of the lattice. The directions order is T, Z, Y, X.
        block_lattice: int[4]
            Size of the first level blocking. The directions order is T, Z, Y, X.
        procs: int[4]
            Number of processes per direction. The directions order is T, Z, Y, X.
        comm: MPI.Comm
            It can be (a) MPI_COMM_WORLD, (b) A split of MPI_COMM_WORLD, 
            (c) A cartesian communicator with 4 dims and number of processes in
            each directions equal to procs[4] and with proper bondary conditions.
        boundary_conditions: int or int[4]
            It can be +1 (periodic), -1 (anti-periodic) or four floats (twisted)
            i.e. a phase proportional to M_PI * [T, Z, Y, X] will multiplies
            the links in the respective directions.
        number_of_levels: int
            Number of levels for the multigrid, from 1 (no MG) to 4 (max number of levels)
        number_openmp_threads: int
            Number of openmp threads, from 1 to omp_get_num_threads()
        """
        self._init_params = lib.DDalphaAMG_init()
        self._run_params = lib.DDalphaAMG_parameters()
        self._status = lib.DDalphaAMG_status()
        self._setup = 0
        self.updated = True

        global_lattice, block_lattice, procs, comm = get_lattice_partitioning(
            global_lattice, block_lattice, procs, comm
        )

        self._init_params.comm_cart = ll.cast["MPI_Comm"](MPI._handleof(comm))
        self._init_params.Cart_rank = nullptr
        self._init_params.Cart_coords = nullptr

        if boundary_conditions == 1:
            self._init_params.bc = 0
        elif boundary_conditions == -1:
            self._init_params.bc = 1
        else:
            assert (
                hasattr(boundary_conditions, "__len__")
                and len(boundary_conditions) == 4
            ), """
            boundary_conditions can be +1 (periodic), -1 (anti-periodic) or four floats
            (twisted), i.e. a phase proportional to M_PI * [T, Z, Y, X] multiplies links
            in the respective directions.
            """
            self._init_params.bc = 2

        for i in range(4):
            self._init_params.global_lattice[i] = global_lattice[i]
            self._init_params.procs[i] = procs[i]

            self._init_params.block_lattice[i] = block_lattice[i]

            if self._init_params.bc == 2:
                self._init_params.theta[i] = boundary_conditions[i]
            else:
                self._init_params.theta[i] = 0

        self._init_params.number_of_levels = number_of_levels
        self._init_params.number_openmp_threads = number_openmp_threads

        self._init_params.kappa = kwargs.pop("kappa", 0)
        self._init_params.mu = kwargs.pop("mu", 0)
        self._init_params.csw = kwargs.pop("csw", 0)

        # self._init_params.init_file = nullptr
        # self._init_params.rnd_seeds = nullptr

        if Solver.initialized:
            self.__del__()
            logging.warning(
                """
                The solver library was already initialized on this node.
                The previously initialized Solver class cannot be used anymore!
                NOTE: The DDalphaAMG library supports only one Solver at time.
                """
            )
        lib.DDalphaAMG_initialize(self._init_params, self._run_params, self._status)
        Solver.inizialized = True

        kwargs.setdefault("print", 1)
        if kwargs:
            self.update_parameters(**kwargs)

    def __del__(self):
        if Solver.initialized:
            lib.DDalphaAMG_finalize()
            Solver.inizialized = False

    def update_parameters(self, **kwargs):
        "Updates multigrid parameters given in kwargs"
        for key, val in kwargs.items():
            setattr(self._run_params, key, val)
        if not self.updated:
            lib.DDalphaAMG_update_parameters(self._run_params, self._status)
            self.updated = True

    @property
    def setup_status(self):
        "Number of setup iterations performed. If 0, then no setup has been done."
        return self._setup

    @property
    def comm(self):
        "Returns the MPI communicator used by the library."
        comm = MPI.Comm()
        comm_ptr = ll.cast["MPI_Comm*"](MPI._addressof(comm))
        ll.assign(comm_ptr, lib.DDalphaAMG_get_communicator())
        return comm

    def setup(self):
        "Runs the setup. If called again, the setup is re-run."
        lib.DDalphaAMG_setup(self._status)
        self._setup = self._status.success

    def update_setup(self, iterations=1):
        "Runs more setup iterations."
        lib.DDalphaAMG_update_setup(iterations, self._status)
        self._setup = self._status.success

    def solve(self, rhs, tolerance=1e-9):
        "Solves D*x=rhs and returns x at the required tolerance."
        assert rhs.shape == list(self.global_lattice) + [
            4,
            3,
        ], """
        Given array has not compatible shape.
        array shape = %s
        expected shape = %s
        """ % (
            rhs.shape,
            list(self.global_lattice) + [4, 3],
        )

        rhs = numpy.array(rhs, dtype="complex128", copy=False)
        sol = numpy.zeros_like(rhs)
        lib.DDalphaAMG_solve(sol, rhs, tolerance, self._status)
        return sol

    def __dir__(self):
        keys = set(object.__dir__(self))
        for key in dir(self._init_params) + dir(self._run_params):
            if not key.startswith("_"):
                keys.add(key)
        return sorted(keys)

    def __getattribute__(self, key):
        assert Solver.inizialized, "The DDalphaAMG library has been finalized!"
        return object.__getattribute__(self, key)

    def __getattr__(self, key):
        if key in Solver.__slots__:
            return object.__getattribute__(self, key)

        try:
            return getattr(self._run_params, key)
        except AttributeError:
            return getattr(self._init_params, key)

    def __setattr__(self, key, val):
        if key in Solver.__slots__:
            object.__setattr__(self, key, val)
            return

        try:
            setattr(self._run_params, key, val)
            self.updated = False
        except AttributeError:
            assert not hasattr(
                self._init_params, key
            ), "An initialization parameter cannot be changed."
            raise

    def read_configuration(self, filename, fileformat="lime"):
        "Reads configuration from file"
        formats = ["DDalphaAMG", "lime"]
        filename = realpath(filename)
        assert fileformat in formats, "fileformat must be one of %s" % formats
        assert isfile(filename), "Filename %s does not exist" % filename

        shape = list(self.global_lattice) + [4, 3, 3]
        conf = numpy.zeros(shape, dtype="complex128")
        lib.DDalphaAMG_read_configuration(
            conf, filename, formats.index(fileformat), self._status
        )
        return conf

    def set_configuration(self, conf):
        "Sets the configuration to be used in the Dirac operator."
        assert conf.shape == list(self.global_lattice) + [
            4,
            3,
            3,
        ], """
        Given array has not compatible shape.
        array shape = %s
        expected shape = %s
        """ % (
            conf.shape,
            list(self.global_lattice) + [4, 3, 3],
        )

        conf = numpy.array(conf, dtype="complex128", copy=False)
        lib.DDalphaAMG_set_configuration(conf, self._status)
        return self._status.info


def get_lattice_partitioning(global_lattice, block_lattice=None, procs=None, comm=None):
    """
    Checks or dermines the block_lattice and procs based on the given global_lattice and comm.
    """
    assert (
        len(global_lattice) == 4
    ), "global_lattice must be a list of length 4 (T, Z, Y, X)"

    comm = comm or default_comm()
    num_workers = comm.size

    local_lattice = list(global_lattice)
    if block_lattice:
        assert (
            len(block_lattice) == 4
        ), "block_lattice must be a list of length 4 (T, Z, Y, X)"
        assert all((i % j == 0 for i, j in zip(global_lattice, block_lattice))), (
            "block_lattice must divide the global_lattice %s %% %s = 0"
            % (global_lattice, block_lattice)
        )
        local_lattice = [i // j for i, j in zip(local_lattice, block_lattice)]

    if procs:
        assert len(procs) == 4, "procs must be a list of length 4 (T, Z, Y, X)"
        assert numpy.prod(procs) == num_workers, (
            "The number of workers (%d) does not match the given procs %s"
            % (num_workers, procs)
        )
        assert all(
            (i % j == 0 for i, j in zip(global_lattice, procs))
        ), "procs must divide the global_lattice %s %% %s = 0" % (global_lattice, procs)
        local_lattice = [i // j for i, j in zip(local_lattice, procs)]
    else:
        procs = [1] * 4
        for factor in reversed(list(prime_factors(num_workers))):
            for local in reversed(sorted(local_lattice)):
                if local % factor == 0:
                    idx = local_lattice.index(local)
                    local_lattice[idx] //= factor
                    procs[idx] *= factor
                    factor = 1
                    break
            assert (
                factor == 1
            ), """
            Could not create the list of procs:
            num_workers = %d
            factors = %s
            global_lattice = %s
            """ % (
                num_workers,
                factors,
                global_lattice,
            )
        logging.info("Determined procs is %s for %d workers", procs, num_workers)

    if not block_lattice:
        block_lattice = [1] * 4
        # DDalphaAMG requires at least one direction to be multiple of 2
        for local in reversed(sorted(local_lattice)):
            if local % 2 == 0:
                idx = local_lattice.index(local)
                local_lattice[idx] //= 2
                break
        # An optimal block size if 4. Here we find the closest factor to 4
        optimal = 4
        for i, local in enumerate(local_lattice):
            get_ratio = lambda i: i / optimal if i >= optimal else optimal / i
            best = get_ratio(1)
            for factor in factors(local):
                if get_ratio(factor) <= best:
                    best = get_ratio(factor)
                    block_lattice[i] = factor
                else:
                    break
        logging.info("Determined block_lattice is %s", block_lattice)

    return global_lattice, block_lattice, procs, comm
