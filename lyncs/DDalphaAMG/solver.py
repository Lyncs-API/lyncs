from lyncs.DDalphaAMG import get_lib

class _Solver:
    "The DDalphaAMG solver class, remote implementation"
    def __init__(
            self,
            comm,
            procs=None,
    ):
        from cppyy import nullptr
        import cppyy.ll
        from mpi4py import MPI
        self.init_params = get_lib().DDalphaAMG_init()
        self.run_params = get_lib().DDalphaAMG_parameters()
        self.status = get_lib().DDalphaAMG_status()
        
        self.init_params.comm_cart = cppyy.ll.cast["MPI_Comm*"](MPI._addressof(comm))
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

        #self.init_params.init_file = nullptr
        #self.init_params.rnd_seeds = nullptr

        get_lib().DDalphaAMG_initialize(self.init_params, self.run_params, self.status)
        
    def __del__(self):
        get_lib().DDalphaAMG_finalize()

        
class Solver:
    "The DDalphaAMG solver class, front-end."
    _initialized_workers = []
    def __init__(
            self,
            comms = None,
            **kwargs,
    ):
        """
        
        """
        from lyncs.mpi import default_client
        self.client = default_client()
        if not comms:
            num_workers = self._get_num_workers(**kwargs)
            comms = self.client.create_comm(num_workers, exclude=_initialized_workers, actor=False)
            
        self._validate(comms, kwargs)
        self._init_params = kwargs
        self._solvers = self.client.map(lyncs.DDalphaAMG._Solver, comms, **kwargs)
        
        workers = self.client.who_has(self._solvers)
        _initialized_workers += workers

        
    def __del__(self):
        workers = self.client.who_has(self._solvers)
        for w in workers:
            _initialized_workers.remove(w)
            

    def _get_num_workers(
            self,
            procs = None,
            global_lattice = None,
            local_lattice = None,
            **kwargs,
    ):
        """
        Returns correct the number of workers for the solver
        """
        if procs:
            # Number of processes explicitelly given
            from numpy import prod
            return prod(procs)
        # TODO: much more
        return 1
        
    def _validate(
            self,
            comms,
            kwargs
    ):
        """
        Checks that the given parameters are appropriate to start the solver.
        In case of missing parameters it fills them as best as possible.
        """
        assert len(comms) == self._get_num_workers(**kwargs)
