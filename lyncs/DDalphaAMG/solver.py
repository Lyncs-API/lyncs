        
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
