"""
Utils for interfacing dask and MPI.
Most of the content of this file is explained in the [notebook](notebooks/Dask-mpi.ipynb).
"""

from dask.distributed import Client as _Client

def default_client():
    import distributed
    client = distributed.default_client()
    assert type(client) is Client, "No MPI client found"
    return client

class Client(_Client):
    "Wrapper to dask.distributed.Client"
    
    def __init__(
            self,
            num_workers = None,
            threads_per_worker = 1,
    ):
        """
        Returns a Client connected to a cluster of `num_workers` workers.
        """
        self._server = None
        
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            # Then the script has been submitted in parallel with mpirun
            num_workers = num_workers or MPI.COMM_WORLD.size-1
            assert MPI.COMM_WORLD.size == num_workers+1, """
            Error: (num_workers + 1) processes required.
            The script has not been submitted on enough processes.
            Got %d processes instead of %d.
            """ % (MPI.COMM_WORLD.size, num_workers+1)
            
            from dask_mpi import initialize
            initialize(nthreads=threads_per_worker, nanny=False)
            
            _Client.__init__(self)
            
        else:
            import sh
            import tempfile
            import multiprocessing

            num_workers = num_workers or (multiprocessing.cpu_count()+1)
            
            # Since dask-mpi produces several file we create a temporary directory
            self._dir = tempfile.mkdtemp()
            self._out = self._dir+"/log.out"
            self._err = self._dir+"/log.err"
            
            # The command runs in the background (_bg=True) and the stdout(err) is stored in self._out(err)
            import os
            pwd=os.getcwd()
            sh.cd(self._dir)
            self._server = sh.mpirun("-n", num_workers+1, "dask-mpi", "--no-nanny", "--nthreads", threads_per_worker,
                               "--scheduler-file", "scheduler.json", _bg = True, _out=self._out, _err=self._err)
            sh.cd(pwd)

            import atexit
            atexit.register(self.close_server)
            
            _Client.__init__(self, scheduler_file=self._dir+"/scheduler.json")

        
        # Waiting for all the workers to connect
        import signal
        import time
        def handler(signum, frame):
            if self._server is not None: self.close_server()
            raise RuntimeError("Couldn't connect to %d processes. Got %d workers."%(num_workers, len(workers)))

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)

        while len(self.workers) != num_workers: time.sleep(0.001)

        signal.alarm(0)

        self.ranks = {key: val["name"] for key,val in self.workers.items()}

        

    @property
    def workers(self):
        "Returns the list of workers."
        return self.scheduler_info()["workers"]


    def close_server(self):
        """
        Closes the running server
        """
        assert self._server is not None
        import shutil
        self.shutdown()
        self.close()
        self._server.wait()
        shutil.rmtree(self._dir)
        self._server = None
        import atexit
        atexit.unregister(self.close_server)

        
    def __del__(self):
        """
        In case of server started, closes the server
        """
        if self._server is not None:
            self.close_server()
        
        if hasattr(_Client, "__del__"):
            _Client.__del__(self)

    def who_has(self, *args, overload=True, **kwargs):
        """
        Overloading of distributed.Client who_has.
        Checks that only one worker owns the futures and returns the list of workers.

        Parameters
        ----------
        overload: bool, default true
            If false the original who_has is used
        """
        if overload:
            _workers = list(_Client.who_has(self, *args, **kwargs).values())
            workers = [w[0] for w in _workers if len(w)==1]
            assert len(workers) == len(_workers), "More than one process has the same reference"
            return workers
        else:
            return _Client.who_has(self, *args, **kwargs)
            
    def select_workers(
            self,
            num_workers = None,
            workers = None,
            exclude = None,
            resources = None,
    ):
        """
        Selects `num_workers` from the one available.
        
        Parameters
        ----------
        workers: list, default all
          List of workers to choose from.
        exclude: list, default none
            List of workers to exclude from the total.
        resources: dict, default none
            Defines the resources the workers should have.
        """

        if not workers: workers = list(self.ranks.keys())
        
        assert len(set(workers)) == len(workers), "Workers has repetitions"
        workers = set(workers)
        
        assert workers.issubset(self.ranks.keys()), "Some workers are unkown %s"%(workers.difference(self.ranks.keys()))

        if exclude:
            assert set(exclude).issubset(self.ranks.keys()), "Some workers to exclude are unkown %s"%(workers.difference(self.ranks.keys()))
            workers = workers.difference(exclude)

        if resources:
            # TODO select accordingly requested resources
            assert False, "Resources not implemented."

        if not num_workers: num_workers = len(workers)
        assert num_workers <= len(workers), "Available workers are less than required"
        
        # TODO implement some rules to choose wisely n workers
        # e.g. workers not busy
        selected = list(workers)[:num_workers]
        
        return selected
    
        
    def create_comm(
            self,
            actor = False,
            **kwargs
        ):
        """
        Return a MPI communicator involving workers available by the client.

        Parameters
        ----------
        actor: bool, default True
            Wether the returned communicator should be a dask actor.
        **kwargs: params
            Following list of parameters for the function select_workers."""

        workers = self.select_workers(**kwargs)
        ranks = [[self.ranks[w] for w in workers]]*len(workers)
        ranks = self.scatter(ranks, workers=workers, hash=False, broadcast=False)

        # Checking the distribution of the group
        _workers = self.who_has(ranks)
        assert set(workers) == set(_workers), """
        Error: Something wrong with scatter. Not all the workers got a piece.
        Expected workers = %s
        Got workers = %s
        """ % (workers, _workers)
        
        def _create_comm(ranks):
            from mpi4py.MPI import COMM_WORLD as comm
            return comm.Create_group(comm.group.Incl(ranks))

        return self.map(_create_comm, ranks, actor=actor)

# Adding select_workers documentation to create_comm
Client.create_comm.__doc__ += Client.select_workers.__doc__.split("----------")[1]
