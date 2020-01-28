import lyncs
def test_solver():
    if not lyncs.config.ddalphaamg_enabled: return
    client = lyncs.mpi.Client(num_workers=1)
    comms = client.create_comm(actor=False)
    _solvers = client.map(lyncs.DDalphaAMG._Solver,comms, procs=[1,1,1,1], actor=True)
    _actors = [s.result() for s in _solvers]
    assert all([s.done() for s in _solvers])
