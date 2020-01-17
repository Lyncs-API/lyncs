import lyncs
def test_solver():
    if not lyncs.config.ddalphaamg_enabled: return
    solver = lyncs.DDalphaAMG.solver()
    assert solver
