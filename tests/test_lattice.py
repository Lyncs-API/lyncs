def test_init():
    from lyncs import Lattice

    lat = Lattice(dims=4, dofs=[4,3])
    assert len(lat.dims) == 4
    assert len(lat.dofs) == 2
    assert set(["t", "x", "y", "z"]) == set(lat.dims.keys())

    lat.x = 5
    assert lat.x == 5 and lat.dims["x"] == lat.x
    assert lat.check()

    lat.dims['w'] = 8
    assert 'w' in lat
    assert lat.space in lat
    assert lat.check()

    import pickle
    assert lat == pickle.loads(pickle.dumps(lat))
    
    assert all((hasattr(lat, key) for key in dir(lat)))
    assert set(lat.dimensions).issubset(dir(lat))
