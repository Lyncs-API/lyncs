def test_init():
    from lyncs import Lattice

    lat = Lattice(dims=4, dofs=[4,3])
    assert len(lat.dims) == 4
    assert len(lat.dofs) == 2
    assert ["t", "x", "y", "z"] == list(lat.dims.keys())

    lat.x = 5
    assert lat.x == 5 and lat.dims["x"] == lat.x
    
    
