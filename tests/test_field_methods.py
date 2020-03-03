def test_ufuncs():
    from lyncs.field_methods import ufuncs, operators, reductions
    from lyncs import Lattice, Field
    import lyncs
    
    lattice = Lattice(dims=4, dofs="QCD")
    field1 = Field(lattice=lattice, dtype="float32")
    field2 = Field(lattice=lattice, dtype="int")
    
    for name, is_member in ufuncs:
        if name == "clip":
            args = (0,1)
        else:
            args = ()
        try:
            res = getattr(lyncs, name)(field1, *args)
        except TypeError:
            res = getattr(lyncs, name)(field1, field2, *args)

    for name, in operators:
        try:
            res = getattr(field1, name)()
        except TypeError:
            res = getattr(field1, name)(field2)

    for name, in reductions:
        for dim in field1.dimensions:
            print(name, dim)
            res = getattr(field1, name)(dim)
