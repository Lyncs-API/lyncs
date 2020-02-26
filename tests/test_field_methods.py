def test_ufuncs():
    from lyncs.field_methods import ufuncs, operators
    from lyncs import Lattice, Field
    import lyncs
    
    lattice = Lattice(dims=4, dofs="QCD")
    field = Field(lattice=lattice, dtype="complex64")
    field2 = Field(lattice=lattice, dtype="complex128")
    field3 = Field(lattice=lattice, dtype="float32")
    field4 = Field(lattice=lattice, dtype="float64")
    
    for name, is_member in ufuncs:
        if name == "clip":
            kwargs = {"a_min":0,"a_max":1}
        else:
            kwargs = {}
        try:
            res = getattr(lyncs, name)(field, **kwargs)
        except ValueError:
            try:
                res = getattr(lyncs, name)(field, field2, **kwargs)
            except:
                res = getattr(lyncs, name)(field3, field4, **kwargs)
        except TypeError:
            res = getattr(lyncs, name)(field3, **kwargs)

    for name, in operators:
        print(name)
        try:
            res = getattr(field3, name)()
        except TypeError:
            res = getattr(field3, name)(field4)
