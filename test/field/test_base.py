import pytest
from lyncs.field import BaseField, squeeze
from lyncs import Lattice

lat = Lattice()
lat.space = 4
lat.time = 8


def test_init():

    with pytest.raises(TypeError):
        squeeze("foo")
    with pytest.raises(TypeError):
        BaseField(lattice="foo")
    with pytest.raises(ValueError):
        BaseField(foo="bar")

    field = BaseField(axes=["dims", "dofs", "dofs"], lattice=lat)

    assert field == field
    assert field == +field
    assert field == field.copy()
    assert dir(field) == field.__dir__()

    assert field.size == 4 * 4 * 4 * 8 * 3 * 3 * 4 * 4
    assert len(field.axes) == 8
    assert len(field.axes) == len(field.indexes)
    assert field.indexes_to_axes(*field.indexes) == field.axes
    assert field.axes_to_indexes(*field.axes) == field.indexes
    assert field.indexes_to_axes(*field.dims) == tuple(field.lattice.expand("dims"))
    assert field.indexes_to_axes(*field.dofs) == tuple(
        field.lattice.expand("dofs", "dofs")
    )
    assert field.labels == ()
    assert set(field.get_axes(*field.axes)) == set(field.axes)
    assert set(field.get_axes("all")) == set(field.axes)
    assert set(field.get_indexes("all")) == set(field.indexes)
    with pytest.raises(TypeError):
        field.get_axes(1)
    with pytest.raises(TypeError):
        field.get_indexes([1, 2])

    types = dict(field.types)
    assert field.type == "Propagator"
    assert field.type == next(iter(types))
    assert "Sites" in types
    assert "Scalar" in types
    assert "Degrees" in types

    source = field.lattice.coords.random_source("source")
    point = field[source]
    assert field["source"] == field[source]
    shape = dict(point.shape)
    for dim in point.dims:
        assert shape[dim] == 1

    dofs = point.squeeze()
    assert point.size == dofs.size
    assert dofs == squeeze(dofs)
    assert not dofs.dims
    assert dofs.dofs == dofs.indexes
    assert point.reshape(dofs.axes) == dofs
    with pytest.raises(ValueError):
        point.reshape(point.axes[:-1])
    assert point == dofs.reshape(point.axes)[source]
    assert dofs.reshape(point.axes) == dofs.extend(lat.dims)

    extended = dofs.extend(lat.dims)[{"x": (0, 1)}]
    assert extended.size == 2 * dofs.size

    everywhere = point.unsqueeze()
    assert everywhere == dofs.extend("dims")


def test_size():
    field = BaseField(axes=["dims", "dirs", "color", "color"], lattice=lat)
    with pytest.raises(KeyError):
        field.get_size("spin")
    assert field.get_size("color") == 3
    with pytest.raises(ValueError):
        field.get_size("dims")


def test_coords():
    field = BaseField(axes=["dims", "dirs", "color", "color"], lattice=lat)

    field.lattice.coords["spin0"] = {"spin": 0}
    assert "spin0" in field.lattice.coords
    with pytest.raises(KeyError):
        field.get(spin=0)
    with pytest.raises(KeyError):
        field["spin0"]

    field.lattice.coords["col0"] = {"color": 0}
    assert "col0" in field.lattice.coords
    assert field["col0"] == field.get(color=0)

    highX = field.get(x=(2, 3))
    assert highX.get_size("x") == 2
    assert highX.get(x=(2, 3)) == highX
    assert highX != field
    assert highX.get(x=(2)).get_size("x") == 1
