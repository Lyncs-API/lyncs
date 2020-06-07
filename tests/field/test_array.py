import pytest
from lyncs.field import ArrayField
from lyncs import Lattice

lat = Lattice()
lat.space = 4
lat.time = 8


def test_init():
    with pytest.raises(ValueError):
        ArrayField(foo="bar")

    field = ArrayField(axes=["dims", "dofs", "dofs"], lattice=lat)

    assert field.__is__(field.copy())
    assert field.iscomplex
    assert field.copy(dtype=int).__is__(field.astype(int))
    assert field.astype(field.dtype).__is__(field)
    assert field[{"x": 0}].__is__(field.get(x=0))
    field2 = field.copy()
    field2.dtype = int
    assert field2.dtype == int
    assert not field2.iscomplex
    assert field2.__is__(field2.conj())
    assert field2.__is__(field.astype(int))
    field2 = field.copy()
    field2[{"x": 0}] = 1

    assert field.size * 16 == field.bytes


def test_init_value():
    unit = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    with pytest.raises(ValueError):
        ArrayField(unit, axes=["color", "color"], lattice=lat)
    with pytest.raises(ValueError):
        ArrayField(unit, axes=["color"], lattice=lat)
    with pytest.raises(ValueError):
        ArrayField(value=unit, axes=["color", "color"], lattice=lat)
    field = ArrayField(
        unit, axes=["color", "color"], lattice=lat, indeces_order=("color_0", "color_1")
    )
    field = ArrayField(unit[0], axes=["color"], lattice=lat)


def test_reorder():
    field = ArrayField(axes=["dims", "dofs"], lattice=lat)
    field2 = field.copy()
    assert field.indeces_order == field2.indeces_order
    field2 = field.reorder(field.indeces)
    assert field2.indeces_order.value == field.indeces
    with pytest.raises(ValueError):
        field.reorder(field.indeces[:-1])
    field2 = field.reorder()
    assert field2.indeces_order != field.indeces_order
    with pytest.raises(ValueError):
        field2.indeces_order = field.indeces[:-1]
    field2.indeces_order = field.indeces
    assert field2.indeces_order == field.indeces
    field2 = field.reorder()
    field2.indeces_order = reversed(field.indeces)
    assert field2.indeces_order == tuple(reversed(field.indeces))


def test_reshape():
    field = ArrayField(axes=["dims", "dofs"], lattice=lat)
    field2 = field.extend("dofs")
    assert field.type == "Vector"
    assert field2.type == "Propagator"
    assert field2.squeeze().indeces == field.indeces


def test_transpose():
    field = ArrayField(axes=["dofs", "dofs"], lattice=lat)
    assert field.T.__is__(field.transpose())
    assert field.T.indeces_order == field.indeces_order
    assert field.transpose(spin=(0, 1)).__is__(field)
    # assert field.transpose(spin=(1, 0)) == field.transpose("spin")

    field = ArrayField(axes=["dofs"], lattice=lat)
    assert field.T.__is__(field)

    with pytest.raises(KeyError):
        field.transpose(foo=(0, 1))
    with pytest.raises(TypeError):
        field.transpose(spin=0)
    with pytest.raises(ValueError):
        field.transpose(spin=(0, 1, 2))
    with pytest.raises(ValueError):
        field.transpose(spin=(22, 33))

    assert field.H.__is__(field.dagger())

    field2 = field.real
    assert not field2.iscomplex
    assert field2.H.__is__(field2.T)


def test_roll():
    field = ArrayField(axes=["dims", "dofs"], lattice=lat)
    assert field.roll(1).__is__(field.roll(1, "all"))
    assert field.roll(1).__is__(field.roll(1, "dims", "dofs", "labels"))
    assert field.roll(1, "color").__is__(field.roll(1, axis="color"))
    assert field.roll(1, "color").__is__(field.roll(1, axes="color"))

    with pytest.raises(ValueError):
        field.roll(1, "color", axes="dofs")
    with pytest.raises(KeyError):
        field.roll(1, "color", foo="bar")
    with pytest.raises(TypeError):
        field.roll(1, 2, 3)
