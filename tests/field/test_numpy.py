import pytest
from lyncs.field import ArrayField
from lyncs import Lattice
import numpy as np


def init_field():
    lat = Lattice()
    lat.space = 4
    lat.time = 8
    field = ArrayField(axes=["dims", "dofs", "dofs"], lattice=lat)
    indeces = field.indeces
    field.indeces_order = indeces
    shape = field.ordered_shape
    return field, indeces, shape


def test_init():
    field, indeces, shape = init_field()

    assert np.all(field.zeros().result == np.zeros(shape))
    assert np.all(field.ones().result == np.ones(shape))

    field = field.rand()
    random = field.result
    assert field == field.copy()
    assert np.all(field.result == random)

    vals = np.arange(9).reshape(3, 3)
    field = ArrayField(
        vals, axes=["color", "color"], indeces_order=["color_0", "color_1"]
    )
    assert np.all(field.result == vals)


def getitem(arr, indeces, **coords):
    return arr.__getitem__(tuple(coords.pop(idx, slice(None)) for idx in indeces))


def test_getitem():
    field, indeces, shape = init_field()
    field = field.rand()
    random = field.result

    assert np.all(field[{"x": 0}].result == getitem(random, indeces, x_0=0))
    assert np.all(
        field[{"y": (0, 1, 2), "z": -1}].result
        == getitem(random, indeces, y_0=range(3), z_0=-1)
    )
    assert np.all(
        field[{"color": 0}].result == getitem(random, indeces, color_0=0, color_1=0)
    )
    assert np.all(field[{"color_0": 0}].result == getitem(random, indeces, color_0=0))


def setitem(arr, value, indeces, **coords):
    arr.__setitem__(tuple(coords.pop(idx, slice(None)) for idx in indeces), value)
    return arr


def test_setitem():
    field, indeces, shape = init_field()
    field = field.rand()
    random = field.result

    field[{"x": 0}] = 0
    assert np.all(field.result == setitem(random, 0, indeces, x_0=0))

    field[{"x": (0, 1)}] = 0
    assert np.all(field.result == setitem(random, 0, indeces, x_0=(0, 1)))
