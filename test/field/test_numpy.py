import pytest
from lyncs.field import ArrayField
from lyncs import Lattice
import numpy as np


def init_field():
    lat = Lattice()
    lat.space = 4
    lat.time = 8
    field = ArrayField(axes=["dims", "color", "color"], lattice=lat)
    indeces = field.indeces
    field.indeces_order = indeces
    shape = field.ordered_shape
    return field, indeces, shape


def test_init():
    field, indeces, shape = init_field()

    assert field == field.copy(copy=True)

    assert field.zeros() == np.zeros(shape)
    assert field.ones() == np.ones(shape)

    field = field.rand()
    random = field.result
    assert field == field.copy()
    assert field == random

    vals = np.arange(9).reshape(3, 3)
    field = ArrayField(
        vals, axes=["color", "color"], indeces_order=["color_0", "color_1"]
    )
    assert field == vals
    assert field.astype("float") == vals.astype("float")


def getitem(arr, indeces, **coords):
    return arr.__getitem__(tuple(coords.pop(idx, slice(None)) for idx in indeces))


def test_getitem():
    field, indeces, shape = init_field()
    field = field.rand()
    random = field.result

    assert field[{"x": 0}] == getitem(random, indeces, x_0=0)
    assert field[{"y": (0, 1, 2), "z": -1}] == getitem(
        random, indeces, y_0=range(3), z_0=-1
    )
    assert field[{"color": 0}] == getitem(random, indeces, color_0=0, color_1=0)
    assert field[{"color_0": 0}] == getitem(random, indeces, color_0=0)


def setitem(arr, value, indeces, **coords):
    arr.__setitem__(tuple(coords.pop(idx, slice(None)) for idx in indeces), value)
    return arr


def test_setitem():
    field, indeces, shape = init_field()
    field = field.rand()
    random = field.result

    field[{"x": 0}] = 0
    assert field == setitem(random, 0, indeces, x_0=0)

    field[{"x": (0, 1)}] = 0
    assert field == setitem(random, 0, indeces, x_0=(0, 1))
