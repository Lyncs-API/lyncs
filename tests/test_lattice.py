import pickle
import pytest
from lyncs import Lattice, default_lattice
from lyncs.lattice import LatticeDict, LatticeAxes, LatticeLabels, LatticeGroups


def test_keys():
    keys = LatticeDict()
    keys["a0"] = None
    keys["a_0b"] = "foo"
    keys["a_b_c"] = 1

    assert keys == LatticeDict(keys)
    assert keys == keys.copy()

    for key in ["a_0", "_a", "a!", "a+", "a_"]:
        with pytest.raises(KeyError):
            keys[key] = 1

    with pytest.raises(KeyError):
        keys.update({"a_0": 1})

    with pytest.raises(KeyError):
        keys = LatticeDict({"a_0": 1})

    with pytest.raises(KeyError):
        keys.setdefault("a_0", 1)

    copy = keys.copy()
    assert copy == keys
    with pytest.raises(KeyError):
        copy.reset({"a_0": 1})
    assert copy == keys

    frozen = keys.freeze()
    assert frozen.freeze() is frozen
    frozen.frozen = True
    with pytest.raises(ValueError):
        frozen.frozen = False
    with pytest.raises(RuntimeError):
        frozen["a0"] = 5

    del keys["a0"]
    with pytest.raises(RuntimeError):
        del frozen["a0"]
    assert "a0" in frozen
    assert "a0" not in keys

    with pytest.raises(ValueError):
        LatticeDict(None, "foo")


def test_axes():
    axes = LatticeAxes()
    axes["x"] = 1
    with pytest.raises(ValueError):
        axes["x"] = -1


def test_labels():
    labels = LatticeLabels()
    labels["x"] = "x"
    assert labels["x"] == ("x",)
    with pytest.raises(TypeError):
        labels["x"] = 1
    with pytest.raises(ValueError):
        labels["x"] = ("x", "x")
    assert labels["x"] == ("x",)
    with pytest.raises(ValueError):
        labels["y"] = ("x",)


def test_init():

    lat = Lattice(dims=4, dofs=[4, 3])
    assert len(lat.dims) == 4
    assert len(lat.dofs) == 2
    assert set(["t", "x", "y", "z"]) == set(lat.dims.keys())
    assert default_lattice() == lat

    lat.x = 5
    assert lat.x == 5 and lat.dims["x"] == lat.x
    lat.dof0 = 6
    assert lat.dof0 == 6
    lat.dirs = lat.dims
    assert lat.dirs == tuple(lat.dims)
    lat.space = 6
    assert lat.x == lat.y and lat.y == lat.z and lat.z == 6

    lat.dims["w"] = 8
    assert "w" in lat
    assert lat.space in lat

    with pytest.raises(KeyError):
        lat.dofs["x"] = 3

    assert lat == pickle.loads(pickle.dumps(lat))

    assert all((hasattr(lat, key) for key in dir(lat)))
    assert set(lat.keys()).issubset(dir(lat))


def test_freeze():
    lat = Lattice()
    lat2 = lat.freeze()
    assert lat2.freeze() is lat2

    lat.frozen = False
    lat2.frozen = True
    with pytest.raises(ValueError):
        lat2.frozen = False
    with pytest.raises(RuntimeError):
        lat2.x = 5


def test_init_dims():
    no_dims = Lattice(dims=None)
    assert no_dims == Lattice(dims=False)
    assert no_dims == Lattice(dims=[])
    assert no_dims == Lattice(dims=0)

    dims5 = Lattice(dims=5)
    assert len(dims5.dims) == 5

    with pytest.raises(TypeError):
        Lattice(dims=3.5)


def test_init_dofs():
    no_dofs = Lattice(dofs=None)
    assert no_dofs == Lattice(dofs=False)
    assert no_dofs == Lattice(dofs=[])
    assert no_dofs == Lattice(dofs=0)

    dofs5 = Lattice(dofs=5)
    assert len(dofs5.dofs) == 5

    with pytest.raises(TypeError):
        Lattice(dofs=3.5)


def test_init_labels():
    no_labels = Lattice(labels=None)
    assert no_labels == Lattice(labels=False)
    assert no_labels == Lattice(labels=[])
    assert no_labels == Lattice(labels=0)

    with pytest.raises(TypeError):
        Lattice(labels=3.5)
