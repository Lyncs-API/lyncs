import pickle
import pytest
from lyncs import Lattice, default_lattice
from lyncs.lattice import (
    LatticeDict,
    LatticeAxes,
    LatticeLabels,
    LatticeGroups,
    Coordinates,
)


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

    with pytest.raises(AttributeError):
        lat.foo

    with pytest.raises(ValueError):
        list(lat.expand("foo"))

    with pytest.raises(ValueError):
        lat.get_axis_range("foo")

    with pytest.raises(ValueError):
        lat.get_axis_size("foo")

    with pytest.raises(AttributeError):
        lat.foo = "bar"

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
    assert lat.fields == lat2.fields
    with pytest.raises(ValueError):
        lat2.frozen = False
    with pytest.raises(RuntimeError):
        lat2.x = 5
    with pytest.raises(RuntimeError):
        lat2.dims = None
    with pytest.raises(RuntimeError):
        lat2.dofs = None
    with pytest.raises(RuntimeError):
        lat2.labels = None
    with pytest.raises(RuntimeError):
        lat2.groups = None
    with pytest.raises(RuntimeError):
        lat2.coords = None


def test_init_dims():
    no_dims = Lattice(dims=None)
    assert no_dims == Lattice(dims=False)
    assert no_dims == Lattice(dims=[])
    assert no_dims == Lattice(dims=0)

    lattice = Lattice(dims=5)
    assert len(lattice.dims) == 5

    lattice = Lattice(dims=["x", "y", "z"])
    assert len(lattice.dims) == 3
    lattice.x = 8
    assert lattice.x == lattice.get_axis_size("x")

    with pytest.raises(TypeError):
        Lattice(dims=3.5)

    with pytest.raises(ValueError):
        Lattice(dims=-1)


def test_init_dofs():
    no_dofs = Lattice(dofs=None)
    assert no_dofs == Lattice(dofs=False)
    assert no_dofs == Lattice(dofs=[])
    assert no_dofs == Lattice(dofs=0)

    lattice = Lattice(dofs=5)
    assert len(lattice.dofs) == 5

    lattice = Lattice(dofs=["a", "b", "c"])
    assert len(lattice.dofs) == 3

    with pytest.raises(TypeError):
        Lattice(dofs=3.5)

    with pytest.raises(ValueError):
        Lattice(dofs=-1)


def test_init_labels():
    no_labels = Lattice(labels=None)
    assert no_labels == Lattice(labels=False)
    assert no_labels == Lattice(labels=[])
    assert no_labels == Lattice(labels=0)

    lattice = Lattice(labels={"trial": ["foo", "bar"]})
    assert "trial" in lattice.labels
    assert lattice.trial == ("foo", "bar")
    assert lattice.trial == lattice.get_axis_range("trial")
    assert len(lattice.trial) == lattice.get_axis_size("trial")

    lattice.add_label("another", ["one", "two"])
    assert lattice["another"] == ("one", "two")

    with pytest.raises(TypeError):
        lattice.labels = 3.5

    with pytest.raises(TypeError):
        Lattice(labels={"trial": 3.5})


def test_init_groups():
    no_groups = Lattice(groups=None)
    assert no_groups == Lattice(groups=False)
    assert no_groups == Lattice(groups=[])
    assert no_groups == Lattice(groups=0)

    lattice = Lattice(groups={"trial": "x"})
    assert "trial" in lattice.groups
    assert lattice.trial == ("x",)

    lattice.add_group("another", ["x", "y"])
    assert lattice["another"] == ("x", "y")

    with pytest.raises(TypeError):
        lattice.groups = 3.5

    with pytest.raises(TypeError):
        Lattice(groups={"trial": 3.5})

    with pytest.raises(ValueError):
        Lattice(groups={"trial": "foo"})


def test_init_coords():
    no_coords = Lattice(coords=None)
    assert no_coords == Lattice(coords=False)
    assert no_coords == Lattice(coords=[])
    assert no_coords == Lattice(coords=0)

    lattice = Lattice(coords={"trial": {"x": 0, "y": 0}})
    assert "trial" in lattice.coords

    with pytest.raises(KeyError):
        lattice.labels["trial"] = ("foo", "bar")

    val = lattice.coords.random_source("source")
    assert "source" in lattice.coords
    set(dict(lattice.source).keys()) == set(lattice.dims)

    lattice.coords = {"trial": {"x": 0, "y": 0}}
    assert "source" not in lattice.coords
    lattice.trial = {"x": 0, "y": 0}

    lattice.add_coord("another", {"x": 0, "y": 0})
    assert lattice["another"] == lattice["trial"]

    with pytest.raises(TypeError):
        lattice.coords = 3.5

    with pytest.raises(TypeError):
        Lattice(coords={"trial": 3.5})


def test_default():
    lat = Lattice()
    assert len(lat.dims) == 4
    assert set(lat.dims) == set(["x", "y", "z", "t"])
    assert len(lat.dofs) == 2
    assert set(lat.dofs) == set(["spin", "color"])
    assert lat.spin == 4
    assert lat.color == 3
    assert "dirs" in lat and len(lat.dirs) == 4


def test_coords():
    lat = Lattice()
    lat.space = 4
    lat.time = 8

    assert lat.coords.resolve(y=slice(None)) == {}
    assert lat.coords.resolve({"y": (2, 3)}, y=(0, 1)) == {}
    assert lat.coords.resolve(lat.dirs) == {}


def test_expand():
    lat = Lattice()
    assert set(lat.expand("space")) == set(["x", "y", "z"])
    assert set(lat.expand("dims")) == set(lat.expand("space", "time"))
    with pytest.raises(TypeError):
        list(lat.expand(1))


def test_coordinates():
    coords = Coordinates()
    with pytest.raises(TypeError):
        coords["x"] = 3.5
    for val in (
        None,
        (None,),
        (None, None),
        [
            None,
            [
                None,
            ],
        ],
    ):
        coords["x"] = val
        assert coords["x"] == None
        coords.update({"x": val})
        assert coords["x"] == None
    assert coords["y"] == slice(None)


def test_rename():
    lat = Lattice()
    lat.rename("t", "t")
    lat.rename("t", "T")
    assert "T" in lat
    assert "t" not in lat
    assert lat["time"] == ("T",)

    with pytest.raises(RuntimeError):
        lat.freeze().rename("T", "t")
