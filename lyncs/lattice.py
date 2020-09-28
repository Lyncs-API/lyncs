"""
Definition of the Lattice class and related routines
"""

__all__ = [
    "default_lattice",
    "Lattice",
]

import re
import random
from types import MappingProxyType
from functools import partial, wraps
from typing import Callable
from inspect import signature
from lyncs_utils import default_repr_pretty, isiterable, FreezableDict, compact_indexes
from .field.base import BaseField
from .field.types.base import Axes, FieldType


def default_lattice():
    "Returns the last defined lattice if any"
    assert Lattice.last_defined is not None, "Any lattice has been defined yet."
    return Lattice.last_defined


class LatticeDict(FreezableDict):
    "Dictionary for lattice attributes. Checks the given keys."
    regex = re.compile(Axes._get_label.pattern + "$")

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self.lattice = None
        return self

    def __init__(self, val=None, lattice=None, check=True):
        if lattice is not None and not isinstance(lattice, Lattice):
            raise ValueError("Lattice must be of Lattice type")
        self.lattice = lattice

        if check:
            super().__init__()
            if val is not None:
                for _k, _v in dict(val).items():
                    self[_k] = _v
        else:
            super().__init__(val)

    def __setitem__(self, key, val):
        if not type(self).regex.match(key):
            raise KeyError(
                """
                Invalid key: %s. Keys can only contain letters, numbers or '_'.
                Keys must start with a letter and cannot end with '_' followed by number.
                """
                % key
            )
        if key not in self and self.lattice is not None:
            if key in self.lattice.__dir__():
                raise KeyError("%s is already in use" % (key))
        super().__setitem__(key, val)

    @wraps(dict.copy)
    def copy(self):
        return type(self)(self, lattice=self.lattice, check=False)

    def rename(self, key, new_key):
        "Renames a key of the dictionary"
        self[new_key] = self.pop(key)

    def reset(self, val=None):
        "Resets the content of the dictionary"
        # TODO: in case of reset/delitem the it should check if all the other
        # entries of the lattice are still valid. And in case remove them.
        tmp = self.copy()
        self.clear()
        try:
            self.update(val)
        except (ValueError, TypeError, KeyError):
            self.clear()
            self.update(tmp)
            raise


class LatticeAxes(LatticeDict):
    "Dictionary for lattice axes. Values must be positive integers."

    def __setitem__(self, key, val):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(
                "%s = %s not allowed. The value must be a positive int." % (key, val)
            )
        super().__setitem__(key, val)


class LatticeLabels(LatticeDict):
    "Dictionary for lattice labels. Values must be unique strings."

    def labels(self):
        "Returns all the field labels"
        for value in self.values():
            yield from value

    def __setitem__(self, key, val):
        if isinstance(val, str):
            val = (val,)
        if not isiterable(val, str):
            raise TypeError("Labels value can only be a list of strings")

        val = tuple(val)
        if not len(set(val)) == len(val):
            raise ValueError("%s contains repeated labels" % (val,))

        labels = set(self.labels())
        if key in self:
            labels = labels.difference(self[key])
        inter = labels.intersection(val)
        if inter:
            raise ValueError("%s are labels already in use" % inter)

        super().__setitem__(key, val)


class LatticeGroups(LatticeDict):
    "Dictionary for lattice groups. Values must be a set of lattice keys."
    regex = re.compile("[a-zA-Z_][a-zA-Z0-9_]*$")

    def __setitem__(self, key, val):
        if key in self and isinstance(val, int):
            for _k in self[key]:
                self.lattice[_k] = val
            return
        if isinstance(val, str):
            val = (val,)
        if not isiterable(val, str):
            raise TypeError("Groups value can only be a list of strings")

        if self.lattice is not None:
            val = tuple(val)
            keys = set(self.lattice.keys())
            if not keys >= set(val):
                raise ValueError("%s are not lattice keys" % set(val).difference(keys))

        super().__setitem__(key, val)

    def replace(self, key, new_key):
        "Replaces a key with a the new key"

        for _key, val in self.items():
            if key in val:
                val = list(val)
                val[val.index(key)] = new_key
                self[_key] = val


class Lattice:
    """
    Lattice base class.
    A container for all the lattice information.
    """

    last_defined = None
    default_dims_labels = ["t", "x", "y", "z"]
    theories = {
        "QCD": {
            "spin": 4,
            "color": 3,
            "groups": {
                "gauge": ["color"],
            },
        }
    }

    __slots__ = [
        "_dims",
        "_dofs",
        "_labels",
        "_groups",
        "_coords",
        "_maps",
        "_fields",
        "_frozen",
    ]
    _repr_pretty_ = default_repr_pretty

    def __new__(cls, *args, **kwargs):
        # pylint: disable=W0613
        self = super().__new__(cls)
        self._fields = None
        self._frozen = False
        self.dims = None
        self.dofs = None
        self.labels = None
        self.groups = None
        self.coords = None
        self.maps = None
        return self

    def __init__(
        self, dims=4, dofs="QCD", labels=None, groups=None, coords=None, maps=None
    ):
        """
        Lattice initializer.

        Notation
        --------
        Dimensions: (dims) are axes of the Lattice which size is variable.
            The volume of the lattice, i.e. number of sites, is given by the product
            of dims. Dims are usually the axes where one can parallelize on.
        Degrees of Freedoms: (dofs) are local axes with fixed size (commonly small).
        Labels: (labels) are labelled axes of the lattice. Similar to dofs but instead
            of having a size they have a list of unique labels (str, int, hash-able)
        Axes: Any of the above, i.e. list of axes of the field.

        Parameters
        ----------
        dims: int, list or dict (default 4)
            Dimensions (default names: t,x,y,z if less than 5 or dim0/1/2...)
            - int: number of dimensions. The default names will be used.
            - list: size of the dimensions. The default names will be used.
            - dict: names of the dimensions (keys) and sizes (value)
        dofs: str, int, list, dict (default QCD)
            Specifies local degree of freedoms. (default naming: dof0/1/2...)
            - str: one of the defined theories (QCD,...). See Lattice.theories
            - int: size of one degree of freedom
            - list: size per dimension of the degrees of freedom
            - dict: names of the degree of freedom (keys) and sizes (value)
        labels: dict
            Labelled dimensions of the lattice. A dictionary with keys the names of the
            dimensions and with values the list of labels. The labels must be unique.
            The size of the dimension is the number of labels.
        groups: dict
            Grouping of the dimensions. Each entry of the dictionary must contain a str
            or a list of strings that refer to either another label a dimension.
        coords: dict
            Coordinates of the lattice. Each entry of the dictionary must contain a set
            of coordinates
        """
        self.dims = dims
        self.dofs = dofs
        self.labels.update(labels)
        self.groups.update(groups)
        self.coords.update(coords)
        self.maps.update(maps)

        Lattice.last_defined = self

    @property
    def frozen(self):
        """
        Returns if the current lattice instance is frozen, i.e. cannot be changed anymore.
        To unfreeze it use lattice.copy.
        """
        return self._frozen

    @frozen.setter
    def frozen(self, value):
        if value != self.frozen:
            if value is False:
                raise ValueError(
                    "Frozen can only be changed to True. To unfreeze do a copy."
                )
            self.dims.frozen = True
            self.dofs.frozen = True
            self.labels.frozen = True
            self.groups.allows_changes = False
            self.groups.allows_changes = False
            self._fields = self.fields
            self._frozen = True

    def freeze(self):
        "Returns a frozen copy of the lattice"
        if self.frozen:
            return self
        copy = self.copy()
        copy.frozen = True
        return copy

    @property
    def dims(self):
        "Map of lattice dimensions and their size"
        return self._dims

    @dims.setter
    def dims(self, value):
        if self.frozen:
            raise RuntimeError("The lattice has been frozen and dims cannot be changed")

        if not value:
            self._dims = LatticeAxes(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.dims.reset(value)

            # Adding default labels and groups
            dirs = list(self.dims)
            self.labels.setdefault("dirs", dirs)
            if len(dirs) > 1:
                self.groups.setdefault("time", (dirs[0],))
                self.groups.setdefault("space", tuple(dirs[1:]))
            return

        if isinstance(value, int):
            if value < 0:
                raise ValueError("Non-positive number of dims")
            self.dims = [1] * value
            return

        if isiterable(value, int):
            if len(value) <= len(Lattice.default_dims_labels):
                self.dims = {
                    Lattice.default_dims_labels[i]: v for i, v in enumerate(value)
                }
            else:
                self.dims = {"dim%d" % i: v for i, v in enumerate(value)}
            return

        if isiterable(value, str):
            self.dims = {v: 1 for v in value}
            return

        raise TypeError("Not allowed type %s for dims" % type(value))

    @property
    def dofs(self):
        "Map of lattice degrees of freedom and their size"
        return self._dofs

    @dofs.setter
    def dofs(self, value):
        if self.frozen:
            raise RuntimeError("The lattice has been frozen and dofs cannot be changed")

        if not value:
            self._dofs = LatticeAxes(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.dofs.reset(value)
            return

        if isinstance(value, str):
            assert value in Lattice.theories, "Unknown dofs name"
            value = Lattice.theories[value].copy()
            labels = value.pop("labels", {})
            groups = value.pop("groups", {})
            self.dofs = value
            self.labels.update(labels)
            self.groups.update(groups)
            return

        if isinstance(value, int):
            if value < 0:
                raise ValueError("Non-positive number of dofs")
            self.dofs = [1] * value
            return

        if isiterable(value, int):
            self.dofs = {"dof%d" % i: v for i, v in enumerate(value)}
            return

        if isiterable(value, str):
            self.dofs = {v: 1 for v in value}
            return

        raise TypeError("Not allowed type %s for dofs" % type(value))

    @property
    def labels(self):
        "List of labels of the lattice"
        return self._labels

    @labels.setter
    def labels(self, value):
        if self.frozen:
            raise RuntimeError(
                "The lattice has been frozen and labels cannot be changed"
            )

        if not value:
            self._labels = LatticeLabels(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.labels.reset(value)
            return

        raise TypeError("Not allowed type %s for labels" % type(value))

    def add_label(self, key, value):
        "Adds a label to the lattice"
        self.labels[key] = value

    @property
    def groups(self):
        "List of groups of the lattice"
        return self._groups

    @groups.setter
    def groups(self, value):
        if self.frozen:
            raise RuntimeError(
                "The lattice has been frozen and groups cannot be changed"
            )

        if not value:
            self._groups = LatticeGroups(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.groups.reset(value)
            return

        raise TypeError("Not allowed type %s for groups" % type(value))

    def add_group(self, key, value):
        "Adds a group to the lattice"
        self.groups[key] = value

    @property
    def coords(self):
        "List of coordinates of the lattice"
        return self._coords

    @coords.setter
    def coords(self, value):
        if self.frozen:
            raise RuntimeError(
                "The lattice has been frozen and coords cannot be changed"
            )

        if not value:
            self._coords = LatticeCoords(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.coords.reset(value)
            return

        raise TypeError("Not allowed type %s for coordinates" % type(value))

    def add_coord(self, key, value):
        "Adds a coord to the lattice"
        self.coords[key] = value

    @property
    def maps(self):
        "List of maps of the lattice"
        return self._maps

    @maps.setter
    def maps(self, value):
        if not value:
            self._maps = LatticeMaps(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.maps.reset(value)
            return

        raise TypeError("Not allowed type %s for maps" % type(value))

    def add_map(self, new_lattice, mapping, unmapping=None, label=None, unlabel=None):
        "Adds a map to the lattice"

        if not isinstance(new_lattice, Lattice):
            raise TypeError(
                f"Given new_lattice of type {type(new_lattice)} is not a Lattice"
            )
        new_lattice = new_lattice.copy()

        key = 1
        while "map%d" % key in self:
            key += 1
        key = label or getattr(mapping, "__name__", "map%d" % key)
        self.maps[key] = LatticeMap(self, new_lattice, mapping)

        if unmapping:
            new_lattice.add_map(self, unmapping, label=unlabel)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Lattice)
            and self.dims == other.dims
            and self.dofs == other.dofs
            and self.labels == other.labels
            and self.groups == other.groups
        )

    @property
    def axes(self):
        "Complete list of axes of the lattice"
        axes = list(self.dims.keys())
        axes.extend(self.dofs.keys())
        axes.extend(self.labels.keys())
        return tuple(axes)

    def keys(self):
        "Complete list of keys of the lattice"
        yield "dims"
        yield "dofs"
        yield "labels"
        yield from self.dims.keys()
        yield from self.dofs.keys()
        yield from self.labels.keys()
        yield from self.groups.keys()
        yield from self.coords.keys()
        yield from self.maps.keys()

    def rename(self, key, new_key):
        "Renames a dimension within the lattice"

        if key == new_key:
            return

        if self.frozen:
            raise RuntimeError(
                "The lattice has been frozen and dimensions cannot be renamed"
            )

        if new_key in dir(self):
            raise KeyError("%s is already in use" % (new_key))

        key_is_axis = key in self.axes
        key_found = False
        for dct in (
            self.dims,
            self.dofs,
            self.labels,
            self.groups,
            self.coords,
            self.maps,
        ):
            if key in dct:
                dct.rename(key, new_key)
                key_found = True

        if not key_found:
            raise KeyError("%s not found" % (key))

        if key_is_axis:
            for dct in self.groups, self.coords, self.maps:
                dct.replace(key, new_key)

    def expand(self, *dimensions):
        "Expand the list of dimensions into the fundamental dimensions and degrees of freedom"
        for dim in dimensions:
            if isinstance(dim, str):
                if dim not in self.keys():
                    raise ValueError("Unknown dimension: %s" % dim)
                if dim in self.axes:
                    yield dim
                else:
                    yield from self.expand(self[dim])
            elif isiterable(dim):
                yield from self.expand(*dim)
            else:
                raise TypeError("Unexpected type %s with value %s" % (type(dim), dim))

    def get_axis_range(self, axis):
        "Returns the range of the given axis"
        if axis not in self.axes:
            raise ValueError("%s is not a lattice axis" % axis)
        if axis in self.labels:
            return self.labels[axis]
        return range(self[axis])

    def get_axis_size(self, axis):
        "Returns the range of the given axis"
        if axis not in self.axes:
            raise ValueError("%s is not a lattice axis" % axis)
        if axis in self.labels:
            return len(self.labels[axis])
        return self[axis]

    @property
    def fields(self):
        "List of available field types on the lattice"
        if self._fields is not None:
            return self._fields
        fields = ["Field"]
        for name, ftype in FieldType.s.items():
            if ftype.axes.labels in self:
                fields.append(name)
        return tuple(sorted(fields))

    @property
    def Field(self):
        "Returns the base Field type class initializer"
        return partial(FieldType.Field, lattice=self)

    def __dir__(self):
        yield from dir(type(self))
        yield from self.keys()
        yield from self.coords
        yield from self.maps
        yield from self.fields

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.keys()
        keys = list(self.keys())
        return all((k in keys for k in key))

    def __getitem__(self, key):
        try:
            return getattr(type(self), key).__get__(self)
        except AttributeError:
            for attr in ["dims", "dofs", "labels", "groups", "coords", "maps"]:
                if key in getattr(self, attr):
                    return getattr(self, attr)[key]
            if key in self.fields:
                return partial(FieldType.s[key], lattice=self)
            raise

    __getattr__ = __getitem__

    def __setitem__(self, key, value):
        try:
            getattr(type(self), key).__set__(self, value)
        except AttributeError:
            for attr in ["dims", "dofs", "labels", "groups", "coords"]:
                if key in getattr(self, attr):
                    getattr(self, attr).__setitem__(key, value)
                    return
            raise

    __setattr__ = __setitem__

    def copy(self, **kwargs):
        "Returns a copy of the lattice."
        kwargs.setdefault("dims", self.dims)
        kwargs.setdefault("dofs", self.dofs)
        kwargs.setdefault("groups", self.groups)
        kwargs.setdefault("labels", self.labels)
        kwargs.setdefault("coords", self.coords)
        kwargs.setdefault("maps", self.maps)
        return Lattice(**kwargs)

    def __copy__(self):
        return self.copy()

    def __getstate__(self):
        return (
            self.dims,
            self.dofs,
            self.labels,
            self.groups,
            self.coords,
            self.maps,
            self.frozen,
        )

    def __setstate__(self, state):
        (
            self.dims,
            self.dofs,
            self.labels,
            self.groups,
            self.coords,
            self.maps,
            self.frozen,
        ) = state


class Coordinates(FreezableDict):
    "Dictionary for coordinates"

    def __init__(self, val=None):
        super().__init__()
        if val is not None:
            for _k, _v in dict(val).items():
                self[_k] = _v

    @classmethod
    def expand(cls, *indexes):
        "Expands all the indexes in the list."
        for idx in indexes:
            if isinstance(idx, (int, str, slice, range, type(None))):
                yield idx
            elif isiterable(idx):
                yield from cls.expand(*idx)
            else:
                raise TypeError("Unexpected type %s" % type(idx))

    def __setitem__(self, key, value):
        value = list(self.expand(value))
        while None in value and len(value) > 1:
            value.remove(None)
        if len(value) == 1:
            value = value[0]
        else:
            value = tuple(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return slice(None)

    def update(self, value=None):
        "Updates the values of existing keys"
        if not value:
            return
        for key, val in dict(value).items():
            if key in self:
                self[key] = (self[key], val)
                continue
            self[key] = val

    def finalize(self, key, interval):
        "Finalizes the list of values for the coordinate"
        if key not in self:
            raise KeyError("Unknown key %s" % key)
        if self[key] is None or self[key] == slice(None):
            return

        values = set()
        interval = tuple(interval)
        for value in self.expand(self[key]):
            if isinstance(value, str):
                if value not in interval:
                    raise ValueError("Value %s not in interval" % value)
                values.add(value)
                continue
            if isinstance(value, int):
                values.add(interval[value])
                continue
            assert isinstance(value, (slice, range)), "Trivial assertion"
            if isinstance(value, range):
                value = slice(value.start, value.stop, value.step)
            values.update(interval[value])
        assert values <= set(interval), "Trivial assertion"
        if values == set(interval):
            values = slice(None)
        elif isiterable(values, str):
            values = tuple(sorted(values, key=interval.index))
        else:
            tmp = tuple(compact_indexes(sorted(values)))
            if len(tmp) == 1:
                values = tmp[0]
        self[key] = values

    def cleaned(self):
        "Removes keys that are slice(None)"
        res = self.copy()
        for key in self.keys():
            if self[key] == slice(None):
                del res[key]
        return res

    def intersection(self, coords):
        "Returns the intersection with the given set of coords"
        res = self.copy()
        for key, val in Coordinates(coords).items():
            if key not in res:
                res[key] = val
                continue
            if val is None or val == slice(None):
                continue
            if res[key] is None:
                if isinstance(val, (int, str)):
                    continue
                raise ValueError("None can only be assigned axis of size one")
            if res[key] == slice(None):
                raise ValueError("slice(None) can only be assigned to slice(None)")
            if isinstance(val, (str, int)):
                raise ValueError("%s=%s not compatible with %s" % (key, res[key], val))
            if isinstance(res[key], (str, int)):
                if res[key] in val:
                    continue
                raise ValueError("%s=%s not in %s" % (key, res[key], val))
            if not set(val) >= set(res[key]):
                raise ValueError(
                    "%s=%s not in field coordinates"
                    % (key, set(res[key]).difference(set(val)))
                )
        return res

    def extract(self, keys):
        "Returns a copy of self including the given keys"
        return type(self)({key: self[key] for key in keys})

    def get_indexes(self, coords):
        "Returns the indexes of the values of coords"
        if self == coords:
            return {}

        indexes = coords.copy()
        for key, val in coords.items():
            if self[key] == val:
                continue
            if val is None:
                if isinstance(self[key], (int, str)):
                    continue
                raise ValueError("None can only be assigned axis of size one")
            if self[key] == slice(None):
                continue
            if self[key] is None:
                indexes[key] = None
                continue
            if isinstance(self[key], (str, int)):
                raise ValueError(
                    "Key %s with value %s is not compatible with %s"
                    % (key, val, self[key])
                )
            if isinstance(val, (str, int)):
                if val not in self[key]:
                    raise ValueError("%s not in field coordinates" % (val))
                if isinstance(val, int):
                    indexes[key] = self[key].index(val)
                continue
            if isiterable(self[key], str):
                if set(val) <= set(self[key]):
                    continue
                raise ValueError(
                    "%s not in field coordinates" % (set(val).difference(self[key]))
                )
            assert isiterable(self[key], int), "Unexpected value %s" % self[key]
            if set(val) <= set(self[key]):
                indexes[key] = tuple(self[key].index(idx) for idx in val)
                continue
            raise ValueError(
                "%s not in field coordinates" % (set(val).difference(self[key]))
            )
        return indexes.cleaned()


class LatticeCoords(LatticeDict):
    "LatticeCoords class"
    regex = re.compile("[a-zA-Z_][a-zA-Z0-9_]*$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.lattice is None:
            raise ValueError("LatticeCoords requires a lattice")

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.deduce(key)

    def __setitem__(self, key, value):
        if key in self.lattice.labels.labels():
            raise KeyError("%s is already used in lattice labels" % key)

        super().__setitem__(key, self.resolve(value))

    @classmethod
    def format_coords(cls, *keys, **coords):
        "Returns a list of keys, coords from the given keys and coords"
        args = set()
        coords = Coordinates(coords)
        for key in keys:
            if key is None:
                continue
            if isinstance(key, str):
                args.add(key)
            elif isinstance(key, dict):
                coords.update(key)
            else:
                if not isiterable(key):
                    raise TypeError(
                        "keys can be str, dict or iterables. %s not accepted." % key
                    )
                _args, _coords = cls.format_coords(*key)
                coords.update(_coords)
                args.update(_args)
        return tuple(args), coords

    def replace(self, key, new_key):
        "Replaces a key with a the new key"
        # TODO

    def random(self, *axes, label=None):
        "A random coordinate in the lattice dims and dofs"
        if not axes:
            axes = self.lattice.axes
        else:
            axes = self.lattice.expand(axes)

        coord = {key: random.choice(self.lattice.get_axis_range(key)) for key in axes}

        if label is not None:
            self[label] = coord

        return coord

    def random_source(self, label=None):
        "A random coordinate in the lattice dims"
        return self.random("dims", label=label)

    def resolve(self, *keys, field=None, **coords):
        "Combines a set of coordinates"
        if field is not None and not isinstance(field, BaseField):
            raise ValueError("field must be a Field type")

        keys, coords = self.format_coords(*keys, **coords)
        if not keys and not coords:
            if field is not None:
                return Coordinates(field.coords).cleaned().freeze()
            return Coordinates().freeze()

        # Adding to resolved all the coordinates
        resolved = Coordinates()
        for axis, val in coords.items():
            if field is not None:
                indexes = field.get_indexes(axis)
                if not indexes:
                    raise KeyError("Index '%s' not in field" % axis)
            else:
                indexes = self.lattice.expand(axis)
            resolved.update({idx: val for idx in indexes})

        for key in keys:
            coords = self.deduce(key)
            if field is not None:
                coords = {
                    index: val
                    for axis, val in coords.items()
                    for index in field.get_indexes(axis)
                }
                if not coords:
                    raise KeyError("Coord '%s' not in field" % key)
            resolved.update(coords)

        # Finalizing the coordinates values
        for key, val in resolved.items():
            interval = self.lattice.get_axis_range(BaseField.index_to_axis(key))
            resolved.finalize(key, interval=interval)

        if field is not None:
            resolved = resolved.intersection(field.coords)

        return resolved.cleaned().freeze()

    def deduce(self, key):
        """
        Deduces the coordinates from the key.

        E.g.
        ----
        "random source"
        "color diagonal"
        "x=0"
        """
        if key in self:
            return dict(self[key])

        # Looking up in lattice labels
        for name, labels in self.lattice.labels.items():
            if key in labels:
                return {name: key}

        # TODO
        raise NotImplementedError


class LatticeMap:
    "Class for defining maps between a lattice and another"

    def __init__(self, lat_from: Lattice, lat_to: Lattice, mapping: Callable):

        self.lat_from = lat_from
        self.lat_to = lat_to
        self.mapping = mapping

        annotations = self.mapping.__annotations__
        params = signature(self.mapping).parameters
        if len(params) > 1 or "return" not in annotations:
            raise TypeError(
                """
            Mapping uses annotations to deduce the input/output coordinates
            E.g. map(**kwargs: ["dims"]) -> ["dims"]
            """
            )
        if annotations["return"] not in self.lat_to:
            raise ValueError(
                f"Output dimentions {annotations['return']} not in second lattice"
            )
        self.out = tuple(self.lat_to.expand(annotations["return"]))

        self.args = []
        for key in params:
            if key in annotations:
                key = annotations[key]
            if key not in self.lat_from:
                raise ValueError(f"Input dimentions {key} not in first lattice")
            self.args.append(key)
        self.args = tuple(self.lat_from.expand(self.args))

    def get(self, **coords):
        "Returns the coords after applying the map"
        to_transform = dict()
        for key in tuple(coords):
            if key in self.args:
                to_transform[key] = coords.pop(key)
        coords.update(self.mapping(**to_transform))
        return coords

    def __repr__(self):
        return f"{self.args} -> {self.out}"

    def __call__(self):
        for key, coords in self.lat_from.coords.items():
            new_coords = self.get(**coords)
            if key not in self.lat_to:
                self.lat_to.add_coord(key, new_coords)
            else:
                assert new_coords == self.lat_to[key]

        return self.lat_to


class LatticeMaps(LatticeDict):
    "LatticeMaps class"

    def __setitem__(self, key, value):
        if key in self.lattice.labels.labels():
            raise KeyError("%s is already used in lattice labels" % key)
        if not isinstance(value, LatticeMap):
            raise TypeError("Expected a LatticeMap for map")

        super().__setitem__(key, value)

    def replace(self, key, new_key):
        "Replaces a key with a the new key"
        # TODO

    def rename(self, key1, key2):
        "Returns a lattice with key1 renamed to key2"
        lattice = self.lattice.copy()
        lattice.rename(key1, key2)
        map_lattice(self.lattice, lattice, {"%s -> %s" % (key1, key2): None})
        return lattice

    def evenodd(self, axis=None):
        """
        Returns a lattice with even-odd decomposition on the given axis.
        Axis must be an even-sized dimension.
        """
        lattice = self.lattice.copy()

        # Getting first dim with even size
        if axis is None:
            for key in lattice.dims():
                if lattice[key] % 2 == 0:
                    axis = key
                    break
        elif axis not in lattice.dims:
            raise KeyError("Axis must be one of the dims")

        if axis is None:
            raise ValueError(
                "Even-odd decomposition can be applied only if at least one dim is even"
            )
        if lattice[axis] % 2 != 0:
            raise ValueError(
                "Even-odd decomposition can be applied only on a even-sized dimension"
            )

        lattice[axis] //= 2
        return lattice
