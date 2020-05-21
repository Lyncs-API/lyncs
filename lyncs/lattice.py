"""
Definition of the Lattice class and related routines
"""
# pylint: disable=C0303,C0330

__all__ = [
    "default_lattice",
    "Lattice",
]

import re
import random
from types import MappingProxyType
from functools import partial, wraps
from .utils import default_repr, isiterable, FrozenDict, compact_indeces
from .field.base import BaseField
from .field.types.base import Axes, FieldType


def default_lattice():
    "Returns the last defined lattice if any"
    assert Lattice.last_defined is not None, "Any lattice has been defined yet."
    return Lattice.last_defined


class LatticeDict(FrozenDict):
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


class Lattice:
    """
    Lattice base class.
    A container for all the lattice information.
    """

    last_defined = None
    default_dims_labels = ["t", "x", "y", "z"]
    theories = {
        "QCD": {"spin": 4, "color": 3, "groups": {"gauge": ["color"],},},
    }

    __slots__ = [
        "_dims",
        "_dofs",
        "_labels",
        "_groups",
        "_coords",
        "_fields",
        "_frozen",
    ]
    __repr__ = default_repr

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
        return self

    def __init__(self, dims=4, dofs="QCD", labels=None, groups=None, coords=None):
        """
        Lattice initializer.

        Notation
        --------
        Dimensions: (dims) are axes of the Lattice which size is variable.
            The volume of the lattice, i.e. number of sites, is given by the product
            of dims. Dims are usually the axes where one can parallelize on.
        Degrees of Freedoms: (dofs) are local axes with fixed size (commonly small).
        Labels: (labels) are labelled axes of the lattice. Similar to dofs but instead
            of having a size they have a list of unique labels (str, int, hashable)
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
            Labeled dimensions of the lattice. A dictionary with keys the names of the
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

        if not value:
            self._dims = LatticeAxes(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.dims.reset(value)

            # Adding default labels and groups
            dirs = list(self.dims)
            self.labels.setdefault("dirs", tuple("dir_" + d for d in dirs))
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
        if not value:
            self._labels = LatticeLabels(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.labels.reset(value)
            return

        raise TypeError("Not allowed type %s for labels" % type(value))

    @property
    def groups(self):
        "List of groups of the lattice"
        return self._groups

    @groups.setter
    def groups(self, value):
        if not value:
            self._groups = LatticeGroups(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.groups.reset(value)
            return

        raise TypeError("Not allowed type %s for groups" % type(value))

    @property
    def coords(self):
        "List of coordinates of the lattice"
        return self._coords

    @coords.setter
    def coords(self, value):
        if not value:
            self._coords = LatticeCoords(lattice=self)
            return

        if isinstance(value, (dict, MappingProxyType)):
            self.coords.reset(value)
            return

        raise TypeError("Not allowed type %s for coordinates" % type(value))

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
            for attr in ["dims", "dofs", "labels", "groups", "coords"]:
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

    # def __dask_tokenize__(self):
    #     from dask.base import normalize_token
    #     return normalize_token((type(self), self.__getstate__()))

    def copy(self):
        "Returns a copy of the lattice."
        return self.__copy__()

    def __copy__(self):
        return Lattice(
            dims=self.dims,
            dofs=self.dofs,
            groups=self.groups,
            labels=self.labels,
            coords=self.coords,
        )

    def __getstate__(self):
        return (
            self.dims,
            self.dofs,
            self.labels,
            self.groups,
            self.coords,
            self.frozen,
        )

    def __setstate__(self, state):
        (
            self.dims,
            self.dofs,
            self.labels,
            self.groups,
            self.coords,
            self.frozen,
        ) = state


class Coordinates(FrozenDict):
    "Dictionary for coordinates"

    def __init__(self, val=None):
        super().__init__()
        if val is not None:
            for _k, _v in dict(val).items():
                self[_k] = _v

    @classmethod
    def expand(cls, *indeces):
        "Expands all the indeces in the list."
        for idx in indeces:
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
            tmp = tuple(compact_indeces(sorted(values)))
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
                raise ValueError("None can only be assigned to None")
            if res[key] == slice(None):
                raise ValueError("slice(None) can only be assigned to slice(None)")
            if not set(val) >= set(res[key]):
                raise ValueError(
                    "%s not in field coordinates" % (set(res[key]).difference(set(val)))
                )
        return res


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
                indeces = field.get_indeces(axis)
                if not indeces:
                    raise ValueError("Index '%s' not in field" % axis)
            else:
                indeces = self.lattice.expand(axis)
            resolved.update({idx: val for idx in indeces})

        for key in keys:
            coords = self.deduce(key)
            if field is not None:
                coords = {
                    index: val
                    for axis, val in coords.items()
                    for index in field.get_indeces(axis)
                }
                if not coords:
                    raise ValueError("'%s' not in field" % key)
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
