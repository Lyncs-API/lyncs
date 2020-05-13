"""
Definition of the Lattice class and related routines
"""
# pylint: disable=C0303,C0330

__all__ = [
    "default_lattice",
    "Lattice",
]

import re
from types import MappingProxyType
from functools import partial
from dask.base import normalize_token
from .utils import default_repr
from .coordinates import Coordinates
from .field.types.base import Axes, FieldType


def default_lattice():
    "Returns the last defined lattice if any"
    assert Lattice.last_defined is not None, "Any lattice has been defined yet."
    return Lattice.last_defined


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
        "_keys",
        "_coordinates",
        "_fields",
        "_frozen",
    ]
    __repr__ = default_repr

    _check_key = re.compile(Axes._get_label.pattern + "$")

    @classmethod
    def check_keys(cls, keys):
        "Checks if the given list of keys if compatible to be label of lattice axes"
        for key in keys:
            if not cls._check_key.match(key):
                raise KeyError(
                    """
                    Invalid key: %s. Keys can only contain only letters, numbers or '_'.
                    Keys must start with a letter and cannot end with '_' followed by number.
                    """
                    % key
                )

    def __init__(self, dims=4, dofs="QCD", labels=None, groups=None):
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
        """
        self._frozen = False
        self._dims = {}
        self._dofs = {}
        self._labels = {}
        self._groups = {}
        self._keys = None
        self._coordinates = Coordinates(self)
        self._fields = None
        self.dims = dims
        self.dofs = dofs
        self.labels = labels
        self.groups = groups

        Lattice.last_defined = self

    @property
    def frozen(self):
        """
        Returns if the current lattice instance is frozen, i.e. cannot be changed anymore.
        To unfreeze it use lattice.copy.
        """
        return getattr(self, "_frozen", False)

    @frozen.setter
    def frozen(self, value):
        if value != self.frozen:
            assert value is True, "Frozen can be only changed to True"
            self._dims = MappingProxyType(self._dims)
            self._dofs = MappingProxyType(self._dofs)
            self._labels = MappingProxyType(self._labels)
            self._groups = MappingProxyType(self._groups)
            self._keys = self.keys
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
        return getattr(self, "_dims", {})

    @dims.setter
    def dims(self, value):

        if not value:
            self._dims = {}

        elif isinstance(value, (dict, MappingProxyType)):
            Lattice.check_keys(value.keys())
            assert all(
                (isinstance(v, int) and v > 0 for v in value.values())
            ), "All sizes of dims must be positive integers"

            self._dims = value.copy()

            dirs = list(self.dims)
            self.labels.setdefault("dirs", tuple("dir_" + d for d in dirs))
            if len(dirs) > 1:
                self.groups.setdefault("time", (dirs[0],))
                self.groups.setdefault("space", tuple(dirs[1:]))

        elif isinstance(value, int):
            assert value > 0, "Non-positive number of dimensions"
            self.dims = [1] * value

        elif isinstance(value, (list, tuple)):
            if len(value) <= len(Lattice.default_dims_labels):
                self.dims = {
                    Lattice.default_dims_labels[i]: v for i, v in enumerate(value)
                }
            else:
                self.dims = {"dim%d" % i: v for i, v in enumerate(value)}

        else:
            assert False, "Not allowed type %s" % type(value)

    @property
    def dofs(self):
        "Map of lattice degrees of freedom and their size"
        return getattr(self, "_dofs", {})

    @dofs.setter
    def dofs(self, value):

        if not value:
            self._dofs = {}

        elif isinstance(value, (dict, MappingProxyType)):
            Lattice.check_keys(value.keys())
            assert all(
                (isinstance(v, int) and v > 0 for v in value.values())
            ), "All dimensions of the dofs must be positive integers"

            self._dofs = value.copy()

        elif isinstance(value, str):
            assert value in Lattice.theories, "Unknown dofs name"
            value = Lattice.theories[value].copy()
            labels = value.pop("labels", {})
            groups = value.pop("groups", {})
            self.dofs = value
            self.labels = labels
            self.groups = groups

        elif isinstance(value, int):
            assert value > 0, "Non-positive size for dof"
            self.dofs = [1] * value

        elif isinstance(value, (list, tuple)):
            self.dofs = {"dof%d" % i: v for i, v in enumerate(value)}

        else:
            assert False, "Not allowed type %s" % type(value)

    @property
    def labels(self):
        "List of labels of the lattice"
        return getattr(self, "_labels", {})

    @labels.setter
    def labels(self, value):
        if value is None:
            return

        if isinstance(value, (dict, MappingProxyType)):
            Lattice.check_keys(value.keys())
            assert all(
                (len(set(v)) == len(v) for v in value.values())
            ), "Labels must be unique"
            value = {key: tuple(val) for key, val in value.items()}
            self._labels.update(value)

        else:
            assert False, "Not allowed type %s" % type(value)

    @property
    def groups(self):
        "List of groups of the lattice"
        return getattr(self, "_groups", {})

    @groups.setter
    def groups(self, value):
        if value is None:
            return

        if isinstance(value, (dict, MappingProxyType)):
            Lattice.check_keys(value.keys())
            assert all(
                (v in self for v in value.values())
            ), """
            Each group must be either a str, a list or a tuple
            of attributes of the lattice object. See lattice.dimensions.
            """
            value = {key: tuple(val) for key, val in value.items()}
            self._groups.update(value)

        else:
            assert False, "Not allowed type %s" % type(value)

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
        axes = set(self.dims.keys())
        axes.update(self.dofs.keys())
        axes.update(self.labels.keys())
        return tuple(sorted(axes))

    @property
    def keys(self):
        "Complete list of keys of the lattice"
        if self._keys is not None:
            return self._keys
        keys = set(["dims", "dofs"])
        keys.update(self.dims.keys())
        keys.update(self.dofs.keys())
        keys.update(self.labels.keys())
        keys.update(self.groups.keys())
        return tuple(sorted(keys))

    def expand(self, *dimensions):
        "Expand the list of dimensions into the fundamental dimensions and degrees of freedom"
        for dim in dimensions:
            if dim not in self:
                raise ValueError("Given unknown dimension: %s" % dimensions)
            if dim in self.dims or dim in self.dofs or dim in self.labels:
                yield dim
            else:
                yield from self.expand(*self[dim])

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
    def coordinates(self):
        "Coordinates on the lattice"
        return self._coordinates

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
        attrs = set(dir(type(self)))
        attrs.update(self.keys)
        attrs.update(self.fields)
        return sorted(attrs)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.keys
        return all((k in self for k in key))

    def __getitem__(self, key):
        try:
            return getattr(type(self), key).__get__(self)
        except AttributeError:
            if key in type(self).__slots__:
                raise
            if key in self.dims:
                return self.dims[key]
            if key in self.dofs:
                return self.dofs[key]
            if key in self.labels:
                return self.labels[key]
            if key in self.groups:
                return self.groups[key]
            if key in self.fields:
                return partial(FieldType.s[key], lattice=self)

            raise

    __getattr__ = __getitem__

    def __setitem__(self, key, value):
        assert not self.frozen, """
        Cannot change a lattice in use by a field. Do a copy first.
        """
        try:
            getattr(type(self), key).__set__(self, value)
        except AttributeError:
            if key in self.dims:
                dims = self.dims
                dims[key] = value
                self.dims = dims
            elif key in self.dofs:
                dofs = self.dofs
                dofs[key] = value
                self.dofs = dofs
            elif key in self.groups:
                if isinstance(value, (int)):
                    for attr in self.groups[key]:
                        self[attr] = value
                elif isinstance(value, (list, tuple)) and all(
                    (isinstance(v, int) for v in value)
                ):
                    assert len(value) == len(
                        self.groups[key]
                    ), """
                    When setting a property with a list, the length must match.
                    """
                    for attr, val in zip(self.groups[key], value):
                        self[attr] = val
                else:
                    groups = self.groups
                    groups[key] = value
                    self.groups = groups
            else:
                raise

    __setattr__ = __setitem__

    def __dask_tokenize__(self):
        return normalize_token((type(self), self.__getstate__()))

    def check(self):
        "Checks if the lattice is valid"
        try:
            return self == self.copy()
        except AssertionError:
            return False

    def copy(self):
        "Returns a copy of the lattice."
        return self.__copy__()

    def __copy__(self):
        # TODO: copy also coordinates
        return Lattice(
            dims=self.dims, dofs=self.dofs, groups=self.groups, labels=self.labels
        )

    def __getstate__(self):
        return (
            self._dims.copy(),
            self._dofs.copy(),
            self._labels.copy(),
            self._groups.copy(),
            self._coordinates.copy(),
            self.frozen,
        )

    def __setstate__(self, state):
        (
            self._dims,
            self._dofs,
            self._labels,
            self._groups,
            self._coordinates,
            self.frozen,
        ) = state
