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
        "QCD": {"spin": 4, "color": 3, "labels": {"gauge": ["color"],},},
    }

    __slots__ = [
        "_dims",
        "_dofs",
        "_labels",
        "_dimensions",
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

    def __init__(
        self, dims=4, dofs="QCD", labels=None,
    ):
        """
        Lattice initializer.

        Notation
        --------
        Dimensions: (dims) are labelled axes of the Lattice which size is variable.
            The volume of the lattice, i.e. number of sites, is given by the product
            of dims. Dims are usually the axes where one can parallelize on.
        Degrees of Freedoms: (dofs) are labelled local axes with fixed size.
        Axes: Any of the dimensions or degree of freedoms.

        Parameters
        ----------
        dims: int, list or dict (default 4)
            Dimensions (default labels: t,x,y,z if less than 5 or dim0/1/2...)
            - int: number of dimensions. The default labels will be used.
            - list: size of the dimensions. The default labels will be used.
            - dict: labels of the dimensions (keys) and sizes (value)
        dofs: str, int, list, dict (default QCD)
            Specifies local degree of freedoms. (default naming: dof0/1/2...)
            - str: one of the labeled theories (QCD,...). See Lattice.theories
            - int: size of one degree of freedom
            - list: size per dimension of the degrees of freedom
            - dict: labels of the degree of freedom (keys) and sizes (value)
        labels: dict
            Re-labelling or grouping of the dimensions. Each entry of the dictionary
            must contain a str or a list of strings which name refers to either another
            label or one of the dimensions or degree of freedoms.
        """
        self._frozen = False
        self._dims = {}
        self._dofs = {}
        self._labels = {}
        self._dimensions = None
        self._coordinates = Coordinates(self)
        self._fields = None
        self.dims = dims
        self.dofs = dofs
        if labels is not None:
            self.labels = labels

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
            self._dimensions = self.dimensions
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

    @property
    def n_dims(self):
        "Number of dimensions"
        return len(self._dims)

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

            if self.n_dims > 1:
                dirs = list(self.dims)
                self.labels.setdefault("time", dirs[0])
                self.labels.setdefault("space", dirs[1:])

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

    @property
    def n_dofs(self):
        "Number of lattice degrees of freedom"
        return len(self._dofs)

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
            props = value.pop("labels", {})
            self.dofs = value
            self.labels = props

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
        if not value:
            self._labels = {}

        elif isinstance(value, (dict, MappingProxyType)):
            Lattice.check_keys(value.keys())
            assert all(
                (v in self for v in value.values())
            ), """
            Each property must be either a str, a list or a tuple
            of attributes of the lattice object. See lattice.dimensions.
            """

            self._labels.update(value)

        else:
            assert False, "Not allowed type %s" % type(value)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Lattice)
            and self.dims == other.dims
            and self.dofs == other.dofs
            and self.labels == other.labels
        )

    @property
    def axes(self):
        "Complete list of axes of the lattice"
        keys = set(self.dims.keys())
        keys.update(self.dofs.keys())
        return tuple(sorted(keys))

    @property
    def dimensions(self):  # RENAME ?
        "Complete list of dimensions of the lattice"
        if self._dimensions is not None:
            return self._dimensions
        keys = set(["n_dims", "dims", "n_dofs", "dofs"])
        keys.update(self.dims.keys())
        keys.update(self.dofs.keys())
        keys.update(self.labels.keys())
        return tuple(sorted(keys))

    def _expand(self, dims):
        if isinstance(dims, str):
            if isinstance(self[dims], int):
                return dims
            return " ".join((self._expand(dim) for dim in self[dims]))
        return " ".join((self._expand(dim) for dim in dims))

    def expand(self, *dimensions):
        "Expand the list of dimensions into the fundamental dimensions and degrees of freedom"
        if dimensions not in self:
            raise ValueError("Given unknown dimension: %s" % dimensions)
        return tuple(self._expand(dimensions).split())

    def get_axis_range(self, axis):
        "Returns the range of the given axis"
        if axis not in self.axes:
            raise ValueError("Axis %s is not a lattice axes" % axis)
        return range(self[axis])

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
        keys = set(dir(type(self)))
        keys.update(self.dimensions)
        keys.update(self.fields)
        return sorted(keys)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.dimensions
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
            elif key in self.labels:
                if isinstance(value, (int)):
                    for attr in self.labels[key]:
                        self[attr] = value
                elif isinstance(value, (list, tuple)) and all(
                    (isinstance(v, int) for v in value)
                ):
                    assert len(value) == len(
                        self.labels[key]
                    ), """
                    When setting a property with a list, the length must match.
                    """
                    for attr, val in zip(self.labels[key], value):
                        self[attr] = val
                else:
                    labels = self.labels
                    labels[key] = value
                    self.labels = labels
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
        return Lattice(dims=self.dims, dofs=self.dofs, labels=self.labels)

    def __getstate__(self):
        return (
            self._dims.copy(),
            self._dofs.copy(),
            self._labels.copy(),
            self._coordinates.copy(),
            self.frozen,
        )

    def __setstate__(self, state):
        self._dims, self._dofs, self._labels, self._coordinates, self.frozen = state
