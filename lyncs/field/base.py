"""
Base class of the Field type that implements
the interface to the Lattice class.
"""
# pylint: disable=C0303,C0330

__all__ = [
    "BaseField",
]

import re
from collections import Counter
from functools import wraps
from .types.base import FieldType
from lyncs_utils import (
    default_repr_pretty,
    compute_property,
    count,
    add_kwargs_of,
    isiterable,
)


class BaseField:
    """
    Base class of the Field type that implements
    the interface to the Lattice class and deduce
    the list of Field types from the field axes.

    The list of types are accessible via field.types
    """

    _repr_pretty_ = default_repr_pretty

    _index_to_axis = re.compile("_[0-9]+$")

    @classmethod
    def indexes_to_axes(cls, *indexes):
        "Converts field indexes to lattice axes"
        return tuple(re.sub(BaseField._index_to_axis, "", index) for index in indexes)

    @classmethod
    def index_to_axis(cls, index):
        "Converts a field index to a lattice axis"
        return re.sub(BaseField._index_to_axis, "", index)

    def axes_to_indexes(self, *axes):
        "Converts lattice axes to field indexes"
        axes = tuple(self.lattice.expand(*axes))
        counters = {axis: count() for axis in set(axes)}
        return tuple(axis + "_" + str(next(counters[axis])) for axis in axes)

    def __init_attributes__(
        self, field, axes=None, lattice=None, coords=None, **kwargs
    ):
        """
        Initializes the field class.

        Parameters
        ----------
        axes: list(str)
            List of axes of the field.
        lattice: Lattice
            The lattice on which the field is defined.
        coords: list/dict
            Coordinates of the field.
        kwargs: dict
            Extra parameters that will be passed to the field types.
        """

        from ..lattice import Lattice, default_lattice

        if lattice is not None and not isinstance(lattice, Lattice):
            raise TypeError("lattice must be of Lattice type")

        self._lattice = (
            lattice
            if lattice is not None
            else field.lattice
            if isinstance(field, BaseField)
            else default_lattice()
        ).freeze()

        self._axes = tuple(
            self.lattice.expand(axes)
            if axes is not None
            else field.axes
            if isinstance(field, BaseField)
            else self.lattice.dims
        )

        if isinstance(field, BaseField):
            same_indexes = set(self.indexes).intersection(field.indexes)
            self._coords = field.coords.extract(same_indexes)
        else:
            self._coords = {}
        self._coords = self.lattice.coords.resolve(coords, field=self)

        self._types = tuple(
            (name, ftype)
            for name, ftype in FieldType.s.items()
            if isinstance(self, ftype)
        )

        # ordering types by relevance
        self._types = tuple(
            (name, ftype)
            for name, ftype in sorted(
                self.types,
                key=lambda item: len(tuple(self.lattice.expand(item[1].axes.expand))),
                reverse=True,
            )
        )

        for (_, ftype) in self.types:
            try:
                kwargs = ftype.__init_attributes__(self, field=field, **kwargs)
            except AttributeError:
                continue

        return kwargs

    @add_kwargs_of(__init_attributes__)
    def __init__(self, field=None, **kwargs):
        """
        Initializes the field class.

        Parameters
        ----------
        field: Field
            If given, then the missing parameters are deduced from it.
        """

        kwargs = self.__init_attributes__(field, **kwargs)

        if kwargs:
            raise ValueError("Could not resolve the following kwargs.\n %s" % kwargs)

    @property
    def lattice(self):
        "The lattice on which the field is defined."
        return self._lattice

    @property
    def axes(self):
        "List of axes of the field. Order is not significant. See indexes_order."
        return self._axes

    @compute_property
    def axes_counts(self):
        "Tuple of axes and counts in the field"
        return tuple(Counter(self.axes).items())

    @compute_property
    def dims(self):
        "List of dims in the field axes"
        return tuple(
            key for key in self.indexes if self.index_to_axis(key) in self.lattice.dims
        )

    @compute_property
    def dofs(self):
        "List of dofs in the field axes"
        return tuple(
            key for key in self.indexes if self.index_to_axis(key) in self.lattice.dofs
        )

    @compute_property
    def labels(self):
        "List of labels in the field axes"
        return tuple(
            key
            for key in self.indexes
            if self.index_to_axis(key) in self.lattice.labels
        )

    @compute_property
    def indexes(self):
        """
        List of indexes of the field. Similar to .axes but axis are enumerated.
        Order is not significant. See field.indexes_order.
        """
        return self.axes_to_indexes(self.axes)

    def reshape(self, *axes, **kwargs):
        """
        Reshapes the field changing the axes.
        Note: only axes with size 1 can be removed and
            new axes are added with size 1 and coord=None
        """
        axes = kwargs.pop("axes", axes)
        indexes = self.axes_to_indexes(axes)
        shape = dict(self.shape)
        _squeeze = (index for index in self.indexes if index not in indexes)
        for index in _squeeze:
            if not shape[index] == 1:
                raise ValueError("Can only remove axes which size is 1")
        _extend = (index for index in indexes if index not in self.indexes)
        coords = kwargs.pop("coords", {})
        for index in _extend:
            coords.setdefault(index, None)
        return self.copy(axes=axes, coords=coords, **kwargs)

    def squeeze(self, *axes, **kwargs):
        "Removes axes with size one."
        axes = kwargs.pop("axes", axes)
        indexes = self.get_indexes(*axes) if axes else self.indexes
        axes = tuple(
            self.index_to_axis(key)
            for key, val in self.shape
            if key not in indexes or val > 1
        )
        return self.copy(axes=axes, **kwargs)

    def unsqueeze(self, *axes, **kwargs):
        "Sets coordinate to None for the axes with size one."
        axes = kwargs.pop("axes", axes)
        indexes = self.get_indexes(*axes) if axes else self.indexes
        coords = kwargs.pop("coords", {})
        coords.update(
            {key: None for key, val in self.shape if key in indexes and val == 1}
        )
        return self.copy(coords=coords, **kwargs)

    def extend(self, *axes, **kwargs):
        "Adds axes with size one (coord=None)."
        axes = kwargs.pop("axes", axes)
        return self.reshape(self.axes + axes, **kwargs)

    def get_axes(self, *axes):
        "Returns the corresponding field axes to the given axes/dimensions"
        if not isiterable(axes, str):
            raise TypeError("The arguments need to be a list of strings")
        if "all" in axes:
            return tuple(sorted(self.axes))
        axes = (axis for axis in self.lattice.expand(*axes) if axis in self.axes)
        return tuple(sorted(axes))

    def get_indexes(self, *axes):
        "Returns the corresponding indexes of the given axes/indexes/dimensions"
        if not isiterable(axes, str):
            raise TypeError("The arguments need to be a list of strings")
        if "all" in axes:
            return tuple(sorted(self.indexes))
        indexes = set(axis for axis in axes if axis in self.indexes)
        axes = tuple(self.lattice.expand(set(axes).difference(indexes)))
        indexes.update([idx for idx in self.indexes if self.index_to_axis(idx) in axes])
        return tuple(sorted(indexes))

    def get_range(self, key):
        "Returns the range of the given index/axis."
        tmp = self.get_indexes(key)
        if len(tmp) > 1:
            tmp = set(self.get_range(_k) for _k in tmp)
            if len(tmp) == 1:
                return tuple(tmp)[0]
            raise ValueError(
                "%s corresponds to more than one index with different size" % key
            )
        if len(tmp) == 0:
            raise KeyError("%s not in field" % key)
        key = tmp[0]
        val = self.coords[key]
        if val == slice(None):
            axis = self.index_to_axis(key)
            return self.lattice.get_axis_range(axis)
        if isinstance(val, (int, str, type(None))):
            return (val,)
        if isinstance(val, slice):
            return range(val.start, val.stop, val.step)
        return val

    def get_size(self, key):
        "Returns the size of the given index/axis."
        return len(self.get_range(key))

    @compute_property
    def shape(self):
        "Returns the list of indexes with size. Order is not significant."
        return tuple((key, self.get_size(key)) for key in self.indexes)

    @compute_property
    def size(self):
        "Returns the number of elements in the field."
        prod = 1
        for _, length in self.shape:
            prod *= length
        return prod

    @property
    def types(self):
        "List of field types that the field is instance of, ordered per relevance"
        return getattr(self, "_types", ())

    @property
    def coords(self):
        "List of coordinates of the field."
        return self._coords

    def __dir__(self):
        attrs = set(super().__dir__())
        for _, ftype in self.types:
            attrs.update((key for key in dir(ftype) if not key.startswith("_")))
        return sorted(attrs)

    def __getattr__(self, key):
        "Looks up for methods in the field types"
        if key == "_types":
            raise AttributeError

        for _, ftype in self.types:
            if isinstance(self, ftype):
                try:
                    return getattr(ftype, key).__get__(self)
                except AttributeError:
                    continue

        raise AttributeError("%s not found" % key)

    def __setattr__(self, key, val):
        "Looks up for methods in the field types"
        for _, ftype in self.types:
            if isinstance(self, ftype):
                try:
                    getattr(ftype, key).__set__(self, val)
                except AttributeError:
                    continue

        super().__setattr__(key, val)

    @property
    def type(self):
        "Name of the Field. Equivalent to the most relevant field type."
        return self.types[0][0]

    def copy(self, **kwargs):
        "Creates a shallow copy of the field"
        return type(self)(self, **kwargs)

    def __getitem__(self, coords):
        return self.get(coords)

    def get(self, *keys, **coords):
        "Gets the components at the given coordinates"
        return self.copy(coords=(keys, coords))

    def __pos__(self):
        return self

    def __eq__(self, other):
        return self is other or (
            isinstance(other, type(self))
            and self.lattice == other.lattice
            and set(self.indexes) == set(other.indexes)
            and self.coords == other.coords
        )


FieldType.BaseField = BaseField
FieldType.Field = BaseField


def wrap_method(method, ftype):
    "Wrapper for field methods"

    fnc = getattr(ftype, method)

    @wraps(fnc)
    def wrapped(field, *args, **kwargs):
        if not isinstance(field, ftype):
            raise TypeError("First argument of %s must be a field." % method)
        return fnc(field, *args, **kwargs)

    return wrapped


METHODS = (
    "squeeze",
    "extend",
    "reshape",
)

for _ in METHODS:
    __all__.append(_)
    globals()[_] = wrap_method(_, BaseField)
