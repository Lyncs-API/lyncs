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
from ..utils import default_repr, compute_property, expand_indeces, count, add_kwargs_of


class BaseField:
    """
    Base class of the Field type that implements
    the interface to the Lattice class and deduce
    the list of Field types from the field axes.
    
    The list of types are accessible via field.types
    """

    __repr__ = default_repr

    _index_to_axis = re.compile("_[0-9]+$")

    @classmethod
    def indeces_to_axes(cls, *indeces):
        "Converts field indeces to lattice axes"
        return tuple(re.sub(BaseField._index_to_axis, "", index) for index in indeces)

    @classmethod
    def index_to_axis(cls, index):
        "Converts a field index to a lattice axis"
        return re.sub(BaseField._index_to_axis, "", index)

    def axes_to_indeces(self, *axes):
        "Converts lattice axes to field indeces"
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

        self._coords = tuple(field.coords) if isinstance(field, BaseField) else ()
        self._coords = tuple(self.lattice.coordinates.resolve(coords, field=self))

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
        "List of axes of the field. Order is not significant. See indeces_order."
        return self._axes

    @compute_property
    def axes_counts(self):
        "Tuple of axes and counts in the field"
        return tuple(Counter(self.axes).items())

    @compute_property
    def dims(self):
        "List of dims in the field axes"
        return tuple(
            key for key in self.indeces if self.index_to_axis(key) in self.lattice.dims
        )

    @compute_property
    def dofs(self):
        "List of dofs in the field axes"
        return tuple(
            key for key in self.indeces if self.index_to_axis(key) in self.lattice.dofs
        )

    @compute_property
    def labels(self):
        "List of labels in the field axes"
        return tuple(
            key
            for key in self.indeces
            if self.index_to_axis(key) in self.lattice.labels
        )

    @compute_property
    def indeces(self):
        """
        List of indeces of the field. Similar to .axes but axis are enumerated.
        Order is not significant. See field.indeces_order.
        """
        return self.axes_to_indeces(self.axes)

    def reshape(self, *axes, **kwargs):
        """
        Reshapes the field changing the axes.
        Note: only axes with size 1 can be removed and
            new axes are added with size 1 and coord=None
        """
        axes = kwargs.pop("axes", axes)
        indeces = self.axes_to_indeces(axes)
        shape = dict(self.shape)
        _squeeze = (index for index in self.indeces if index not in indeces)
        for index in _squeeze:
            if not shape[index] == 1:
                raise ValueError("Can only remove axes which size is 1")
        _extend = (index for index in indeces if index not in self.indeces)
        coords = kwargs.pop("coords", {})
        for index in _extend:
            coords.setdefault(index, None)
        return self.copy(axes=axes, coords=coords, **kwargs)

    def squeeze(self, *axes, **kwargs):
        "Removes axes with size one."
        axes = kwargs.pop("axes", axes)
        indeces = self.get_indeces(*axes) if axes else self.indeces
        axes = tuple(
            self.index_to_axis(key)
            for key, val in self.shape
            if key not in indeces or val > 1
        )
        return self.copy(axes=axes, **kwargs)

    def extend(self, *axes, **kwargs):
        "Adds axes with size one (coord=None)."
        axes = kwargs.pop("axes", axes)
        return self.reshape(self.axes + axes, **kwargs)

    def get_axes(self, *axes):
        "Returns the corresponding field axes to the given axes/dimensions"
        indeces = set()
        for axis in axes:
            for _ax in self.lattice.expand(axis):
                if _ax in self.axes:
                    indeces.add(_ax)
        return tuple(indeces)

    def get_indeces(self, *axes):
        "Returns the corresponding indeces of the given axes/indeces/dimensions"
        indeces = set()
        counts = dict(self.axes_counts)
        for axis in axes:
            if axis in self.indeces:
                indeces.add(axis)
            else:
                for _ax in self.lattice.expand(self.index_to_axis(axis)):
                    if _ax in self.axes:
                        indeces.update([_ax + "_" + str(i) for i in range(counts[_ax])])
        return tuple(indeces)

    @compute_property
    def shape(self):
        "Returns the list of indeces with size. Order is not significant."

        def get_size(key):
            axis = self.index_to_axis(key)
            coords = dict(self.coords)
            if key in coords:
                if coords[key] is None:
                    return 1
                return len(tuple(expand_indeces(coords[key])))
            return self.lattice.get_axis_size(axis)

        return tuple((key, get_size(key)) for key in self.indeces)

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
