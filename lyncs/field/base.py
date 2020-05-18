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
from tunable import (
    TunableClass,
    tunable_property,
    derived_property,
    Function,
    function,
    derived_method,
    Permutation,
    Variable,
    Tunable,
    finalize,
    nameit,
)
from .types.base import FieldType
from ..utils import default_repr, compute_property, expand_indeces, count, add_kwargs_of


class BaseField(TunableClass):
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
        self, field, axes=None, lattice=None, coords=None, indeces_order=None, **kwargs
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
        indeces_order: tuple
            The order of the field indeces (field.indeces).
            This also fixes the field shape (field.ordered_shape).
            It is a tunable parameter and the decision can be postpone.
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

        indeces_order = self._get_indeces_order(
            field if isinstance(field, BaseField) else None, indeces_order
        )
        if indeces_order is not None:
            self.indeces_order = indeces_order
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

    def __initialize_value__(self, value, **kwargs):
        "Initializes the value of the field"

        self.value = self.backend.init(value, self.indeces_order)
        return kwargs

    def __validate_value__(self, value, **kwargs):
        "Checks if the field is well defined to have a value"

        if not self.indeces_order.fixed and not finalize(self.value).depends_on(
            self.indeces_order
        ):
            raise ValueError("Value has been given but indeces_order is not fixed.")

        self.value = value

        return kwargs

    def __update_value__(self, field, **kwargs):
        "Checks if something changed wrt field and updates the field value"

        if {idx: val for idx, val in self.coords if idx in field.indeces} != {
            idx: val for idx, val in field.coords if idx in self.indeces
        }:
            self.value = self.backend.getset(self.coords, field.coords)

        if set(self.indeces) != set(field.indeces):
            self.value = self.backend.reshape(self.indeces_order, field.indeces_order)

        if not finalize(self.value).depends_on(self.indeces_order):
            self.value = self.backend.reorder(self.indeces_order, field.indeces_order)

        return kwargs

    @add_kwargs_of(__init_attributes__)
    def __init__(self, field=None, value=None, **kwargs):
        """
        Initializes the field class.
        
        Parameters
        ----------
        field: Field
            If given, then the missing parameters are deduced from it.
        value: Tunable
            The underlying value of the field. Not for the faint of heart.
            If it is given, then all the attributes of the initialization
            are considered proparties of the valu and no transformation
            will be applied.
        """
        super().__init__(field)

        kwargs = self.__init_attributes__(field, **kwargs)

        if value is not None:
            kwargs = self.__validate_value__(value, **kwargs)
        elif isinstance(field, BaseField):
            kwargs = self.__update_value__(field, **kwargs)
        else:
            kwargs = self.__initialize_value__(field, **kwargs)

        if kwargs:
            raise ValueError("Could not resolve the following kwargs.\n %s" % kwargs)

    @property
    def backend(self):
        "Returns the computational backend of the field (dummy)."
        return BaseBackend(self)

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

    @tunable_property
    def indeces_order(self):
        "Order of the field indeces"
        return Permutation(self.indeces)

    def _get_indeces_order(self, field=None, indeces_order=None):
        if indeces_order is not None:
            if (
                not isinstance(indeces_order, Variable)
                and not isinstance(indeces_order, Tunable)
                and set(indeces_order) != set(self.indeces)
            ):
                raise ValueError(
                    "Not valid indeces_order. It has %s, while expected %s"
                    % (indeces_order, self.indeces)
                )
            return indeces_order
        if field is None:
            return None
        if set(self.indeces) == set(field.indeces):
            return field.indeces_order
        if set(self.indeces) <= set(field.indeces):
            select = lambda indeces: (idx for idx in indeces if idx in self.indeces)
            select.__name__ = "select"
            return function(select, field.indeces_order)
        return None

    def reorder(self, *indeces_order, **kwargs):
        "Changes the indeces_order of the field."
        indeces_order = kwargs.pop("indeces_order", indeces_order)
        if indeces_order is ():
            indeces_order = self.indeces_order.copy(uid=True)
        if not isinstance(indeces_order, Variable) and not set(indeces_order) == set(
            self.indeces
        ):
            raise ValueError("All the indeces need to be specified in the reordering")
        return self.copy(indeces_order=indeces_order, **kwargs)

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

    @derived_method(indeces_order)
    def get_indeces_index(self, *axes):
        "Returns the position of indeces of the given axes/indeces/dimensions"
        indeces = self.get_indeces(*axes)
        return tuple(
            (i for i, idx in enumerate(self.indeces_order.value) if idx in indeces)
        )

    @derived_method(indeces_order)
    def get_indeces_order(self, *axes):
        "Returns the ordered indeces of the given axes/indeces/dimensions"
        indeces = self.get_indeces(*axes)
        return tuple((idx for idx in self.indeces_order.value if idx in indeces))

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

    @derived_property(indeces_order)
    def ordered_shape(self):
        "Shape of the field after fixing the indeces_order"
        shape = dict(self.shape)
        return tuple(shape[key] for key in self.indeces_order.value)

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

    def copy(self, value=None, **kwargs):
        "Creates a shallow copy of the field"
        kwargs.setdefault("field", self)
        kwargs["value"] = value
        return type(self)(**kwargs)

    def __getitem__(self, coords):
        return self.get(coords)

    def get(self, *keys, **coords):
        "Gets the components at the given coordinates"
        return self.copy(coords=(keys, coords))

    def __setitem__(self, coords, value):
        return self.set(value, coords)

    def set(self, value, *keys, **coords):
        "Sets the components at the given coordinates"
        coords = self.lattice.coordinates.resolve(*keys, **coords)
        self.value = self.backend.getset(coords, self.coords, value)

    def __pos__(self):
        return self


FieldType.Field = BaseField


def backend_method(fnc):
    "Decorator for backend methods"

    @wraps(fnc)
    def method(self, *args, **kwargs):
        return function(fnc, self.field.value, *args, **kwargs)

    return method


class BaseBackend:
    "Base backend for the field class"

    def __init__(self, field):
        self.field = field

    def getset(self, coords, old_coords=None, value=None):
        "Implementation of get/set field items"
        old_coords = {} if old_coords is None else dict(old_coords)
        new_coords = dict(coords)
        for key, vals in tuple(new_coords.items()):
            vals = tuple(expand_indeces(vals))
            if key in old_coords:
                old_vals = tuple(expand_indeces(old_coords[key]))
                if vals == old_vals:
                    del new_coords[key]
                else:
                    vals = tuple(old_vals.index(val) for val in vals)
            else:
                new_coords[key] = vals

        if value is None:
            return self.getitem(self.field.indeces_order, coords)
        return self.setitem(self.field.indeces_order, coords, value)

    @backend_method
    def getitem(self, indeces_order, coords):
        "Direct implementation of getitem"
        indeces = tuple(
            coords[idx] if idx in coords else slice(None) for idx in indeces_order
        )
        return self.__getitem__(indeces)

    @backend_method
    def setitem(self, indeces_order, coords, value):
        "Direct implementation of setitem"
        indeces = tuple(
            coords[idx] if idx in coords else slice(None) for idx in indeces_order
        )
        return self.__setitem__(indeces, value)

    @backend_method
    def reorder(self, new_order, old_order):
        "Direct implementation of reordering"
        indeces = tuple(new_order.index(idx) for idx in old_order)
        return self.transpose(axes=indeces)

    def __getattr__(self, value):
        def method(self, *args, **kwargs):
            raise NotImplementedError("%s not implemented" % value)

        def attr(*args, **kwargs):
            return Function(method, label=value, args=[self.field.value],)(
                *args, **kwargs
            )

        attr.__name__ = value
        method.__name__ = value
        return attr
