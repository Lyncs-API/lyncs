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
from tunable import (
    TunableClass,
    tunable_property,
    derived_property,
    Function,
    derived_method,
    Permutation,
)
from .types.base import FieldType
from ..utils import default_repr, compute_property, expand_indeces


class BaseField(TunableClass):
    """
    Base class of the Field type that implements
    the interface to the Lattice class and deduce
    the list of Field types from the field axes.
    
    The list of types are accessible via field.types
    """

    __repr__ = default_repr

    def __init__(
        self, field=None, value=None, axes=None, lattice=None, coords=None, **kwargs
    ):
        """
        Initializes the field class.
        
        Parameters
        ----------
        field: Field
            If given, then the missing parameters are deduced from it.
        value: Tunable
            The underlying value of the field. Not for the faint of heart.
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

        assert lattice is None or isinstance(
            lattice, Lattice
        ), "lattice must be of Lattice type"

        self._coords = ()
        if isinstance(field, BaseField):
            super().__init__(field if value is None else value)
            self._lattice = (lattice or field.lattice).freeze()
            self._axes = tuple(
                self.lattice.expand(field.axes if axes is None else axes)
            )
            self._coords = self.lattice.coordinates.resolve(coords, field=field)
        else:
            super().__init__(value)
            self._lattice = (lattice or default_lattice()).freeze()
            self._axes = tuple(
                self.lattice.expand(lattice.dims if axes is None else axes)
            )
            self._coords = self.lattice.coordinates.resolve(coords, field=self)

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

        self.locked_value = value is not None

        if isinstance(field, BaseField):
            if dict(self.coords) != dict(field.coords):
                self.update(**self.backend.getitem(self.coords, field.coords))
            if dict(self.axes_counts) != dict(field.axes_counts):
                self.update(**self.backend.reshape(self.axes, field.axes))

        if not isinstance(field, BaseField):
            assert (
                not self.locked_value
            ), "Does it make sense to give a value without a field?"
            self.update(**self.backend.initialize(field))

        self.update(**kwargs)

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
            key for key in self.indeces if index_to_axis(key) in self.lattice.dims
        )

    @compute_property
    def dofs(self):
        "List of dofs in the field axes"
        return tuple(
            key for key in self.indeces if index_to_axis(key) in self.lattice.dofs
        )

    @compute_property
    def labels(self):
        "List of labels in the field axes"
        return tuple(
            key for key in self.indeces if index_to_axis(key) in self.lattice.labels
        )

    @compute_property
    def indeces(self):
        """
        List of indeces of the field. Similar to .axes but repeted axis are numerated.
        Order is not significant. See field.indeces_order.
        """
        counts = dict(self.axes_counts)
        idxs = {axis: 0 for axis in counts}
        indeces = []
        for axis in self.axes:
            if counts[axis] > 1:
                indeces.append(axis + "_" + str(idxs[axis]))
                idxs[axis] += 1
            else:
                indeces.append(axis)
        assert len(set(indeces)) == len(indeces), "Trivial assertion"

        return tuple(indeces)

    @tunable_property
    def indeces_order(self):
        "Order of the field indeces"
        return Permutation(self.indeces)

    def reshape(self, *axes):
        """
        Reshapes the field changing the axes.
        Note: only axes with size 1 can be removed and
            new axes are added with size 1 and coord=None
        """
        return self.copy(axes=axes)

    def reorder(self, *indeces_order):
        "Changes the indeces_order of the field."
        if not set(indeces_order) == set(self.indeces):
            raise ValueError("All the indeces need to be specified in the reordering")
        return self.copy(indeces_order=indeces_order)

    def squeeze(self, *axes):
        "Removes axes with size one."
        indeces = self.get_indeces(*axes) if axes else self.indeces
        axes = tuple(
            index_to_axis(key)
            for key, val in self.shape
            if key not in indeces or val > 1
        )
        return self.copy(axes=axes)

    def extend(self, *axes):
        "Adds axes with size one (coord=None)."
        axes = tuple(self.lattice.expand(*axes))
        return self.copy(axes=self.axes + axes)

    def get_axes(self, *axes):
        "Returns the corresponding field axes to the given axes/dimensions"
        indeces = set()
        for axis in axes:
            for _ax in self.lattice.expand(axis):
                if _ax in self.axes:
                    indeces.add(_ax)
        return tuple(indeces)

    def get_indeces(self, *axes):
        "Returns the corresponding indeces to the given axes/indeces/dimensions"
        indeces = set()
        for axis in axes:
            if axis in self.indeces:
                indeces.add(axis)
            else:
                for _ax in self.lattice.expand(axis):
                    count = self.axes.count(_ax)
                    if count == 1:
                        indeces.add(_ax)
                    else:
                        indeces.update([_ax + "_" + str(i) for i in range(count)])
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
            axis = index_to_axis(key)
            coords = dict(self.coords)
            if key in coords:
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

    def update(self, **kwargs):
        "Updates field attributes"
        assert kwargs.pop("field", self) is self, "Cannot update the field itself"
        for key, val in kwargs.items():
            setattr(self, key, val)

    def copy(self, **kwargs):
        "Creates a shallow copy of the field"
        kwargs.setdefault("field", self)
        return type(self)(**kwargs)

    def prepare(self, *fields, elemwise=True, **kwargs):
        """
        Prepares a set of fields for a calculation and 
        creates the field where to store the output.
        
        Returns:
        --------
        fields, out_field
        where fields is a tuple of the fields to use in the calculation
        and out_field is the Field type where to store the result
        
        Parameters
        ----------
        fields: Field(s)
            List of fields involved in the calculation.
        elemwise: bool
            Whether the calculation is performed element-wise,
            i.e. all the fields must have the same axes and in the same order.
        kwargs: dict
            List of field parameters fixed in the calculation (e.g. specific indeces_order)
        """
        assert all([isinstance(field, type(self)) for field in fields])
        # TODO: add more checks for compatibility

        if not fields and not kwargs:
            return self
        if not fields:
            # TODO: should check kwargs and do a copy only if needed
            return self.copy(**kwargs)

        fields = (self,) + list(fields)

        if elemwise:
            # TODO: should reorder the field giving the same order
            pass

        # TODO: should check for coords and restrict all the fields to the intersection

        return fields

    def __getitem__(self, coords):
        return self.get(coords)

    def get(self, *keys, **coords):
        return self.copy(coords=(keys, coords))

    def __setitem__(self, coords, value):
        return self.set(value, coords)

    def set(self, value, *keys, **coords):
        self.update(
            **self.backend.setitem(
                value, self.lattice.coordinates.resolve(*keys, **coords)
            )
        )

    def __pos__(self):
        return self


FieldType.Field = BaseField


def index_to_axis(index):
    "Converts and field index to a field axis"
    return re.sub("_[0-9]+$", "", index)


def default_operator(key, fnc=None, doc=None):
    "Default implementation of a field operator"

    def method(self, *args, **kwargs):
        if isinstance(self, BaseField):
            back = getattr(self.backend, key)(*args, **kwargs)
            if isinstance(back, tuple):
                return tuple((self.copy(**attrs) for attrs in back))
            return self.copy(**back)

        if fnc is None:
            raise TypeError(
                "First argument of %s must be of type Field. Given %s"
                % (key, type(self).__name__)
            )

        return fnc(self, *args, **kwargs)

    method.__name__ = key

    if doc:
        method.__doc__ = doc
    elif fnc:
        method.__doc__ = fnc.__doc__

    return method


OPERATORS = (
    ("__abs__",),
    ("__add__",),
    ("__radd__",),
    ("__eq__",),
    ("__gt__",),
    ("__ge__",),
    ("__lt__",),
    ("__le__",),
    ("__mod__",),
    ("__rmod__",),
    ("__mul__",),
    ("__rmul__",),
    ("__ne__",),
    ("__neg__",),
    ("__pow__",),
    ("__rpow__",),
    ("__sub__",),
    ("__rsub__",),
    ("__truediv__",),
    ("__rtruediv__",),
    ("__floordiv__",),
    ("__rfloordiv__",),
    ("__divmod__",),
    ("__rdivmod__",),
)

for (op,) in OPERATORS:
    setattr(BaseField, op, default_operator(op))


class BaseBackend:
    "Base backend for the field class"

    def __init__(self, field):
        self.field = field

    def initialize(self, field):
        "Initializes the field value"
        if field is None:
            return dict(value=self.field.indeces_order)
        return self.init(self.field)

    def __getattr__(self, value):
        def method(self, *args, **kwargs):
            raise NotImplementedError("%s not implemented" % value)

        def attr(*args, **kwargs):
            return dict(
                value=Function(method, label=value, args=[self.field.value],)(
                    *args, **kwargs
                )
            )

        attr.__name__ = value
        method.__name__ = value
        return attr
