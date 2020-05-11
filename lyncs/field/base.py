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
from itertools import permutations
from numpy import arange
from tunable import TunableClass, tunable_property, derived_property, Function
from .types.base import FieldType
from ..utils import default_repr, compute_property


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

        coords = coords or ()

        if isinstance(field, BaseField):
            super().__init__(field if value is None else value)
            self._lattice = (lattice or field.lattice).freeze()
            self._axes = self.lattice.expand(axes or field.axes)
            self._coords = self.lattice.coordinates.resolve(
                *coords, **dict(field.coords)
            )
        else:
            super().__init__(value)
            self._lattice = (lattice or default_lattice()).freeze()
            self._axes = self.lattice.expand(axes or lattice.dims)
            self._coords = self.lattice.coordinates.resolve(*coords)

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
                key=lambda item: len(self.lattice.expand(item[1].axes.expand)),
                reverse=True,
            )
        )

        if value is None and not isinstance(field, BaseField):
            self.update(**self.backend.initialize(field))

        if (
            value is None
            and isinstance(field, BaseField)
            and self.coords != field.coords
        ):
            self.update(**self.backend.getitem(self.coords, field.coords))

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
    def dims(self):
        "List of dims in the field axes"
        return tuple(key for key in self.axes if key in self.lattice.dims)

    @compute_property
    def dofs(self):
        "List of dofs in the field axes"
        return tuple(key for key in self.axes if key in self.lattice.dofs)

    @compute_property
    def indeces(self):
        """
        List of indeces of the field. Similar to .axes but repeted axis are numerated.
        Order is not significant. See field.indeces_order.
        """
        counts = Counter(self.axes)
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
        return permutations(self.indeces)

    @compute_property
    def shape(self):
        "Returns the list of indeces with size. Order is not significant."

        def get_size(key):
            axis = re.sub("_[0-9]+$", "", key)
            if key in self.coords:
                return len(arange(self.lattice[axis])[self.coords[key]])
            return self.lattice[axis]

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

    def __getitem__(self, coords):
        return self.copy(coords=coords)

    def __setitem__(self, coords, value):
        if not isinstance(coords, tuple):
            coords = (coords,)
        self.update(
            **self.backend.setitem(self.lattice.coordinates.resolve(*coords), value)
        )


FieldType.Field = BaseField


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
