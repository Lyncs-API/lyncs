"""
Array class of the Field type that implements
the interface to the numpy array functions
"""
# pylint: disable=C0103,C0303,C0330,W0221

__all__ = [
    "ArrayField",
    "NumpyBackend",
]

from collections import defaultdict
import numpy as np
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
)
from .base import BaseField, wrap_method
from .types.base import FieldType
from ..utils import add_kwargs_of, compute_property, isiterable


class ArrayField(BaseField, TunableClass):
    """
    Array class of the Field type that implements
    the interface to the numpy array functions.
    """

    default_dtype = "complex128"

    @add_kwargs_of(BaseField.__init__)
    def __init_attributes__(
        self, field=None, dtype=None, indeces_order=None, labels_order=None, **kwargs
    ):
        """
        Initializes the field class.
        
        Parameters
        ----------
        dtype: str or numpy dtype compatible
            Data type of the field.
        indeces_order: tuple
            The order of the field indeces (field.indeces).
            This also fixes the field shape (field.ordered_shape).
            It is a tunable parameter and the decision can be postpone.
        copy: bool
            Whether the input field should be copied. 
            If False the field is copied only if needed
            otherwise the input field will be used;
            if True, the field is copied.
        """
        kwargs = super().__init_attributes__(field, **kwargs)

        indeces_order = self._get_indeces_order(
            field if isinstance(field, BaseField) else None, indeces_order
        )
        if indeces_order is not None:
            self.indeces_order = indeces_order

        if labels_order is not None:
            for key, val in labels_order:
                self.labels_order[key].value = val

        self._dtype = np.dtype(
            dtype
            if dtype is not None
            else field.dtype
            if hasattr(field, "dtype")
            else ArrayField.default_dtype
        )

        return kwargs

    def __initialize_value__(self, value, **kwargs):
        "Initializes the value of the field"

        if value is not None:
            if not self.indeces_order.fixed:
                raise ValueError(
                    "Cannot initilize a field with an array without fixing the indeces_order"
                )
            value = np.array(value)
            if not value.shape == self.ordered_shape:
                raise ValueError("Shape of field and given array do not match")

        self.value = self.backend.init(value, self.ordered_shape, self.dtype)
        return kwargs

    def __validate_value__(self, value, **kwargs):
        "Checks if the field is well defined to have a value"

        if not self.indeces_order.fixed and not finalize(self.value).depends_on(
            self.indeces_order
        ):
            raise ValueError("Value has been given but indeces_order is not fixed.")

        self.value = value

        return kwargs

    def __update_value__(self, field, copy=False, **kwargs):
        "Checks if something changed wrt field and updates the field value"

        if copy:
            self.value = self.backend.copy()

        same_indeces = set(self.indeces).intersection(field.indeces)
        indeces = field.coords.extract(same_indeces).get_indeces(
            self.coords.extract(same_indeces)
        )
        if indeces:
            labels = {key: val for key, val in field.labels_order if key in indeces}
            self.value = self.backend.getitem(field.indeces_order, indeces, **labels)

        if set(self.indeces) != set(field.indeces):
            if not self.size == field.size:
                raise ValueError("When reshaping, the size of the field cannot change")
            self.value = self.backend.reshape(
                self.ordered_shape, self.indeces_order, field.indeces_order
            )

        if not self.indeces_order.fixed and not finalize(self.value).depends_on(
            self.indeces_order
        ):
            self.value = self.backend.reorder(self.indeces_order, field.indeces_order)

        if isinstance(field, ArrayField) and self.dtype != field.dtype:
            self.value = self.backend.astype(self.dtype)

        return kwargs

    @add_kwargs_of(__init_attributes__)
    def __init__(self, field=None, value=None, **kwargs):
        """
        Initializes the field class.
        
        Parameters
        ----------
        value: Tunable
            The underlying value of the field. Not for the faint of heart.
            If it is given, then all the attributes of the initialization
            are considered proparties of the valu and no transformation
            will be applied.
        """
        TunableClass.__init__(self, field)

        kwargs = self.__init_attributes__(field, **kwargs)

        if value is not None:
            kwargs = self.__validate_value__(value, **kwargs)
        elif isinstance(field, BaseField):
            kwargs = self.__update_value__(field, **kwargs)
        else:
            kwargs = self.__initialize_value__(field, **kwargs)

        if kwargs:
            raise ValueError("Could not resolve the following kwargs.\n %s" % kwargs)

    def __is__(self, other):
        "This is a direct implementation of __eq__, while the latter is an element wise comparison"
        return self is other or (
            super().__eq__(other)
            and self.dtype == other.dtype
            and self.node.key == other.node.key
            and self.indeces_order == other.indeces_order
        )

    __eq__ = __is__

    def __bool__(self):
        if self.dtype == "bool":
            return self.all().result
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
        )

    def copy(self, value=None, **kwargs):
        "Creates a shallow copy of the field"
        return super().copy(value=value, **kwargs)

    @property
    def backend(self):
        "Returns the computational backend of the field (numpy)."
        return NumpyBackend(self)

    @property
    def dtype(self):
        "Data type of the field (numpy style)"
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if self.dtype != value:
            self._dtype = np.dtype(value)
            self.value = self.backend.astype(self.dtype)

    def astype(self, dtype):
        "Changes the dtype of the field."
        if self.dtype == dtype:
            return self
        return self.copy(dtype=dtype)

    @compute_property
    def bytes(self):
        "Returns the size of the field in bytes"
        return self.size * self.dtype.itemsize

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
        if len(self.indeces) <= 1:
            return self.indeces
        if set(self.indeces) == set(field.indeces):
            return field.indeces_order
        if set(self.indeces) <= set(field.indeces):
            return function(filter, self.indeces.__contains__, field.indeces_order)
        return None

    @compute_property
    def labels_order(self):
        "Order of the field indeces"
        return tuple(
            (key, Permutation(self.get_range(key), label=key)) for key in self.labels
        )

    def reorder(self, indeces_order=None, **kwargs):
        "Changes the indeces_order of the field."
        indeces_order = kwargs.pop("indeces_order", indeces_order)
        if indeces_order is None:
            indeces_order = self.indeces_order.copy(reset=True)
        if not isinstance(indeces_order, Variable) and not set(indeces_order) == set(
            self.indeces
        ):
            raise ValueError("All the indeces need to be specified in the reordering")
        return self.copy(indeces_order=indeces_order, **kwargs)

    @derived_property(indeces_order)
    def ordered_shape(self):
        "Shape of the field after fixing the indeces_order"
        shape = dict(self.shape)
        return tuple(shape[key] for key in self.indeces_order.value)

    def __setitem__(self, coords, value):
        return self.set(value, coords)

    def set(self, value, *keys, **coords):
        "Sets the components at the given coordinates"
        coords = self.lattice.coords.resolve(*keys, **coords, field=self)
        indeces = self.coords.get_indeces(coords)
        self.value = self.backend.setitem(self.indeces_order, indeces, value)

    def zeros(self, dtype=None):
        "Returns the field with all components put to zero"
        return self.copy(self.backend.zeros(dtype), dtype=dtype)

    def ones(self, dtype=None):
        "Returns the field with all components put to one"
        return self.copy(self.backend.ones(dtype), dtype=dtype)

    def rand(self):
        "Returns a real field with random numbers distributed uniformely between [0,1)"
        return self.copy(self.backend.rand(), dtype="float64")

    @property
    def T(self):
        "Transposes the field."
        return self.transpose()

    def transpose(self, *axes, **axes_order):
        """
        Transposes the matrix/tensor indeces of the field.

        *NOTE*: this is conceptually different from numpy.transpose
                where all the axes are transposed.

        Parameters
        ----------
        axes: str
            If given, only the listed axes are transposed, 
            otherwise all the tensorial axes are changed.
            By default the order of the indeces is inverted.
        axes_order: dict
            Same as axes, but specifying the reordering of the indeces.
            The key must be one of the axis and the value the order using
            an index per repetition of the axis numbering from 0,1,...
        """
        counts = dict(self.axes_counts)
        for (axis, val) in axes_order.items():
            if not axis in counts:
                raise KeyError("Axis %s not in field" % (axis))
            if not isiterable(val):
                raise TypeError("Type of value for axis %s not valid" % (axis))
            val = tuple(val)
            if not len(val) == counts[axis]:
                raise ValueError(
                    "%d indeces have been given for axis %s but it has count %d"
                    % (len(val), axis, counts[axis])
                )
            if not set(val) == set(range(counts[axis])):
                raise ValueError(
                    "%s has been given for axis %s. Not a permutation of %s."
                    % (val, axis, tuple(range(counts[axis])))
                )

        if not axes and not axes_order:
            axes = ("dofs",)

        axes = [
            axis
            for axis in self.get_axes(*axes)
            if axis not in axes_order and counts[axis] > 1
        ]

        for (axis, val) in tuple(axes_order.items()):
            if val == tuple(range(counts[axis])):
                del axes_order[axis]

        if not axes and not axes_order:
            return self.copy()
        return self.copy(
            self.backend.transpose(self.indeces_order, axes=axes, **axes_order)
        )

    @property
    def iscomplex(self):
        "Returns if the field is complex"
        return self.dtype in [np.csingle, np.cdouble, np.clongdouble]

    def conj(self):
        "Conjugates the field."
        if not self.iscomplex:
            return self.copy()
        return self.copy(self.backend.conj())

    @property
    def H(self):
        "Conjugate transpose of the field."
        return self.dagger()

    def dagger(self, *axes, **axes_order):
        """
        Conjugate and transposes the matrix/tensor indeces.
        See help(transpose) for more details.
        """
        return self.conj().transpose(*axes, **axes_order)

    @classmethod
    def get_input_axes(cls, *axes, **kwargs):
        "Auxiliary function to uniform the axes input parameters"
        if not (bool(axes), "axes" in kwargs, "axis" in kwargs).count(True) <= 1:
            raise ValueError("Only one between *axes, axes= or axis= can be used")
        axes = kwargs.pop("axis", kwargs.pop("axes", axes))
        if isinstance(axes, str):
            axes = (axes,)
        if not isiterable(axes, str):
            raise TypeError("Type for axes not valid. %s" % (axes))
        return axes, kwargs

    def roll(self, shift, *axes, **kwargs):
        """
        Rolls axis of shift.
        
        Parameters:
        -----------
        shift: int or list of int
            The number of places by which elements are shifted.
        axis: str or list of str
            Axis/axes to roll of shift amount.
        """
        axes, kwargs = self.get_input_axes(*axes, **kwargs)
        if kwargs:
            raise KeyError("Unknown parameter %s" % kwargs)
        indeces = self.get_indeces(*axes) if axes else self.get_indeces("all")
        from pickle import dumps

        return self.copy(self.backend.roll(shift, indeces, self.indeces_order))


FieldType.Field = ArrayField


class backend_method:
    "Decorator for backend methods"

    def __init__(self, fnc, cls=None):
        self.fnc = fnc
        self.__name__ = self.fnc.__name__
        if cls:
            self.fnc.__qualname__ = cls.__name__ + "." + self.key
            setattr(cls, self.key, self)

    @property
    def key(self):
        return self.__name__

    def __get__(self, obj, owner):
        if obj is None:
            return self.fnc
        return Function(self.fnc, args=(obj.field.value,), label=self.key)


class NumpyBackend:
    "Numpy array backend for the field class"

    def __init__(self, field):
        self.field = field

    @classmethod
    def init(cls, field, shape, dtype):
        "Initializes a new field"
        if field is None:
            return function(np.zeros, shape, dtype=dtype)

        return function(np.array, field, dtype=dtype)

    @backend_method
    def copy(self, dtype=None):
        "Returns a copy of the field"
        return self.copy(dtype=dtype)

    @backend_method
    def astype(self, dtype):
        "Changes the dtype of the field"
        return self.astype(dtype)

    @backend_method
    def conj(self):
        "Conjugates the field"
        return self.conj()

    @backend_method
    def zeros(self, dtype):
        "Fills the field with zeros"
        return np.zeros_like(self, dtype=dtype)

    @backend_method
    def ones(self, dtype):
        "Fills the field with ones"
        return np.ones_like(self, dtype=dtype)

    @backend_method
    def rand(self):
        "Fills the field with random numbers"
        return np.random.rand(*self.shape)

    @backend_method
    def getitem(self, indeces_order, coords, **labels):
        "Direct implementation of getitem"
        for label, order in labels.items():
            coords[label] = tuple(order.index(val) for val in coords[label])
        indeces = tuple(coords.pop(idx, slice(None)) for idx in indeces_order)
        assert not coords, "Coords didn't empty"
        return self.__getitem__(indeces)

    @backend_method
    def setitem(self, indeces_order, coords, value, **labels):
        "Direct implementation of setitem"
        for label, order in labels.items():
            coords[label] = tuple(order.index(val) for val in coords[label])
        indeces = tuple(coords.pop(idx, slice(None)) for idx in indeces_order)
        assert not coords, "Coords didn't empty"
        self.__setitem__(indeces, value)
        return self

    @backend_method
    def reorder(self, new_order, old_order):
        "Direct implementation of reordering"
        indeces = tuple(old_order.index(idx) for idx in new_order)
        return self.transpose(axes=indeces)

    @backend_method
    def reshape(self, shape, new_order=None, old_order=None):
        "Direct implementation of reshaping"
        common = tuple(idx for idx in old_order if idx in new_order)
        order = list(old_order.index(idx) for idx in common)

        if order != sorted(order):
            # We need to reorder first
            for idx in range(len(old_order)):
                if idx not in order:
                    order.insert(idx, idx)
            self.transpose(axes=order)

        return self.reshape(shape)

    @backend_method
    def transpose(self, indeces_order=None, axes=None, **axes_order):
        "Direct implementation of transposing"
        indeces = defaultdict(list)
        axes_order = {key: list(val) for key, val in axes_order.items()}
        for idx in indeces_order:
            axis = BaseField.index_to_axis(idx)
            indeces[axis].append(idx)
        new_order = []
        for idx in indeces_order:
            axis = BaseField.index_to_axis(idx)
            if axis in axes:
                idx = indeces[axis].pop()
            elif axis in axes_order:
                idx = indeces[axis][axes_order[axis].pop(0)]
            else:
                idx = indeces[axis].pop(0)
        indeces = tuple(new_order.index(idx) for idx in indeces_order)
        return self.transpose(axes=indeces)

    @backend_method
    def roll(self, shift, indeces=None, indeces_order=None):
        "Direct implementation of rolling"
        indeces = tuple(indeces_order.index(idx) for idx in indeces)
        return self.roll(shift, indeces)

    def __getattr__(self, key):
        raise AttributeError("Unknown %s" % key)


METHODS = (
    "reorder",
    "transpose",
    "conj",
    "dagger",
    "roll",
)

for _ in METHODS:
    __all__.append(_)
    globals()[_] = wrap_method(_, ArrayField)
