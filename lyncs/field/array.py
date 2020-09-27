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
from functools import wraps
import numpy as np
from tuneit import (
    TunableClass,
    tunable_property,
    derived_property,
    Function,
    function,
    Permutation,
    Variable,
    Tunable,
    finalize,
)
from lyncs_utils import add_kwargs_of, compute_property, isiterable
from .base import BaseField, wrap_method
from .types.base import FieldType


class ArrayField(BaseField, TunableClass):
    """
    Array class of the Field type that implements
    the interface to the numpy array functions.
    """

    default_dtype = "complex128"

    @add_kwargs_of(BaseField.__init__)
    def __init_attributes__(
        self, field=None, dtype=None, indexes_order=None, labels_order=None, **kwargs
    ):
        """
        Initializes the field class.

        Parameters
        ----------
        dtype: str or numpy dtype compatible
            Data type of the field.
        indexes_order: tuple
            The order of the field indexes (field.indexes).
            This also fixes the field shape (field.ordered_shape).
            It is a tunable parameter and the decision can be postpone.
        copy: bool
            Whether the input field should be copied.
            If False the field is copied only if needed
            otherwise the input field will be used;
            if True, the field is copied.
        """
        kwargs = super().__init_attributes__(field, **kwargs)

        indexes_order = self._get_indexes_order(
            field if isinstance(field, BaseField) else None, indexes_order
        )
        if indexes_order is not None:
            self.indexes_order = indexes_order

        self._labels_order, kwargs = self._get_labels_order(
            field if isinstance(field, BaseField) else None, labels_order, **kwargs
        )

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
            if not self.indexes_order.fixed:
                raise ValueError(
                    "Cannot initilize a field with an array without fixing the indexes_order"
                )
            for key, val in self.labels_order:
                if not val.fixed:
                    raise ValueError(
                        "Cannot initilize a field with an array without fixing the %s order"
                        % key
                    )
            value = np.array(value)
            if value.shape != self.ordered_shape:
                raise ValueError("Shape of field and given array do not match")

        self.value = self.backend.init(value, self.ordered_shape, self.dtype)
        return kwargs

    def __validate_value__(self, value, **kwargs):
        "Checks if the field is well defined to have a value"

        if not self.indexes_order.fixed and not finalize(self.value).depends_on(
            self.indexes_order
        ):
            raise ValueError("Value has been given but indexes_order is not fixed.")

        for key, val in self.labels_order:
            if (
                not val.fixed
                and any((var.startswith(key) for var in self.variables))
                and not finalize(self.value).depends_on(val)
            ):
                raise ValueError(
                    "Value has been given but %s order is not fixed." % key
                )

        self.value = value

        return kwargs

    def __update_value__(self, field, copy=False, **kwargs):
        "Checks if something changed wrt field and updates the field value"

        if copy:
            self.value = self.backend.copy()

        same_indexes = set(self.indexes).intersection(field.indexes)
        indexes = field.coords.extract(same_indexes).get_indexes(
            self.coords.extract(same_indexes)
        )
        if indexes:
            labels = {key: val for key, val in field.labels_order if key in indexes}
            self.value = self.backend.getitem(field.indexes_order, indexes, **labels)

        if set(self.indexes) != set(field.indexes):
            if not self.size == field.size:
                raise ValueError("When reshaping, the size of the field cannot change")
            self.value = self.backend.reshape(
                self.ordered_shape, self.indexes_order, field.indexes_order
            )

        if self.indexes_order != field.indexes_order and self.indexes_order.size > 1:
            self.value = self.backend.reorder(self.indexes_order, field.indexes_order)

        labels = {}
        coords = {}
        old_order = dict(field.labels_order)
        for key, val in self.labels_order:
            if key in old_order and val != old_order[key] and val.size > 1:
                coords[key] = val
                labels[key] = old_order[key]
                self.value = self.backend.reorder_label(
                    key, val, old_order[key], self.indexes_order
                )

        if self.dtype != field.dtype:
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
            and self.indexes_order == other.indexes_order
        )

    __eq__ = __is__

    def __bool__(self):
        if self.dtype == "bool":
            return bool(self.all().result)
        raise ValueError(
            """
            The truth value of an field with more than one element is ambiguous.
            Use field.any() or field.all()
            """
        )

    def copy(self, value=None, **kwargs):
        "Creates a shallow copy of the field"
        return super().copy(value=value, **kwargs)

    @wraps(TunableClass.compute)
    def compute(self, **kwargs):
        "Adds consistency checks on the value"
        super().compute(**kwargs)
        array = self.node.value.obj
        assert array.shape == self.ordered_shape, "Mistmatch in the shape %s != %s" % (
            array.shape,
            self.shape,
        )
        assert array.dtype == self.dtype, "Mistmatch in the dtype %s != %s" % (
            array.dtype,
            self.dtype,
        )

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
    def indexes_order(self):
        "Order of the field indexes"
        return Permutation(self.indexes)

    def reorder(self, indexes_order=None, **kwargs):
        "Changes the indexes_order of the field."
        if indexes_order is None:
            indexes_order = self.indexes_order.copy(reset=True)
        return self.copy(indexes_order=indexes_order, **kwargs)

    def _get_indexes_order(self, field=None, indexes_order=None):
        if indexes_order is not None:
            if (
                not isinstance(indexes_order, Variable)
                and not isinstance(indexes_order, Tunable)
                and set(indexes_order) != set(self.indexes)
            ):
                raise ValueError(
                    "Not valid indexes_order. It has %s, while expected %s"
                    % (indexes_order, self.indexes)
                )
            return indexes_order
        if field is None:
            return None
        if len(self.indexes) <= 1:
            return self.indexes
        if set(self.indexes) == set(field.indexes):
            return field.indexes_order
        if set(self.indexes) <= set(field.indexes):
            return function(filter, self.indexes.__contains__, field.indexes_order)
        return None

    @property
    def labels_order(self):
        "Order of the field indexes"
        return self._labels_order

    def reorder_label(self, label, label_order=None, **kwargs):
        "Changes the order of the label."
        rng = self.get_range(label)
        if not isiterable(self.get_range(label), str):
            raise KeyError("%s is not a label of the field" % label)
        if len(rng) <= 1:
            return self.copy()
        if label_order is None:
            label_order = Permutation(rng, label=label)
        labels_order = kwargs.pop("labels_order", {})
        labels_order[label] = label_order
        return self.copy(labels_order=labels_order, **kwargs)

    def _get_labels_order(self, field=None, labels_order=None, **kwargs):
        if labels_order is not None and not isinstance(labels_order, dict):
            raise TypeError("labels_order must be a dict")
        if labels_order is None:
            labels_order = {}

        # Checking for keys in kwargs
        for key in self.labels:
            if key + "_order" in kwargs:
                labels_order[key] = kwargs.pop(key + "_order")
                continue
            key = self.index_to_axis(key)
            if key + "_order" in kwargs:
                labels_order[key] = kwargs.pop(key + "_order")
                continue

        # Here we check the given values and unpack axes into indexes
        given_values = labels_order
        labels_order = {}
        for key, val in given_values.items():
            rng = self.get_range(key)  # This does also some quality control on the key
            if (
                not isinstance(val, Variable)
                and not isinstance(val, Tunable)
                and set(val) != set(rng)
            ):
                raise ValueError(
                    "Not valid %s order. It has %s, while expected %s" % (key, val, rng)
                )
            for _k in self.get_indexes(key):
                if _k == key or _k not in given_values:
                    labels_order[_k] = val

        # Getting labels_order from the field
        if field is not None:
            for key, val in field.labels_order:
                if key in self.labels and key not in labels_order:
                    rng = self.get_range(key)
                    if len(rng) <= 1:
                        continue
                    if set(rng) == set(field.get_range(key)):
                        labels_order[key] = val
                    elif set(rng) <= set(field.get_range(key)):
                        labels_order[key] = function(filter, rng.__contains__, val)

        # Creating variables
        for key in self.labels:
            if key in labels_order and isinstance(labels_order[key], Permutation):
                continue
            rng = self.get_range(key)
            var = Permutation(rng, label=key)
            if key in labels_order:
                var.value = labels_order[key]
            labels_order[key] = var

        return tuple(labels_order.items()), kwargs

    @derived_property(indexes_order)
    def ordered_shape(self):
        "Shape of the field after fixing the indexes_order"
        shape = dict(self.shape)
        return tuple(shape[key] for key in self.indexes_order.value)

    def __setitem__(self, coords, value):
        return self.set(value, coords)

    def set(self, value, *keys, **coords):
        "Sets the components at the given coordinates"
        coords = self.lattice.coords.resolve(*keys, **coords, field=self)
        indexes = self.coords.get_indexes(coords)
        self.value = self.backend.setitem(self.indexes_order, indexes, value)

    def zeros(self, dtype=None):
        "Returns the field with all components put to zero"
        return self.copy(self.backend.zeros(dtype), dtype=dtype)

    def ones(self, dtype=None):
        "Returns the field with all components put to one"
        return self.copy(self.backend.ones(dtype), dtype=dtype)

    def random(self, seed=None):
        """
        Returns a random field generator. If seed is given, reproducibility is ensured
        independently on the field parameters, e.g. indexes_order, etc.

        Parameters
        ----------
        seed: int
            The seed to use for starting the random number generator.
            Note: There is a performance penality in initializing the field if seed is given.
        """
        from .random import RandomFieldGenerator

        return RandomFieldGenerator(self, seed)

    def rand(self, seed=None):
        "Returns a real field with random numbers distributed uniformely between [0,1)"
        # return self.random(seed).random()
        return self.copy(self.backend.rand(), dtype="float64")

    @property
    def T(self):
        "Transposes the field."
        return self.transpose()

    def transpose(self, *axes, **axes_order):
        """
        Transposes the matrix/tensor indexes of the field.

        *NOTE*: this is conceptually different from numpy.transpose
                where all the axes are transposed.

        Parameters
        ----------
        axes: str
            If given, only the listed axes are transposed,
            otherwise all the tensorial axes are changed.
            By default the order of the indexes is inverted.
        axes_order: dict
            Same as axes, but specifying the reordering of the indexes.
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
                    "%d indexes have been given for axis %s but it has count %d"
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
            self.backend.transpose(self.indexes_order, axes=axes, **axes_order)
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
        Conjugate and transposes the matrix/tensor indexes.
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
        indexes = self.get_indexes(*axes) if axes else self.get_indexes("all")
        return self.copy(self.backend.roll(shift, indexes, self.indexes_order))


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
        "Name of the method"
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
    def copy(self):
        "Returns a copy of the field"
        return self.copy()

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
    def getitem(self, indexes_order, coords, **labels):
        "Direct implementation of getitem"
        for label, order in labels.items():
            coords[label] = tuple(order.index(val) for val in coords[label])
        indexes = tuple(coords.pop(idx, slice(None)) for idx in indexes_order)
        assert not coords, "Coords didn't empty"
        return self.__getitem__(indexes)

    @backend_method
    def setitem(self, indexes_order, coords, value, **labels):
        "Direct implementation of setitem"
        for label, order in labels.items():
            coords[label] = tuple(order.index(val) for val in coords[label])
        indexes = tuple(coords.pop(idx, slice(None)) for idx in indexes_order)
        assert not coords, "Coords didn't empty"
        self.__setitem__(indexes, value)
        return self

    @backend_method
    def reorder(self, new_order, old_order):
        "Direct implementation of reordering"
        indexes = tuple(old_order.index(idx) for idx in new_order)
        return self.transpose(axes=indexes)

    @backend_method
    def reorder_label(self, key, new_order, old_order, indexes_order):
        "Direct implementation of label reordering"
        indexes = tuple(old_order.index(idx) for idx in new_order)
        return self.take(indexes, axis=indexes_order.index(key))

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
    def transpose(self, indexes_order=None, axes=None, **axes_order):
        "Direct implementation of transposing"
        indexes = defaultdict(list)
        axes_order = {key: list(val) for key, val in axes_order.items()}
        for idx in indexes_order:
            axis = BaseField.index_to_axis(idx)
            indexes[axis].append(idx)
        new_order = []
        for idx in indexes_order:
            axis = BaseField.index_to_axis(idx)
            if axis in axes:
                idx = indexes[axis].pop()
            elif axis in axes_order:
                idx = indexes[axis][axes_order[axis].pop(0)]
            else:
                idx = indexes[axis].pop(0)
        indexes = tuple(new_order.index(idx) for idx in indexes_order)
        return self.transpose(axes=indexes)

    @backend_method
    def roll(self, shift, indexes=None, indexes_order=None):
        "Direct implementation of rolling"
        indexes = tuple(indexes_order.index(idx) for idx in indexes)
        return self.roll(shift, indexes)

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
