"""
Array class of the Field type that implements
the interface to the numpy array functions
"""
# pylint: disable=C0103,C0303,C0330,W0221

__all__ = [
    "ArrayField",
    "NumpyBackend",
    #    "zeros_like",
    #    "ones_like",
]

from functools import wraps
import numpy as np
from tunable import function, tunable
from .base import BaseField, BaseBackend
from .types.base import FieldType
from ..utils import add_kwargs_of


class ArrayField(BaseField):
    """
    Array class of the Field type that implements
    the interface to the numpy array functions.
    """

    default_dtype = "complex128"

    def __init_attributes__(self, field, dtype=None, **kwargs):
        """
        Initializes the field class.
        
        Parameters
        ----------
        dtype: str or numpy dtype compatible
            Data type of the field.
        copy: bool
            Whether the input field should be copied. 
            If False the field is copied only if needed
            otherwise the input field will be used;
            if True, the field is copied.
        """
        kwargs = super().__init_attributes__(field, **kwargs)

        self._dtype = np.dtype(
            dtype
            if dtype is not None
            else field.dtype
            if hasattr(field, "dtype")
            else ArrayField.default_dtype
        )

        return kwargs

    __init__ = add_kwargs_of(__init_attributes__)(BaseField.__init__)

    def __update_value__(self, field, copy=False, **kwargs):
        if copy:
            self.update(**self.backend.copy())

        kwargs = super().__update_value__(field, **kwargs)

        if isinstance(field, ArrayField) and self.dtype != field.dtype:
            self.update(**self.backend.astype(self.dtype))

        return kwargs

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
        if np.dtype(value) != self.dtype:
            self.update(**self.backend.astype(self.dtype))

    def astype(self, dtype):
        "Changes the dtype of the field."
        if self.dtype == dtype:
            return self
        return self.copy(dtype=dtype)

    def zeros(self, dtype=None):
        "Returns the field with all components put to zero"
        return self.copy(**self.backend.zeros_like(dtype))

    def ones(self, dtype=None):
        "Returns the field with all components put to one"
        return self.copy(**self.backend.ones_like(dtype))

    @property
    def real(self):
        "Returns the real part of the field"
        # pylint: disable=E0602
        return real(self)

    @property
    def imag(self):
        "Returns the imaginary part of the field"
        # pylint: disable=E0602
        return imag(self)

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
                raise ValueError("Axis %s not in field" % (axis))
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

        axes = [
            axis
            for axis in self.get_axes(axes)
            if axis not in axes_order and counts[axis] > 1
        ]

        if not axes and not axes_order:
            return self
        return self.copy(**self.backend.transpose(*axes, **axes_order))

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

    def __matmul__(self, other):
        return self.dot(other)

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
        axes, kwargs = uniform_input_axes(*axes, **kwargs)
        indeces = self.get_indeces(axes)
        return self.copy(**self.backend.roll(shift, *indeces, **kwargs))


FieldType.Field = ArrayField
# zeros_like = default_method("zeros", fnc=np.zeros_like)
# ones_like = default_method("ones", fnc=np.ones_like)


class NumpyBackend(BaseBackend):
    "Numpy array backend for the field class"

    def initialize(self, field):
        "Initializes a new field"
        if field is None:
            return dict(
                value=function(
                    np.ndarray, self.field.ordered_shape, dtype=self.field.dtype
                )
            )

        field = np.array(field)
        if not self.field.indeces_order.fixed:
            raise ValueError(
                "Cannot initilize a field with an array without fixing the indeces_order"
            )
        if not field.shape == self.field.ordered_shape:
            raise ValueError("Shape of field and given array do not match")
        return dict(value=tunable(field, label="input"))

    def copy(self):
        "Returns a copy of the field"
        return dict(value=function(np.copy, self.field.value))

    def astype(self, dtype):
        "Changes the dtype of the field"
        return dict(value=function(np.ndarray.astype, self.field.value, dtype))

    def zeros_like(self, dtype):
        "Fills the field with zeros"
        return dict(
            dtype=dtype, value=function(np.zeros_like, self.field.value, dtype=dtype),
        )

    def ones_like(self, dtype):
        "Fills the field with ones"
        return dict(
            dtype=dtype, value=function(np.ones_like, self.field.value, dtype=dtype),
        )


def uniform_input_axes(*axes, **kwargs):
    "Auxiliary function to uniform the axes input parameters"
    axes = list(axes)
    tmp = kwargs.pop("axis", [])
    axes.extend([tmp] if isinstance(tmp, str) else tmp)
    tmp = kwargs.pop("axes", [])
    axes.extend([tmp] if isinstance(tmp, str) else tmp)

    return tuple(axes), kwargs


def wrap_method(fnc):
    "Wrapper for field methods"

    @wraps(fnc)
    def wrapped(field, *args, **kwargs):
        if not isinstance(field, ArrayField):
            raise TypeError("First argument of %s must be a field." % fnc.__name__)
        return fnc(field, *args, **kwargs)

    return wrapped


METHODS = (
    ("reorder",),
    ("squeeze",),
    ("transpose",),
    ("dagger",),
    ("roll",),
)

for (method,) in METHODS:
    __all__.append(method)
    globals()[method] = wrap_method(getattr(ArrayField, method))
