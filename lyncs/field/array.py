"""
Array class of the Field type that implements
the interface to the numpy array functions
"""
# pylint: disable=C0103,C0303,C0330,W0221

__all__ = [
    "ArrayField",
    "NumpyBackend",
    "zeros_like",
    "ones_like",
]

from functools import wraps
import numpy as np
from tunable import function, tunable
from .base import BaseField, BaseBackend, default_method, OPERATORS, index_to_axis
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
zeros_like = default_method("zeros", fnc=np.zeros_like)
ones_like = default_method("ones", fnc=np.ones_like)


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


def wrap_ufunc(fnc):
    "Wrapper for numpy ufunc"

    @wraps(fnc)
    def wrapped(self, *args, **kwargs):

        args = [self.field,] + list(args)

        # Deducing the number of outputs and the output dtype
        tmp_args = (
            np.ones((1), dtype=arg.dtype) if isinstance(arg, ArrayField) else arg
            for arg in args
        )
        trial = fnc(*tmp_args, **kwargs)

        # Uniforming the fields involved
        i_fields = (
            (i, arg) for i, arg in enumerate(args) if isinstance(arg, ArrayField)
        )
        fields = self.field.prepare(*(field for (_, field) in i_fields), elemwise=True)

        for (i, _), field in zip(i_fields, fields):
            args[i] = field.value

        # Calling ufunc
        res = function(fnc, *args, **kwargs)

        if isinstance(trial, tuple):
            return tuple(
                dict(field=fields[0], value=res[i], dtype=part.dtype)
                for i, part in enumerate(trial)
            )

        return dict(field=fields[0], value=res, dtype=trial.dtype)

    return wrapped


def uniform_input_axes(*axes, **kwargs):
    "Auxiliary function to uniform the axes input parameters"
    axes = list(axes)
    tmp = kwargs.pop("axis", [])
    axes.extend([tmp] if isinstance(tmp, str) else tmp)
    tmp = kwargs.pop("axes", [])
    axes.extend([tmp] if isinstance(tmp, str) else tmp)

    return tuple(axes), kwargs


def wrap_reduction(fnc):
    "Wrapper for reduction functions"

    @wraps(fnc)
    def wrapped(self, *axes, **kwargs):

        # Extracting the axes to reduce
        axes, kwargs = uniform_input_axes(*axes, **kwargs)
        dtype = fnc(np.ones((1,), dtype=self.field.dtype), **kwargs).dtype
        if axes:
            reduce = self.field.get_indeces(*axes)
            indeces = list(self.field.indeces)
            for idx in set(reduce):
                indeces.remove(idx)

            axes = [index_to_axis(idx) for idx in indeces]
            indeces_order = self.field.get_indeces_order(indeces)
            kwargs["axis"] = self.field.get_indeces_index(*reduce)
        else:
            axes = ()
            indeces_order = ()

        return dict(
            axes=axes,
            value=function(fnc, self.field.value, **kwargs),
            dtype=dtype,
            indeces_order=indeces_order,
        )

    return wrapped


UFUNCS = (
    # math operations
    ("add", True,),
    ("subtract", True,),
    ("multiply", True,),
    ("divide", True,),
    ("logaddexp", False,),
    ("logaddexp2", False,),
    ("true_divide", True,),
    ("floor_divide", True,),
    ("negative", True,),
    ("power", True,),
    ("float_power", True,),
    ("remainder", True,),
    ("mod", True,),
    ("fmod", True,),
    ("conj", True,),
    ("exp", False,),
    ("exp2", False,),
    ("log", False,),
    ("log2", False,),
    ("log10", False,),
    ("log1p", False,),
    ("expm1", False,),
    ("sqrt", True,),
    ("square", True,),
    ("cbrt", False,),
    ("reciprocal", True,),
    # trigonometric functions
    ("sin", False,),
    ("cos", False,),
    ("tan", False,),
    ("arcsin", False,),
    ("arccos", False,),
    ("arctan", False,),
    ("arctan2", False,),
    ("hypot", False,),
    ("sinh", False,),
    ("cosh", False,),
    ("tanh", False,),
    ("arcsinh", False,),
    ("arccosh", False,),
    ("arctanh", False,),
    ("deg2rad", False,),
    ("rad2deg", False,),
    # comparison functions
    ("greater", True,),
    ("greater_equal", True,),
    ("less", True,),
    ("less_equal", True,),
    ("not_equal", True,),
    ("equal", True,),
    ("isneginf", False,),
    ("isposinf", False,),
    ("logical_and", False,),
    ("logical_or", False,),
    ("logical_xor", False,),
    ("logical_not", False,),
    ("maximum", False,),
    ("minimum", False,),
    ("fmax", False,),
    ("fmin", False,),
    # floating functions
    ("isfinite", True,),
    ("isinf", True,),
    ("isnan", True,),
    ("signbit", False,),
    ("copysign", False,),
    ("nextafter", False,),
    ("spacing", False,),
    ("modf", False,),
    ("ldexp", False,),
    ("frexp", False,),
    ("fmod", False,),
    ("floor", True,),
    ("ceil", True,),
    ("trunc", False,),
    ("round", True,),
    # more math routines
    ("degrees", False,),
    ("radians", False,),
    ("rint", True,),
    ("fabs", True,),
    ("sign", True,),
    ("absolute", True,),
    # non-ufunc elementwise functions
    ("clip", True,),
    ("isreal", True,),
    ("iscomplex", True,),
    ("real", False,),
    ("imag", False,),
    ("fix", False,),
    ("i0", False,),
    ("sinc", False,),
    ("nan_to_num", True,),
    ("isclose", True,),
    ("allclose", True,),
)

REDUCTIONS = (
    ("any",),
    ("all",),
    ("min",),
    ("max",),
    ("argmin",),
    ("argmax",),
    ("sum",),
    ("prod",),
    ("mean",),
    ("std",),
    ("var",),
)

for (ufunc, is_member) in UFUNCS:
    __all__.append(ufunc)
    globals()[ufunc] = default_method(ufunc, fnc=getattr(np, ufunc))
    if is_member:
        setattr(ArrayField, ufunc, globals()[ufunc])
    setattr(NumpyBackend, ufunc, wrap_ufunc(getattr(np, ufunc)))

for (ufunc,) in OPERATORS:
    setattr(NumpyBackend, ufunc, wrap_ufunc(getattr(np.ndarray, ufunc)))

for (reduction,) in REDUCTIONS:
    __all__.append(reduction)
    globals()[reduction] = default_method(reduction, fnc=getattr(np, reduction))
    if is_member:
        setattr(ArrayField, reduction, globals()[reduction])
    setattr(NumpyBackend, reduction, wrap_reduction(getattr(np, reduction)))


def wrap_method(fnc):
    "Wrapper for field methods"

    @wraps(fnc)
    def wrapped(field, *args, **kwargs):
        if not isinstance(field, ArrayField):
            raise TypeError("First argument of %s must be a field." % fnc.__name__)
        return fnc(field, *args, **kwargs)

    return wrapped


METHODS = (
    ("astype",),
    ("reorder",),
    ("squeeze",),
    ("transpose",),
    ("dagger",),
    ("roll",),
)

for (method,) in METHODS:
    __all__.append(method)
    globals()[method] = wrap_method(getattr(ArrayField, method))

setattr(NumpyBackend, "astype", wrap_ufunc(np.ndarray.astype))
