"""
Universal functions for the fields
"""

__all__ = [
    "prepare",
]

import numpy as np
from .array import ArrayField, NumpyBackend, backend_method


def prepare(self, *fields, elemwise=True, **kwargs):
    """
    Prepares a set of fields for a calculation.

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
        List of field parameters fixed in the calculation (e.g. specific indexes_order)
    """
    if not isinstance(self, ArrayField):
        raise ValueError("First field is not of ArrayField type")
    for idx, field in enumerate(fields):
        if not isinstance(field, type(self)):
            raise ValueError(
                "Field #%d of type %s is not compatible with %s"
                % (idx + 1, type(field), type(self))
            )

    # TODO: add more checks for compatibility

    if not fields and not kwargs:
        return self, ()
    if not fields:
        # TODO: should check kwargs and do a copy only if needed
        return self.copy(**kwargs), ()

    if elemwise:
        # TODO: should reorder the field giving the same order
        pass

    # TODO: should check for coords and restrict all the fields to the intersection

    return self, fields


ArrayField.prepare = prepare


def ufunc_method(key, elemwise=True, fnc=None, doc=None):
    """
    Implementation of a field ufunc

    Parameters
    ----------
    key: str
        The key of the method
    elemwise: bool
        Whether the calculation is performed element-wise,
        i.e. all the fields must have the same axes and in the same order.
    fnc: callable
        Fallback for the method in case self it is not a field
    """

    def method(self, *args, **kwargs):
        if not isinstance(self, ArrayField):
            if fnc is None:
                raise TypeError(
                    "First argument of %s must be of type Field. Given %s"
                    % (key, type(self).__name__)
                )

            return fnc(self, *args, **kwargs)

        # Deducing the dtype of the output
        tmp_args = (
            np.ones((1), dtype=arg.dtype) if isinstance(arg, ArrayField) else arg
            for arg in args
        )
        if fnc is not None:
            trial = fnc(np.ones((1), dtype=self.dtype), *tmp_args, **kwargs)
        else:
            trial = getattr(np.ones((1), dtype=self.dtype), key)(*tmp_args, **kwargs)

        # Uniforming the fields involved
        args = list(args)
        i_fields = tuple(
            (i, arg) for i, arg in enumerate(args) if isinstance(arg, ArrayField)
        )
        self, fields = self.prepare(
            *(field for (_, field) in i_fields), elemwise=elemwise
        )

        for (i, _), field in zip(i_fields, fields):
            args[i] = field.value

        result = getattr(self.backend, key)(*args, **kwargs)
        if isinstance(trial, tuple):
            return tuple(
                (
                    self.copy(result[i], dtype=trial.dtype)
                    for i, trial in enumerate(trial)
                )
            )
        return self.copy(result, dtype=trial.dtype)

    method.__name__ = key

    if doc:
        method.__doc__ = doc
    elif fnc:
        method.__doc__ = fnc.__doc__

    return method


def comparison(key, eq=True):
    "Additional wrapper for comparisons"
    fnc = ufunc_method(key)

    def method(self, other):
        if self.__is__(other):
            return eq
        return fnc(self, other)

    return method


def backend_ufunc_method(key, fnc=None, doc=None):
    """
    Returns a method for the backend that calls
    the given method (key) of the field value.
    """

    def method(self, *args, **kwargs):
        if fnc is None:
            return getattr(self, key)(*args, **kwargs)
        return fnc(self, *args, **kwargs)

    method.__name__ = key
    if doc is not None:
        method.__doc__ = doc
    elif fnc is not None:
        method.__doc__ = fnc.__doc__

    return method


OPERATORS = (
    ("__abs__",),
    ("__add__",),
    ("__radd__",),
    ("__mod__",),
    ("__rmod__",),
    ("__mul__",),
    ("__rmul__",),
    ("__neg__",),
    ("__pow__",),
    ("__rpow__",),
    ("__sub__",),
    ("__rsub__",),
    ("__truediv__",),
    ("__rtruediv__",),
    ("__floordiv__",),
    ("__rfloordiv__",),
)

for (op,) in OPERATORS:
    setattr(ArrayField, op, ufunc_method(op))
    backend_method(backend_ufunc_method(op), NumpyBackend)

COMPARISONS = (
    ("__eq__", True),
    ("__gt__", False),
    ("__ge__", True),
    ("__lt__", False),
    ("__le__", True),
    ("__ne__", False),
)

for (op, eq) in COMPARISONS:
    setattr(ArrayField, op, comparison(op, eq))
    backend_method(backend_ufunc_method(op), NumpyBackend)


UFUNCS = (
    # math operations
    ("add", True),
    ("subtract", True),
    ("multiply", True),
    ("divide", True),
    ("logaddexp", False),
    ("logaddexp2", False),
    ("true_divide", True),
    ("floor_divide", True),
    ("negative", True),
    ("power", True),
    ("float_power", True),
    ("remainder", True),
    ("mod", True),
    ("fmod", True),
    ("conj", False),
    ("exp", False),
    ("exp2", False),
    ("log", False),
    ("log2", False),
    ("log10", False),
    ("log1p", False),
    ("expm1", False),
    ("sqrt", True),
    ("square", True),
    ("cbrt", False),
    ("reciprocal", True),
    # trigonometric functions
    ("sin", False),
    ("cos", False),
    ("tan", False),
    ("arcsin", False),
    ("arccos", False),
    ("arctan", False),
    ("arctan2", False),
    ("hypot", False),
    ("sinh", False),
    ("cosh", False),
    ("tanh", False),
    ("arcsinh", False),
    ("arccosh", False),
    ("arctanh", False),
    ("deg2rad", False),
    ("rad2deg", False),
    # comparison functions
    ("greater", True),
    ("greater_equal", True),
    ("less", True),
    ("less_equal", True),
    ("not_equal", True),
    ("equal", True),
    ("isneginf", False),
    ("isposinf", False),
    ("logical_and", False),
    ("logical_or", False),
    ("logical_xor", False),
    ("logical_not", False),
    ("maximum", False),
    ("minimum", False),
    ("fmax", False),
    ("fmin", False),
    # floating functions
    ("isfinite", True),
    ("isinf", True),
    ("isnan", True),
    ("signbit", False),
    ("copysign", False),
    ("nextafter", False),
    ("spacing", False),
    ("modf", False),
    ("ldexp", False),
    ("frexp", False),
    ("fmod", False),
    ("floor", True),
    ("ceil", True),
    ("trunc", False),
    ("round", True),
    # more math routines
    ("degrees", False),
    ("radians", False),
    ("rint", True),
    ("fabs", True),
    ("sign", True),
    ("absolute", True),
    # non-ufunc elementwise functions
    ("clip", True),
    ("isreal", False),
    ("iscomplex", False),
    ("real", False),
    ("imag", False),
    ("fix", False),
    ("i0", False),
    ("sinc", False),
    ("nan_to_num", True),
    ("isclose", True),
    ("allclose", True),
)

for (ufunc, is_member) in UFUNCS:
    __all__.append(ufunc)
    globals()[ufunc] = ufunc_method(ufunc, fnc=getattr(np, ufunc))
    if is_member:
        setattr(ArrayField, ufunc, globals()[ufunc])
    if hasattr(np.ndarray, ufunc):
        fnc = backend_ufunc_method(ufunc, doc=getattr(np, ufunc).__doc__)
    else:
        fnc = backend_ufunc_method(ufunc, fnc=getattr(np, ufunc))
    backend_method(fnc, NumpyBackend)

setattr(ArrayField, "real", property(globals()["real"]))
setattr(ArrayField, "imag", property(globals()["imag"]))
