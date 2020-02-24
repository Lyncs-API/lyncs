"""
Dynamic interface to dask array methods
"""

from importlib import import_module
from functools import wraps
from dask import array as dask
import numpy


__all__ = [
    "FieldMethods",
]


class FieldMethods:
    pass


ufuncs = (
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "ceil",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "expm1",
    "fabs",
    "fix",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "fmod",
    "frexp",
    "hypot",
    "imag",
    "iscomplex",
    "isfinite",
    "isinf",
    "isnan",
    "isreal",
    "ldexp",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "nextafter",
    "rad2deg",
    "radians",
    "real",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
)

def wrap_ufunc(ufunc):
    @wraps(ufunc)
    def wrapped(field, *args, **kwargs):
        from .field import Field
        assert isinstance(field, Field), "First argument must be a field type"
        field = Field(field)
        field.field = ufunc(field.field, *args, **kwargs)
        return field
    return wrapped

for name in ufuncs:
    ufunc = getattr(dask,name)
    globals()[name] = wrap_ufunc(ufunc)
    __all__.append(name)
