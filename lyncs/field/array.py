"""
Array class of the Field type that implements
the interface to the numpy array functions
"""
# pylint: disable=C0303,C0330

__all__ = [
    "ArrayField",
]

import numpy as np
from .base import BaseField, default_operator
from .types.base import FieldType
from ..utils import add_kwargs_of


class ArrayField(BaseField):
    """
    Array class of the Field type that implements
    the interface to the numpy array functions.
    """

    default_dtype = "complex128"

    @add_kwargs_of(BaseField.__init__)
    def __init__(self, field=None, dtype=None, copy=False, zeros=False, **kwargs):
        """
        Initializes the field class.
        
        Parameters
        ----------
        dtype: str or numpy dtype compatible
            Data type of the field.
        zeros: bool
            Initializes the field with zeros.
        copy: bool
            Whether the input field should be copied. 
            If False the field is copied only if needed
            otherwise the input field will be used;
            if True, the field is copied.
        """

        self._dtype = np.dtype(
            dtype
            if dtype is not None
            else field.dtype
            if hasattr(field, "dtype")
            else ArrayField.default_dtype
        )

        super().__init__(field, **kwargs)

        if isinstance(field, ArrayField) and self.dtype != field.dtype:
            self.value = self.backend.astype(self.dtype)

        if zeros:
            self.value = self.backend.zeros()
        elif copy:
            self.value = self.backend.copy()

    @property
    def dtype(self):
        "Data type of the field (numpy style)"
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if np.dtype(value) != self.dtype:
            self._dtype = np.dtype(value)
            self.value = self.backend.astype(self.dtype)


FieldType.Field = ArrayField

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

for (ufunc, is_member) in UFUNCS:
    __all__.append(ufunc)
    globals()[ufunc] = default_operator(ufunc, fnc=getattr(np, ufunc))
    if is_member:
        setattr(ArrayField, ufunc, globals()[ufunc])
