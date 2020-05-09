"""
Array class of the Field type that implements
the interface to the numpy array functions
"""
# pylint: disable=C0303,C0330

__all__ = [
    "ArrayField",
]

import numpy as np
from .base import BaseField
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
