"""
Interface to lattice_field.h
"""

__all__ = [
    "LatticeField",
]

from array import array
import numpy
import cupy
from .lib import lib


class LatticeField:
    "Mimics the quda::LatticeField object"

    def __init__(self, field):
        self.field = field

    @property
    def field(self):
        "The underlaying lattice field"
        return self._field

    @field.setter
    def field(self, field):
        if not isinstance(field, (numpy.ndarray, cupy.ndarray)):
            raise TypeError("Supporting only numpy or cupy for field")
        if isinstance(field, cupy.ndarray) and field.device.id != lib.device_id:
            raise TypeError("Field is stored on a different device than the quda lib")
        if len(field.shape) < 4:
            raise ValueError("A lattice field should not have shape smaller than 4")
        self._field = field

    @property
    def shape(self):
        "Shape of the field"
        return self.field.shape

    @property
    def location(self):
        "Memory location of the field (CPU or CUDA)"
        return "CPU" if isinstance(self.field, numpy.ndarray) else "CUDA"

    @property
    def quda_location(self):
        "Quda enum for memory location of the field (CPU or CUDA)"
        return getattr(lib, f"QUDA_{self.location}_FIELD_LOCATION")

    @property
    def ndims(self):
        "Number of lattice dimensions"
        return 4

    @property
    def dims(self):
        "Shape of the lattice dimensions"
        return self.shape[-self.ndims :]

    @property
    def quda_dims(self):
        "Memory array with lattice dimensions"
        return array("i", self.dims)

    @property
    def dofs(self):
        "Shape of the per-site degrees of freedom"
        return self.shape[: -self.ndims]

    @property
    def dtype(self):
        "Field data type"
        return self.field.dtype

    @property
    def precision(self):
        "Field data type precision"
        if str(self.dtype).endswith("64"):
            return "DOUBLE"
        if str(self.dtype).endswith("32"):
            return "SINGLE"
        if str(self.dtype).endswith("16"):
            return "HALF"
        return "INVALID"

    @property
    def quda_precision(self):
        "Quda enum for field data type precision"
        return getattr(lib, f"QUDA_{self.precision}_PRECISION")

    @property
    def pad(self):
        "Memory padding"
        return 0

    @property
    def ptr(self):
        "Memory pointer"
        if isinstance(self.field, numpy.ndarray):
            return self.field.__array_interface__["data"][0]
        return self.field.data.ptr

    @property
    def quda_params(self):
        "Returns and instance of quda::LatticeFieldParam"
        return lib.LatticeFieldParam(
            self.ndims, self.quda_dims, self.pad, self.quda_precision
        )
