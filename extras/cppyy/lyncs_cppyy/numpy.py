"Utils for interfacing to numpy"
# pylint: disable=C0103

__all__ = [
    "dtype_map",
    "char_map",
]

import numpy as np

dtype_map = (
    (np.bool_, "bool"),
    (np.byte, "signed char"),
    (np.ubyte, "unsigned char"),
    (np.short, "short"),
    (np.ushort, "unsigned short"),
    (np.intc, "int"),
    (np.uintc, "unsigned int"),
    (np.int_, "long"),
    (np.uint, "unsigned long"),
    (np.longlong, "long long"),
    (np.ulonglong, "unsigned long long"),
    (np.half, ""),
    (np.single, "float"),
    (np.double, "double"),
    (np.longdouble, "long double"),
    (np.csingle, "float complex"),
    (np.cdouble, "double complex"),
    (np.clongdouble, "long double complex"),
    (np.int8, "int8_t"),
    (np.int16, "int16_t"),
    (np.int32, "int32_t"),
    (np.int64, "int64_t"),
    (np.uint8, "uint8_t"),
    (np.uint16, "uint16_t"),
    (np.uint32, "uint32_t"),
    (np.uint64, "uint64_t"),
    (np.intp, "intptr_t"),
    (np.uintp, "uintptr_t"),
    (np.float32, "float"),
    (np.float64, "double"),
    (np.complex64, "float complex"),
    (np.complex128, "double complex"),
)

char_map = dict((dtype().dtype.char, ctype) for dtype, ctype in dtype_map)
