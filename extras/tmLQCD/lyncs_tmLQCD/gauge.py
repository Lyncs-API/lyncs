"""
Functions for gauge field
"""

__all__ = [
    "Gauge",
    "get_g_gauge_field",
]

from numpy import array, prod, frombuffer
from lyncs_cppyy.ll import array_to_pointers, to_pointer, addressof
from .lib import lib


class Gauge:
    "Interface for gauge fields"

    def __init__(self, arr, copy=False):
        arr = array(arr, copy=copy)
        if len(arr.shape) != 4 + 1 + 2 or arr.shape[-3:] != (4, 3, 3):
            raise ValueError("Array must have shape (X,Y,Z,T,4,3,3)")
        if arr.dtype != "complex128":
            raise TypeError("Expected a complex field")

        lib.initialize(*arr.shape[:4])
        self.shape = arr.shape
        self._field = arr.reshape((-1, 4 * 9))
        self._pointers = array_to_pointers(self._field)

    @property
    def field(self):
        "The array field"
        return self._field.reshape(self.shape)

    @property
    def su3_field(self):
        "su3 view of the field"
        return to_pointer(self._pointers.ptr, "su3 **")

    @property
    def volume(self):
        "The total lattice volume"
        return lib.VOLUME * lib.g_nproc

    def volume_plaquette(self, coeff=0):
        """
        Returns the sum over the plaquettes.
        coeff is used to weight differently the spatial and temporal plaquette, having
        P(c) = (1+c) P_time + (1-c) P_space
        """
        if coeff == 0:
            return lib.measure_plaquette(self.su3_field)
        return lib.measure_gauge_action(self.su3_field, coeff)

    def plaquette(self):
        "Returns the averaged plaquette"
        return self.volume_plaquette() / self.volume / 6

    def temporal_plaquette(self):
        "Returns the averaged temporal plaquette"
        return self.volume_plaquette(1) / self.volume / 6

    def spatial_plaquette(self):
        "Returns the averaged spatial plaquette"
        return self.volume_plaquette(-1) / self.volume / 6

    def volume_rectangles(self):
        "Returns the sum over the rectangles"
        return lib.measure_rectangles(self.su3_field)

    def rectangles(self):
        "Returns the averaged rectangles"
        return self.volume_rectangles() / self.volume / 12

    def gauge_action(self, plaq_coeff=0, rect_coeff=0):
        """
        Returns the gauge action.

        The coefficients are use as follows

            (1-8*c1) ((1+c0) P_time + (1-c0) P_space) + c1*R

        where P is the sum over plaquettes, R over the rectangles,
        c0 is the plaq_coeff (see volume_plaquette) and c1 the rect_coeff
        """
        return (1 - 8 * rect_coeff) * self.volume_plaquette(plaq_coeff) + (
            rect_coeff * self.rectangles() if rect_coeff != 0 else 0
        )

    def symanzik_gauge_action(self, plaq_coeff=0):
        "Returns the tree-level Symanzik improved gauge action"
        return self.gauge_action(plaq_coeff, -1 / 12)

    def iwasaki_gauge_action(self, plaq_coeff=0):
        "Returns the Iwasaki gauge action"
        return self.gauge_action(plaq_coeff, -0.331)

    def unity(self):
        "Creates a unity field"
        self.field[:] = 0
        self.field.reshape(-1, 9)[:, (0, 4, 8)] = 1

    def random(self, repro=False):
        "Creates a random field"
        lib.random_gauge_field(repro, self.su3_field)


def get_g_gauge_field():
    "Returns the global gauge field in usage by tmLQCD"
    assert lib.initialized
    shape = (lib.LX, lib.LY, lib.LZ, lib.T, 4, 3, 3)
    ptr = to_pointer(addressof(lib.g_gauge_field[0]))
    ptr = to_pointer(ptr[0], "double *")
    ptr.reshape((int(prod(shape)) * 2,))
    return Gauge(frombuffer(ptr, dtype="complex", count=prod(shape)).reshape(shape))


def get_g_iup():
    "Returns the neighboring indeces defined by tmLQCD"
    assert lib.initialized
    shape = (lib.LX, lib.LY, lib.LZ, lib.T, 4)
    ptr = to_pointer(addressof(lib.g_iup))
    ptr = to_pointer(ptr[0], "int *")
    ptr.reshape((int(prod(shape)),))
    return frombuffer(ptr, dtype="int32", count=prod(shape)).reshape(shape)
