"""
Interface to gauge_field.h
"""
# pylint: disable=C0303
__all__ = [
    "GaugeField",
]

from functools import reduce
from time import time
import numpy
import cupy
from lyncs_cppyy.ll import to_pointer, array_to_pointers
from .lib import lib
from .lattice_field import LatticeField


class GaugeField(LatticeField):
    "Mimics the quda::LatticeField object"

    @LatticeField.field.setter
    def field(self, field):
        LatticeField.field.fset(self, field)
        if not str(self.dtype).startswith("float"):
            raise TypeError("GaugeField support only float type")
        if self.reconstruct == "INVALID":
            raise TypeError(f"Unrecognized field dofs {self.dofs}")

    @staticmethod
    def get_reconstruct(dofs):
        "Returns the reconstruct type of dofs"
        dofs = reduce((lambda x, y: x * y), dofs)
        if dofs == 18:
            return "NO"
        if dofs == 12:
            return "12"
        if dofs == 8:
            return "8"
        if dofs == 10:
            return "10"
        return "INVALID"

    @property
    def reconstruct(self):
        "Reconstruct type of the field"
        geo = self.geometry
        if geo == "INVALID":
            return "INVALID"
        if geo == "SCALAR" and self.dofs[0] == 1:
            return self.get_reconstruct(self.dofs[1:])
        if geo != "SCALAR":
            return self.get_reconstruct(self.dofs[1:])
        return self.get_reconstruct(self.dofs)

    @property
    def quda_reconstruct(self):
        "Quda enum for reconstruct type of the field"
        return getattr(lib, f"QUDA_RECONSTRUCT_{self.reconstruct}")

    @property
    def geometry(self):
        """
        Geometry of the field 
            VECTOR = all links
            SCALAR = one link
            TENSOR = Fmunu antisymmetric (upper triangle)
        """
        if self.dofs[0] == self.ndims:
            return "VECTOR"
        if self.dofs[0] == 1:
            return "SCALAR"
        if self.dofs[0] == self.ndims * (self.ndims - 1) / 2:
            return "TENSOR"
        if self.get_reconstruct(self.dofs) != "INVALID":
            return "SCALAR"
        return "INVALID"

    @property
    def quda_geometry(self):
        "Quda enum for geometry of the field"
        return getattr(lib, f"QUDA_{self.geometry}_GEOMETRY")

    @property
    def ghost_exchange(self):
        "Ghost exchange"
        return "NO"

    @property
    def quda_ghost_exchange(self):
        "Quda enum for ghost exchange"
        return getattr(lib, f"QUDA_GHOST_EXCHANGE_{self.ghost_exchange}")

    @property
    def order(self):
        "Data order of the field"
        return "FLOAT2"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_GAUGE_ORDER")

    @property
    def t_boundary(self):
        "Boundary conditions in time"
        return "PERIODIC"

    @property
    def quda_t_boundary(self):
        "Quda enum for boundary conditions in time"
        return getattr(lib, f"QUDA_{self.t_boundary}_T")

    @property
    def quda_params(self):
        "Returns and instance of quda::GaugeFieldParams"
        params = lib.GaugeFieldParam(
            self.quda_dims,
            self.quda_precision,
            self.quda_reconstruct,
            self.pad,
            self.quda_geometry,
            self.quda_ghost_exchange,
        )
        params.link_type = lib.QUDA_SU3_LINKS
        params.gauge = to_pointer(self.ptr)
        params.create = lib.QUDA_REFERENCE_FIELD_CREATE
        params.location = self.quda_location
        params.t_boundary = self.quda_t_boundary
        params.order = self.quda_order
        return params

    @property
    def quda_field(self):
        "Returns and instance of quda::GaugeField"
        return lib.GaugeField.Create(self.quda_params)

    def zero(self):
        "Set all field elements to zero"
        self.quda_field.zero()

    def gaussian(self, epsilon=1, seed=None):
        """
        Generates Gaussian distributed su(N) or SU(N) fields.  
        If U is a momentum field, then generates a random Gaussian distributed
        field in the Lie algebra using the anti-Hermitation convention.
        If U is in the group then we create a Gaussian distributed su(n)
        field and exponentiate it, e.g., U = exp(sigma * H), where H is
        the distributed su(n) field and sigma is the width of the
        distribution (sigma = 0 results in a free field, and sigma = 1 has
        maximum disorder).
        """
        seed = seed or int(time() * 1e9)
        lib.gaugeGauss(self.quda_field, seed, epsilon)

    def plaquette(self):
        """
        Computes the plaquette of the gauge field 
        
        Returns
        -------
        tuple(total, spatial, temporal) plaquette site averaged and
        normalized such that each  plaquette is in the range [0,1]
        """
        plaq = lib.plaquette(self.quda_field)
        return plaq.x, plaq.y, plaq.z

    def topological_charge(self):
        "Computes the topological charge"
        return lib.computeQCharge(self.quda_field)

    def norm1(self, link_dir=-1):
        "Computes the L1 norm of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.norm1(link_dir)

    def norm2(self, link_dir=-1):
        "Computes the L2 norm of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.norm2(link_dir)

    def abs_max(self, link_dir=-1):
        "Computes the absolute maximum of the field (Linfinity norm)"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.abs_max(link_dir)

    def abs_min(self, link_dir=-1):
        "Computes the absolute minimum of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.abs_min(link_dir)

    def compute_paths(self, paths, coeffs=None, add_to=None, add_coeff=1):
        """
        Computes the gauge paths on the lattice.
        
        The same paths are computed for every direction.

        - The paths are given with respect to direction "1" and 
          this must be the first number of every path list.
        - Directions go from 1 to self.ndims
        - Negative value (-1,...) means backward movement in the direction
        - Paths are then rotated for every direction.
        """

        if coeffs is None:
            coeffs = [1] * len(paths)
        if not len(paths) == len(coeffs):
            raise ValueError("Paths and coeffs must have the same length")

        num_paths = len(paths)
        coeffs = numpy.array(coeffs, dtype="float64")
        lengths = numpy.array(list(map(len, paths)), dtype="int32") - 1
        max_length = int(lengths.max())
        paths_array = numpy.zeros((self.ndims, num_paths, max_length), dtype="int32")

        for i, path in enumerate(paths):
            if min(path) < -self.ndims:
                raise ValueError(
                    f"Path {i} = {path} has direction smaller than {-self.ndims}"
                )
            if max(path) > self.ndims:
                raise ValueError(
                    f"Path {i} = {path} has direction larger than {self.ndims}"
                )
            if path[0] != 1:
                raise ValueError(f"Path {i} = {path} does not start with 1")
            if 0 in path:
                raise ValueError(f"Path {i} = {path} has zeros")

            for dim in range(self.ndims):
                for j, step in enumerate(path[1:]):
                    if step > 0:
                        paths_array[dim, i, j] = (step - 1 + dim) % 4
                    else:
                        paths_array[dim, i, j] = 7 - (-step - 1 + dim) % 4

        if add_to is None:
            shape = (self.ndims, 10,) + self.dims
            add_to = GaugeField(cupy.zeros(shape, dtype=self.dtype))

        quda_paths_array = array_to_pointers(paths_array)
        lib.gaugeForce(
            add_to.quda_field,
            self.quda_field,
            add_coeff,
            quda_paths_array.view,
            lengths,
            coeffs,
            num_paths,
            max_length,
        )
        return add_to
