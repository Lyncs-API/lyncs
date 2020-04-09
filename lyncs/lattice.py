__all__ = [
    "default_lattice",
    "Lattice",
]

from types import MappingProxyType
from dask.base import normalize_token
from .utils import default_repr, default_property


def default_lattice():
    assert Lattice.last_defined is not None, "Any lattice has been defined yet."
    return Lattice.last_defined


class Lattice:
    """
    Lattice base class.
    A container for all the information on the lattice theory.
    """

    last_defined = None
    default_dims_labels = ["t", "x", "y", "z"]
    theories = {
        "QCD": {"spin": 4, "color": 3, "properties": {"gauge_dofs": ["color"],},},
    }
    __slots__ = ["_dims", "_dofs", "_properties", "_frozen"]
    __repr__ = default_repr

    def __init__(
        self, dims=4, dofs="QCD", properties=None,
    ):
        """
        Lattice initializer.

        Notation
        --------
        Dimensions: (dims) are labelled axes of the Lattice which size is variable.
            The volume of the lattice, i.e. number of sites, is given by the product
            of dims. Dims are usually the axes where one can parallelize on.
        Degrees of Freedoms: (dofs) are labelled local axes with fixed size.
        Axes: Any of the dimensions or degree of freedoms.

        Parameters
        ----------
        dims: int, list or dict (default 4)
            Dimensions (default labels: t,x,y,z if less than 5 or dim_0/1/2...)
            - int: number of dimensions. The default labels will be used.
            - list: size of the dimensions. The default labels will be used.
            - dict: labels of the dimensions (keys) and sizes (value)
        dofs: str, int, list, dict (default QCD)
            Specifies local degree of freedoms. (default naming: dof_0/1/2...)
            - str: one of the labeled theories (QCD,...). See Lattice.theories
            - int: size of one degree of freedom
            - list: size per dimension of the degrees of freedom
            - dict: labels of the degree of freedom (keys) and sizes (value)
        properties: dict
            Re-labelling or grouping of the dimensions. Each entry of the dictionary
            must contain a str or a list of strings which name refers to either another
            property or one of the dimensions or degree of freedoms.
        """
        self._frozen = False
        self._dims = {}
        self._dofs = {}
        self._properties = {}
        self.dims = dims
        self.dofs = dofs
        if properties is not None:
            self.properties = properties

        Lattice.last_defined = self

    @property
    def frozen(self):
        return getattr(self, "_frozen", False)

    @frozen.setter
    def frozen(self, value):
        if value != self.frozen:
            assert value is True, "Frozen can be only changed to True"
            self.freeze()

    def freeze(self):
        self._dims = MappingProxyType(self._dims)
        self._dofs = MappingProxyType(self._dofs)
        self._properties = MappingProxyType(self._properties)
        self._frozen = True

    @property
    def dims(self):
        return getattr(self, "_dims", {})

    @property
    def n_dims(self):
        return len(self._dims)

    @dims.setter
    def dims(self, value):

        if not value:
            self._dims = {}

        elif isinstance(value, int):
            assert value > 0, "Non-positive number of dimensions"
            self.dims = [1] * value
            return

        elif isinstance(value, (list, tuple)):
            assert all(
                (isinstance(v, int) and v > 0 for v in value)
            ), "All entries of the list must be positive integers"

            if len(value) <= len(Lattice.default_dims_labels):
                self._dims = {
                    Lattice.default_dims_labels[i]: v for i, v in enumerate(value)
                }
            else:
                self._dims = {"dim_%d" % i: v for i, v in enumerate(value)}

        elif isinstance(value, (dict, MappingProxyType)):
            assert all(
                (isinstance(v, int) and v > 0 for v in value.values())
            ), "All entries of the dictionary must be positive integers"

            self._dims = value.copy()

        else:
            assert False, "Not allowed type %s" % type(value)

        if len(self._dims) > 1:
            dirs = list(self._dims.keys())
            self.properties.setdefault("time", dirs[0])
            self.properties.setdefault("space", dirs[1:])

    @property
    def dofs(self):
        return getattr(self, "_dofs", {})

    @property
    def n_dofs(self):
        return len(self._dofs)

    @dofs.setter
    def dofs(self, value):

        if not value:
            self._dofs = {}

        elif isinstance(value, str):
            assert value in Lattice.theories, "Unknown dofs name"
            self._dofs = Lattice.theories[value].copy()
            self.properties = self._dofs.pop("properties", {})

        elif isinstance(value, int):
            assert value > 0, "Non-positive size for dof"
            self.dofs = [1] * value
            return

        elif isinstance(value, (list, tuple)):
            assert all(
                (isinstance(v, int) and v > 0 for v in value)
            ), "All entries of the list must be positive integers"

            self._dofs = {"dof_%d" % i: v for i, v in enumerate(value)}

        elif isinstance(value, (dict, MappingProxyType)):
            assert all(
                (isinstance(v, int) and v > 0 for v in value.values())
            ), "All entries of the dict must be positive integers"

            self._dofs = value.copy()

        else:
            assert False, "Not allowed type %s" % type(value)

    @property
    def properties(self):
        return getattr(self, "_properties", {})

    @properties.setter
    def properties(self, value):
        if not value:
            self._properties = {}

        elif isinstance(value, (dict)):
            assert all(
                (v in self for v in value.values())
            ), """
            Each property must be either a str, a list or a tuple
            of attributes of the lattice object. See lattice.dimensions.
            """

            self._properties.update(value)

        else:
            assert False, "Not allowed type %s" % type(value)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Lattice)
            and self.dims == other.dims
            and self.dofs == other.dofs
            and self.properties == other.properties
        )

    @property
    def dimensions(self):
        keys = set(["n_dims", "dims", "n_dofs", "dofs"])
        keys.update(self.dims.keys())
        keys.update(self.dofs.keys())
        keys.update(self.properties.keys())
        return sorted(keys)
    
    def __dir__(self):
        keys = set(dir(type(self)))
        keys.update(self.dimensions)
        return sorted(keys)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.dimensions
        return all((k in self.dimensions for k in key))

    def __getitem__(self, key):
        try:
            return getattr(type(self), key).__get__(self)
        except AttributeError:
            if key in type(self).__slots__:
                raise
            if key in self.dims:
                return self.dims[key]
            if key in self.dofs:
                return self.dofs[key]
            if key in self.properties:
                return self.properties[key]

            raise

    __getattr__ = __getitem__

    def __setitem__(self, key, value):
        assert not self.frozen, """
        Cannot change a lattice in use by a field. Do a copy first.
        """
        try:
            getattr(type(self), key).__set__(self, value)
        except AttributeError:
            if key in self.dims:
                dims = self.dims
                dims[key] = value
                self.dims = dims
            elif key in self.dofs:
                dofs = self.dofs
                dofs[key] = value
                self.dofs = dofs
            elif key in self.properties:
                if isinstance(value, (int)):
                    for attr in self.properties[key]:
                        self[attr] = value
                elif isinstance(value, (list, tuple)) and all(
                    (isinstance(v, int) for v in value)
                ):
                    assert len(value) == len(
                        self.properties[key]
                    ), """
                    When setting a property with a list, the length must match.
                    """
                    for attr, val in zip(self.properties[key], value):
                        self[attr] = val
                else:
                    properties = self.properties
                    properties[key] = value
                    self.properties = properties
            else:
                raise

    __setattr__ = __setitem__

    def __dask_tokenize__(self):
        return normalize_token((type(self), self.__getstate__()))

    def check(self):
        try:
            return self == self.copy()
        except AssertionError:
            return False

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Lattice(dims=self.dims, dofs=self.dofs, properties=self.properties)

    def __getstate__(self):
        return (
            self._dims.copy(),
            self._dofs.copy(),
            self._properties.copy(),
            self.frozen,
        )

    def __setstate__(self, state):
        self._dims, self._dofs, self._properties, self.frozen = state
