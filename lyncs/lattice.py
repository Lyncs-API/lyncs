__all__ = [
    'default_lattice',
    'Lattice',
]

_last_lattice = None
def default_lattice():
    assert _last_lattice is not None, "Any lattice has been defined yet."
    return _last_lattice


class Lattice:
    """
    Lattice base class.
    A container for all the information on the lattice theory.
    """
    
    default_dims_labels = ["t", "x", "y", "z"]
    default_dofs = {
        "QCD": {
            "spin": 4,
            "color": 3,
            "properties": {
                "gauge_dofs": ["color"],
                },
            },
        }
    
    def __init__(
            self,
            dims = 4,
            dofs = "QCD",
            dtype = "complex128",
            properties = {},
    ):
        """
        Lattice initializer.
        Note: order of dimensions and degree of freedoms has any significance.

        Parameters
        ----------
        dims: dimensions (default naming: t,x,y,z if less than 5 or dim_0/1/2...)
        -> int: number of dimensions (default 4)
        -> list/tuple: size of the dimensions and len(list) = number of dimensions
        -> dict: names of the dimensions (keys) + size (value)

        dofs: specifies local degree of freedoms.
        -> str: one of the labeled theories (QCD,...)
        -> int: size of one-dimensional degree of freedom
        -> list: size per dimension of the degrees of freedom (default naming: dof_0/1/2...)
        -> dict: names of the degree of freedom (keys) + size (value)
        """

        self._properties = {}
        self.dims = dims
        self.dofs = dofs
        self.properties = properties

        global _last_lattice
        _last_lattice = self

    @property
    def dims(self):
        if "_dims" in self.__dict__:
            return self._dims.copy()
        else:
            return {}
    
    @property
    def n_dims(self):
        return len(self.dims)

    @dims.setter
    def dims(self, value):
        
        if isinstance(value, int):
            assert isinstance(value,int), "Non-integer number of dimensions"
            assert value > 0, "Non-positive number of dimensions"
            
            if value<=len(Lattice.default_dims_labels):
                self._dims = { Lattice.default_dims_labels[i]:1 for i in range(value) }
            else:
                self._dims = { "dim_%d"%i:1 for i in range(value) }
                    
        elif isinstance(value, (list,tuple)):
            assert all([isinstance(v,int) for v in value]), "Non-integer size of dimensions"
            assert all([v>0 for v in value]), "Non-positive size of dimensions"
            
            if len(value)<=len(Lattice.default_dims_labels):
                self._dims = { Lattice.default_dims_labels[i]:v for i,v in enumerate(value) }
            else:
                self._dims = { "dim_%d"%i:v for i,v in enumerate(value) }

        elif isinstance(value, (dict)):
            assert all([isinstance(v,int) for v in value.values()]), "Non-integer size of dimensions"
            assert all([v>0 for v in value.values()]), "Non-positive size of dimensions"
            
            self._dims = value.copy()

        else:
            assert False, "Not allowed type %s"%type(value)

        dirs = list(self.dims.keys())
        if len(dirs)>1: self.properties = { "time": [dirs[0]], "space": dirs[1:] }

    @property
    def dofs(self):
        if "_dofs" in self.__dict__:
            return self._dofs.copy()
        else:
            return {}
    
    @property
    def n_dofs(self):
        return len(self.dofs)

    @dofs.setter
    def dofs(self, value):
        
        if isinstance(value, str):
            assert value in Lattice.default_dofs, "Unknown dofs name"
            self._dofs = Lattice.default_dofs[value].copy()
            self.properties = self._dofs.pop("properties", {})
                    
        elif isinstance(value, int):
            assert isinstance(value,int), "Non-integer size for dof"
            assert value > 0, "Non-positive size for dof"
            
            self._dofs = { "dof_0":value }
                    
        elif isinstance(value, (list,tuple)):
            assert all([isinstance(v,int) for v in value]), "Non-integer size of dofs"
            assert all([v>0 for v in value]), "Non-positive size of dofs"
            
            self._dofs = { "dof_%d"%i:v for i,v in enumerate(value) }

        elif isinstance(value, (dict)):
            assert all([isinstance(v,int) for v in value.values()]), "Non-integer size of dofs"
            assert all([v>0 for v in value.values()]), "Non-positive size of dofs"
            
            self._dofs = value.copy()

        else:
            assert False, "Not allowed type %s"%type(value)

    @property
    def properties(self):
        if "_properties" in self.__dict__:
            return self._properties.copy()
        else:
            return {}

    @properties.setter
    def properties(self, value):
        if isinstance(value, (dict)):
            assert all([isinstance(v,(list,tuple)) for v in value.values()]), "Each property must be a list or a tuple"
            assert all([hasattr(self,key) for v in value.values() for key in v]), "Each property must be a list of attributes"
            
            self._properties.update(value)

        else:
            assert False, "Not allowed type %s"%type(value)


    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(attr for attr in self.dims)
        o.update(attr for attr in self.dofs)
        o.update(attr for attr in self.properties)
        return sorted(o)

    def __contains__(self, key):
        return key in ["n_dims", "dims", "dofs"] or key in self.dims or key in self.dofs or key in self.properties
    
    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            try: 
                return getattr(type(self), key).__get__(self)
            except AttributeError:
                if key in self.dims:
                    return self.dims[key]
                elif key in self.dofs:
                    return self.dofs[key]
                elif key in self.properties:
                    return self.properties[key]
                else:
                    raise
    __getattr__ = __getitem__

    def __setitem__(self, key, value):
        try:
            getattr(type(self), key).__set__(self,value)
        except AttributeError:
            if key in self.__dict__:
                self.__dict__[key] = value
            elif key in self.dims:
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
                else:
                    properties = self.properties
                    properties[key] = value
                    self.properties = properties
            else:
                self.__dict__[key] = value

    __setattr__ = __setitem__


    def __dask_tokenize__(self):
        from dask.base import normalize_token
            
        return normalize_token(
            (type(self), self.dims, self.dofs, self.properties)
        )

