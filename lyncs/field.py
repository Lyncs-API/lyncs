
from .tunable import Tunable, tunable_property

class Field(Tunable):
    _field_types = {
        "scalar": ["dims"],
        "vector": ["dims", "dofs"],
        "propagator": ["vector", "dofs"],
        "gauge_links": ["dims", "n_dims", "gauge_dofs", "gauge_dofs"],
        }
    _default_field_type = "vector"
    
    def __init__(
            self,
            array = None,
            lattice = None,
            field_type = None,
            coords = {},
            labels = {},
            tunable_options = {},
            tuned_options = {},
            **kwargs
    ):
        """
        A field defined on the lattice.
        
        Parameters
        ----------
        array : array_like
            Values for this field. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. A view of the array's data is used
            instead of a copy if possible. If instead it is a field object,
            then a copy of the field is made with appropriate transformations
            induced by the given parameters.
            E.g. 
            Field(array=field, coords={'x':0}) selects values at x=0
            Field(array=field, shape_order=[...]) changes the shape_order if needed.
            etc...
        lattice: Lattice object.
            The lattice on which the field is defined.
        field_type: str or list(str).
          --> If str, then must be one of the labeled field types. See Field._field_types
          --> If list, then a list of variables of lattice.
        coords: dict, str or list of str
           Coordinates of the field, i.e. range of values for any of the dimensions.
           It can be integer, range, slice or None. If None then is a reduced global field.
           It can be also one or a list of labels.
        labels: dict
           Dictionary of labeled coordinates of the field, e.g. "source": dict(x=0, y=0, z=0, t=0)
        tunable_options: dict
           List of tunable parameters with default values.
           Tunable options are attributes of the field and can be used to condition the computation. 
        tuned_options: dict
           Same as tunable options but with a fixed value.
        kwargs: dict
           Extra paramters that will be passed to the child classes of the field during initialization.
        """
        from .lattice import default_lattice
        
        self.lattice = lattice or (array.lattice if isinstance(array, Field) else default_lattice())
        
        self.field_type = field_type or (array.field_type if isinstance(array, Field) else Field._default_field_type)
        
        if isinstance(array, Field): self.labels = array.labels
        self.labels = labels
        
        if isinstance(array, Field): self.coords = array.coords
        self.coords = coords
        
        from .tunable import Permutation, ChunksOf

        tunable_options["shape_order"] = Permutation([v[0] for v in self.shape])
        tunable_options["chunks"] = ChunksOf(self.dims)
        
        Tunable.__init__(self, tunable_options=tunable_options, tuned_options=tuned_options)

        from importlib import import_module
        for name in self.dimensions:
            try:
                module = import_module(".fields.%s"%name, package="lyncs")
                for attr in module.__all__:
                    val = getattr(module, attr)
                    if attr == "__init__":
                        val(self, **kwargs)
                    else:
                        setattr(self, attr, val)
            except ModuleNotFoundError:
                pass

            
        self.array = array
        

    @property
    def lattice(self):
        try:
            # TODO: should return a copy of lattice
            return self._lattice
        except AttributeError:
            return None
        
    @lattice.setter
    def	lattice(self, value):
        assert self.lattice is None, "Not allowed to change lattice, if needed ask to implement it"
        from .lattice import Lattice
        assert isinstance(value, Lattice)
        
        self._lattice = value

        
    @property
    def dtype(self):
        return self.lattice.dtype


    @property
    def dims(self):
        return {key:size for key,size in self.shape if key in self.lattice.dims}

    
    @property
    def dofs(self):
        return {key:size for key,size in self.shape if key in self.lattice.dofs}


    def _expand(self, prop):
        "Expands a lattice/field property into the fundamental dimensions"
        def __expand(prop):
            if prop in self.lattice and isinstance(self.lattice[prop], int):
                return prop
            else:
                if prop in self.lattice:
                    return " ".join([__expand(key) for key in self.lattice[prop]])
                else:
                    return " ".join([__expand(key) for key in self._field_types[prop]])
                
        return __expand(prop).split()
    
    
    @property
    def dimensions(self):
        dims = set()
        
        def add(key):
            if isinstance(key, (list, tuple, dict)):
                for k in key: add(k)
            elif isinstance(key, str):
                dims.add(key)
                if key in self.lattice: add(self.lattice[key])
                elif key in self._field_types: add(self._field_types[key])
                    
        add(self.field_type)
        
        for prop in self.lattice.properties:
            names = self._expand(prop)
            if set(names).issubset(dims): dims.add(prop)
            
        return dims
    
    
    @property
    def field_type(self):
        try:
            return self._field_type
        except AttributeError:
            return None

    @field_type.setter
    def	field_type(self, value):
        assert self.field_type is None, "Not allowed to change field_type, if needed ask to implement it"
        def is_known(self, key):
            if isinstance(key, (list, tuple)):
                return all(is_known(self,k) for k in key)
            elif isinstance(key, str):
                return key in self.lattice.__dir__() or key in self._field_types
            else:
                assert False, "Got key that is neither list or str, %s" % key
                
        assert is_known(self, value), "Got unknown field type"
        self._field_type = value


    @property
    def coords(self):
        try:
            return self._given_coords
        except AttributeError:
            return {}

    @coords.setter
    def coords(self, coords):
        coords = coords.copy()
        for key, val in self.coords:
            assert key not in coords or coords[key]==val, "Cannot change value of fixed coordinate %s" % key
            coords[key] = val
            
        _coords = {}
        for key, val in coords.items():
            assert key in self.dimensions, "Unknown dimesion %s" % key
            dims = self._expand(key)
            for dim in dims:
                assert dim not in _coords or _coords[dim]==val, "Setting multiple time the same dimension"
                _coords[dim]=val
                
        self._coords = _coords
        self._given_coords = coords


    @property
    def array(self):
        try:
            from dask.array import Array
            from .tunable import LyncsMethodsMixin
            if not isinstance(self._array, Array) and isinstance(self._array, LyncsMethodsMixin):
                try:
                    self._array = self._array.compute(tune=False)
                except:
                    pass
            return self._array
        except AttributeError:
            return None

    @array.setter
    def array(self, value):
        if value is None:
            self.zeros()
        else:
            self._array = value


    @property
    def shape(self):
        """
        Returns the list of dimensions with size. The order is not significant.
        """
        def get_shape(self, key):
            if isinstance(key, (list, tuple)):
                ret = []
                for k in key:
                    ret += get_shape(self,k)
                return ret
            elif isinstance(key, dict):
                assert all(isinstance(value, int) for value in key.values())
                return list(key.items())
            elif isinstance(key, str):
                if key in self.lattice.__dir__():
                    value = getattr(self.lattice, key)
                elif key in self._field_types:
                    value = self._field_types[key]
                else:
                    assert False, "Unknown attribute %s"%key
                if isinstance(value, int):
                    return [(key, value)]
                else:
                    return get_shape(self,value)
            else:
                assert False, "Got key that is neither list or str, %s" % key
                
        return get_shape(self, self.field_type)


    @tunable_property
    def array_shape(self):
        return tuple(self.lattice[key] for key in self.shape_order)


    @tunable_property
    def array_chunks(self):
        return tuple(self.chunks[key] if key in self.chunks else self.lattice[key] for key in self.shape_order)
        

    @property
    def size(self):
        """
        Returns the number of elements in the field.
        """
        prod = 1
        for t in self.shape: prod*=t[1]
        return prod


    @property
    def byte_size(self):
        """
        Returns the size of the field in bytes
        """
        return self.size*self.dtype.itemsize
    
    
    def get(self, **coords):
        return Field(array=self, coords=coords)
    
    
    @property
    def labels(self):
        try:
            return self._labels
        except AttributeError:
            return {}

    @labels.setter
    def labels(self, labels):
        for label, coords in labels.items():
            self.label(label, **coords)

    def label(self, name, **coords):
        """
        Labels a given set of coordinates.
        Labels are accessible via __getitem__.

        E.g.
        
        field.label("source", x=0, y=0, z=0, t=0)
        dofs = field["source"]
        
        Parameters
        ----------
        name: (str)
           Name of the label.
        coords: (dict)
           The coordinate of a subset of dimensions.
        """
        if not hasattr(self, "_labels"): self._labels={}
        assert all([key in self.dimensions for key in coords]), "Some of the dimensions are not known"
        self._labels[name] = coords
        

    def compute(self, **kwargs):
        self.tune(**kwargs.pop("tune_kwargs",{}))
        return self.array.compute(**kwargs)


    def visualize(self, **kwargs):
        return self.array.visualize(**kwargs)

        
    def load(
            self,
            filename,
            format = None,
            **info,
    ):
        """
        Loads data from file.
        
        Parameters
        ----------
        filename: (str) path and filename of the data file to read.
        format: (str) format of the file to read (see load for help).
        info: information needed to perform the reading.
        """
        from .io import file_manager
        from .tunable import delayed
        from dask.array import from_delayed
        
        io = file_manager(filename, format=format, field=self, **info)
        
        def read_array(shape, chunks):
            from dask.highlevelgraph import HighLevelGraph
            from dask.array.core import normalize_chunks, Array
            from itertools import product
            
            chunks = normalize_chunks(chunks, shape=shape)
            chunks_id = list(product(*[range(len(bd)) for bd in chunks]))

            reads = [io.read(chunk_id) for chunk_id in chunks_id]
            
            keys = [(filename, *chunk_id) for chunk_id in chunks_id]
            vals = [read.key for read in reads]
            dsk = dict(zip(keys, vals))

            graph = HighLevelGraph.from_collections(filename, dsk, dependencies=reads)

            return Array(graph, filename, chunks, dtype=self.dtype)
        
        self.array = delayed(read_array)(self.array_shape, self.array_chunks)
        

    def save(
            filename,
            format = None,
            **info,
    ):
        """
        Saves data into file.
        
        Parameters
        ----------
        filename: (str) path and filename of the data file to save.
        format: (str) format of the file to save (see load for help).
        info: information needed to perform the writing.
        """
        from .io import save_field
        save(self, filename, format=format, field=overwrite)


    def zeros(self):
        """
        Initialize the field with zeros.
        """
        from .tunable import delayed
        from dask.array import zeros
        
        def zero_array(*args, **kwargs):
            return zeros(*args, **kwargs)
        
        self.array = delayed(zero_array)(
            shape = self.array_shape,
            chunks = self.array_chunks,
            dtype = self.dtype,
        )


    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)
    

    def __dask_tokenize__(self):
        import uuid
        from dask.base import normalize_token
        if not hasattr(self, "_unique_id"): self._unique_id = str(uuid.uuid4())
            
        return normalize_token(
            (type(self), self.lattice, self.field_type, self._unique_id)
        )

    
    def __getitem__(self, *labels):
        assert all([label in self.labels for label in labels]), "Unknown label"
        coords = {}
        for label in labels:
            coord = self._labels[label]
            assert all([key not in coords or coords[key]==val for key, val in coord])
            coords.update(coord)
            
        return self.get(**coords)
