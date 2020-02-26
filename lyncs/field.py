__all__ = [
    "Field",
]

from .tunable import Tunable, tunable_property
from .field_methods import FieldMethods
from functools import wraps
from .tunable import visualize, compute

class Field(Tunable, FieldMethods):
    _field_types = {
        "scalar": ["dims"],
        "vector": ["dims", "dofs"],
        "propagator": ["dims", "dofs", "dofs"],
        "gauge_links": ["dims", "n_dims", "gauge_dofs", "gauge_dofs"],
        }
    _default_field_type = "vector"
    _default_dtype = "complex128"
    
    def __init__(
            self,
            field = None,
            lattice = None,
            field_type = None,
            dtype = None,
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
        field : array_like or Field
            Values for this field. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. A view of the array's data is used
            instead of a copy if possible. If instead it is a field object,
            then a copy of the field is made with appropriate transformations
            induced by the given parameters.
            E.g. 
            Field(field=field, coords={'x':0}) selects values at x=0
            Field(field=field, shape_order=[...]) changes the shape_order if needed.
            etc...
        lattice: Lattice object.
            The lattice on which the field is defined.
        field_type: str or list(str).
          --> If str, then must be one of the labeled field types. See Field._field_types
          --> If list, then a list of variables of lattice.        
        dtype: str or numpy dtype compatible
           Data type of the field
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
        
        self.lattice = lattice or (field.lattice if isinstance(field, Field) else default_lattice())
        
        self.field_type = field_type or (field.field_type if isinstance(field, Field) else Field._default_field_type)

        self.dtype = dtype or (field.dtype if isinstance(field, Field) else Field._default_dtype)

        if isinstance(field, Field): self.labels = field.labels
        self.labels = labels
        
        if isinstance(field, Field): self.coords = field.coords
        self.coords = coords
        
        if isinstance(field, Field):
            self._tunable_options = field._tunable_options
            self._tuned_options = field._tuned_options
        else:
            from .tunable import Permutation, ChunksOf
            
            tunable_options["shape_order"] = Permutation([key for key,val in self.shape])
            tunable_options["chunks"] = ChunksOf(self.dims)
            
            Tunable.__init__(self, tunable_options=tunable_options, tuned_options=tuned_options)

        # Loading dynamically methods and attributed from the field types in fields
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

        # Considering the remaining kwargs as tunable options
        for key, val in kwargs:
            self.add_option(key,val)
            
        self.field = field
        

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
        from dask.array import Array
        if isinstance(self.field, Array):
            import warnings
            if self.field.dtype != self._dtype:
                warnings.warn("Mistmatch between set dtype and field dtype")
            return self.field.dtype
        elif hasattr(self, "_dtype"):
            return self._dtype
        else:
            return None
    
    @dtype.setter
    def dtype(self, value):
        from numpy import dtype
        value = dtype(value)
        if value != self.dtype:
            self._dtype = value
            if self.field is not None:
                self.field = self.field.astype(value)


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
        """
        Returns all the possible dimensions valid for the field.
        """
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
            
        dims = list(dims)
        dims.sort()
        return dims


    @property
    def axes(self):
        """
        Returns the list of fundamental dimensions. The order is not significant.
        """
        return self._expand(self.field_type)
    
    
    @property
    def shape(self):
        """
        Returns the list of dimensions with size. The order is not significant.
        """
        def get_size(key):
            if key in self._coords:
                return len(self._coords[key])
            else:
                return self.lattice[key]
                
        return [(key, get_size(key)) for key in self.axes]
    
    
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
        from .tunable import Delayed, RaiseNotTuned
        import numpy as np
        
        coords = coords.copy()
        for key, val in self.coords.items():
            assert key not in coords or coords[key]==val, "Cannot change value of fixed coordinate %s" % key
            coords[key] = val
            
        _coords = {}
        for key, val in coords.items():
            assert key in self.dimensions, "Unknown dimesion %s" % key
            dims = self._expand(key)
            for dim in dims:
                assert dim not in _coords or _coords[dim]==val, "Setting multiple time the same dimension"
                
                try:
                    assert len(list(val)) > 0, "Empty list not allowed"
                except TypeError:
                    val = [val]

                _coords[dim]=val
                
        self._coords = _coords
        self._given_coords = coords


    @property
    def field(self):
        try:
            from dask.array import Array
            from .tunable import LyncsMethodsMixin
            if not isinstance(self._field, Array) and isinstance(self._field, LyncsMethodsMixin) and not self.__dict__.get("_field_lock", False):
                try:
                    self._field_lock = True
                    self._field = self._field.compute(tune=False)
                except:
                    pass
                finally:
                    self._field_lock = False
                    del self._field_lock
            return self._field
        except AttributeError:
            return None

    @field.setter
    def field(self, value):
        from .tunable import Delayed, tunable_property
        from dask.array import Array
        
        if value is None:
            self.zeros()
            
        elif isinstance(value, Field):
            self._field = value.field
            
            if self.coords != value.coords:
                field_coords = value._coords if hasattr(value, "_coords") else {}
                coords = {key:val for key,val in self._coords.items() if key not in field_coords}

                @tunable_property
                def mask(self):
                    mask = [slice(None) for i in self.shape]
                    for key,val in coords.items():
                        mask[self.shape_order.index(key)] = val
                    return tuple(mask)
                
                self._field = self._field[mask(value)]
                
        elif isinstance(value, Delayed):
            self._field = value

        elif isinstance(value, Array):
            assert type(self.field_shape) is tuple, "Field order not defined yet"
            assert self.field_shape == value.shape, """
            Shape mismatch:
            field_shape = %s
            new_field_shape = %s
            """ % (self.field_shape, value.shape)
            
            if type(self.field_chunks) is not tuple or self.field_chunks != value.chunksize:
                self.chunks = {key: val for key, val in zip(self.shape_order, value.chunksize)}
                
            self._field = value

        else:
            # TODO specialize
            assert False, "Not implemented yet"

            
    @tunable_property
    def field_shape(self):
        from dask.array import Array
        import warnings
        shape = {key:val for key,val in self.shape}
        field_shape = tuple(shape[key] for key in self.shape_order)
        if isinstance(self.field, Array):
            if self.field.shape != field_shape:
                warnings.warn("Mistmatch between computed and real field shape")
            return self.field.shape
        else:
            return field_shape    


    @tunable_property
    def field_chunks(self):
        from dask.array import Array
        import warnings
        shape = {key:val for key,val in self.shape}
        field_chunks = tuple(self.chunks[key] if key in self.chunks else shape[key] for key in self.shape_order)
        if isinstance(self.field, Array):
            if self.field.chunksize != field_chunks:
                warnings.warn("Mistmatch between computed and real field chunks")
            return self.field.chunksize
        else:
            return field_chunks
        

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
        return Field(field=self, coords=coords)
    
    
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
        

    @wraps(compute)
    def compute(self, **kwargs):
        self.tune(**kwargs.pop("tune_kwargs",{}))
        return self.field.compute(**kwargs)


    @wraps(visualize)
    def visualize(self, **kwargs):
        return self.field.visualize(**kwargs)

        
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
        
        def read_field(shape, chunks):
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
        
        self.field = delayed(read_field)(self.field_shape, self.field_chunks)
        

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
        
        def zero_field(*args, **kwargs):
            return zeros(*args, **kwargs)
        
        self.field = delayed(zero_field)(
            shape = self.field_shape,
            chunks = self.field_chunks,
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
            assert all([key not in coords or coords[key]==val for key, val in coord.items()])
            coords.update(coord)
            
        return self.get(**coords)
