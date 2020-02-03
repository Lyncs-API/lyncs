#from xarray import DataArray

class Field:#(DataArray)
    _field_types = {
        "scalar": ["dims"],
        "vector": ["dims", "dofs"],
        "propagator": ["vector", "dofs"],
        "gauge": ["dims", "gauge_dofs", "gauge_dofs"],
        "gauge_links": ["gauge", "n_dims"],
        }
    
    def __init__(
            self,
            data = None,
            lattice = None,
            field_type = None,
    ):
        """
        A field defined on the lattice.
        
        Parameters
        ----------
        data : array_like
            Values for this field. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. A view of the array's data is used
            instead of a copy if possible. If intest is a field object, then 
            a copy of the field is made.
        lattice: Lattice object.
            The lattice on which the field is defined.
        field_type: str or list(str).
        --> If str, then must be one of the labeled field types.
        --> If list, then a list of variables of lattice.
        """
        self.data = data
        self.lattice = lattice
        self.field_type = field_type

        
    @property
    def dims(self):
        def get_size(self, key):
            if isinstance(key, (list, dict)):
                ret = []
                for k in key:
                    ret += get_size(self,k)
                return ret
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
                    return get_size(self,value)
            else:
                assert False, "Got key that is neither list or str, %s" % key
                
        return get_size(self, self.field_type)
        
    def load(
            self,
            filename,
            format = None,
    ):
        """
        Loads data from file.
        
        Parameters
        ----------
        filename: (str) path and filename of the data file to read.
        
        format: (str) format of the file to read (see load for help).
        """
        from .io import load
        load(fielname, format=format, field=self)
        

    def save(
            filename,
            format = None,
            overwrite = False,
    ):
        """
        Saves data into file.
        
        Parameters
        ----------
        filename: (str) path and filename of the data file to save.
        
        format: (str) format of the file to save (see load for help).
        
        overwrite: (bool) whether to overwrite data in case exist already.
        """
        from .io import save
        save(self, fielname, format=format, field=overwrite)
        

    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)


    
