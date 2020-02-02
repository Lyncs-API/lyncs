
class Field:
    def __init__(
            self,
            lattice,
            type="scalar",
    ):
        self.lattice = lattice
        self.type = type

    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)

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
        

