"""
Implementation of IO functions for Lyncs.
"""

__all__ = [
    'load',
    'save',
]

def load(
        filename,
        format = None,
        lattice = None
        type = None,
        field = None,
):
    """
    Loads data from file and returns it as lyncs object.

    Parameters
    ----------
    filename: (str) path and filename of the data file to read.

    format: (str) format of the file to read. Allowed formats (case non-sensitive):
    - None: deduced from file
    - "TXT", "ASCII": txt file format
    - "HDF5", "H5": HDF5 file format
    - "lime": lime file format

    lattice: (Lattice) lyncs Lattice information (see Lattice for help).
             If none is deduced from the file.

    type: (str) Type of the lyncs Field to store the data in (see Field for help). 
          If none is deduced from the file.

    field: (Field) The field object where to store the data. If none a new one is created and returned.
    """
    
    pass


def save(
        field,
        filename,
        format = None,
        overwrite = False,
):
    """
    Saves data into file from a lyncs object.

    Parameters
    ----------
    field: a lyncs Field object
    
    filename: (str) path and filename of the data file to save.

    format: (str) format of the file to save (see load for help).

    overwrite: (bool) whether to overwrite data in case exist already.
    """
    
    pass
