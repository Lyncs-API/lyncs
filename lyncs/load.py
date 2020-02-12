"""
Implementation of load function for Lyncs.
"""

__all__ = [
    'load',
]


def load(
        filename,
        format = None,
        lattice = None,
        field_type = None,
        **kwargs,
):
    """
    Loads data from file and returns it as a lyncs object.
    The resulting object can be either a field, or ...

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

    field_type: (str) Type of the lyncs Field to store the data in (see Field for help). 
          If none is deduced from the file.

    """

    from .io import deduce_format, deduce_field, formats
    
    format = deduce_format(filename, format=format)

    # TODO: if needed should support also other kind of files. E.g. config files, complex datasets etc.
    # i.e. anything that will have a save option should be able to be loaded easily.

    field = deduce_field(filename, format=format, lattice=lattice, field_type=field_type)    
    field.load(filename, format=format, **kwargs)
    return field
