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

    type: (str) Type of the lyncs Field to store the data in (see Field for help). 
          If none is deduced from the file.

    field: (Field) The field object where to store the data. If none a new one is created and returned.
    """

    format = format or deduce_format(filename)

    # TODO: if needed should support also other kind of files. E.g. config files, complex datasets etc.
    # i.e. anything that will have a save option.
    
    lattice = lattice or deduce_lattice(filename, format=format, field_type=field_type)
    field_type = field_type or deduce_field_type(filename, format=format, lattice=lattice)

    field = Field(lattice=lattice,field_type=field_type)
    field.load(filename, format=format, **kwargs)
    
    return field
