"""
Several high-level function for creating a field.
"""

__all__ = [
    'load',
    'zeros_like',
]


def load(
        filename,
        format = None,
        **kwargs,
):
    """
    Loads data from file and returns it as a lyncs object.
    The file must contain 

    Parameters
    ----------
    filename: (str) path and filename of the data file to read.

    format: (str) format of the file to read. Allowed formats (case non-sensitive):
    - None: deduced from file
    - "TXT", "ASCII": txt file format
    - "HDF5", "H5": HDF5 file format
    - "lime": lime file format

    **kwargs: Additional list of information to perform the reading. E.g. the following options.

    lattice: (Lattice) lyncs Lattice information (see Lattice for help).
             If none is deduced from the file.

    field_type: (str) Type of the lyncs Field to store the data in (see Field for help). 
          If none is deduced from the file.

    """

    from os import access, R_OK
    assert access(filename, R_OK), "File %s does not exist or is not readable" % filename
    
    from .io import deduce_format, deduce_type
    
    format = deduce_format(filename, format=format)

    obj = deduce_type(filename, format=format, **kwargs)
    obj.load(filename, format=format, **kwargs)
    return obj


def zeros_like(
        field,
        **kwargs,
):
    """
    Creates a zero field with the same properties of the one given.

    Parameters
    ----------
    field: (Field)
       The field to take information from.
    kwargs: (dict)
       List of parameter to pass to the field. They may replace arguments of field.
    """

    from .field import Field
    kwargs["zeros_init"] = True
    
    return Field(field, **kwargs)
