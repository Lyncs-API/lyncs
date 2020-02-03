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
    
    import os
    assert os.access(filename, os.R_OK), "File does not exists or not readable"

    if format: check_format(filename, format)
    else: format = deduce_format(filename)

    if lattice or field:
        if field:
            assert not lattice or lattice is field.lattice, "Both lattice and field given, but not compatible"
            lattice = field.lattice
        check_lattice(filename, format, lattice)
    else:
        lattice = deduce_lattice(filename, format)
        
    if type or field:
        if field:
            assert not type or type == field.type, "Both type and field given, but not compatible"
            type = field.type
        check_type(filename, format, lattice, type)
    else:
        type = deduce_type(filename, format, lattice)

    from lyncs import Field
    if not field: field = Field(lattice,type=type)
    
    _load(field, fielname, format)
    return field


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
