"""
Base tools for saving and loading data
"""

__all__ = [
    "load",
    "loads",
    "dump",
    "dumps",
]

from contextlib import contextmanager
from importlib import import_module


DOC = """
    dformat: str
        The format of the file to read. Allowed formats (case non-sensitive):
        - None: deduced from file
        - "pkl": pickle file format
        - "txt", "ASCII": txt file format
        - "HDF5", "H5": HDF5 file format
        - "lime": lime file format
    kwargs: dict
        Additional list of information for performing the reading/writing.
"""


def load(
    filein,
    dformat=None,
    **kwargs,
):
    """
    Loads data from a file and returns it as a lyncs object.

    Parameters
    ----------
    filein: str, file-object
        The filename of the data file to read. It can also be a file-like object."""

    return Format(dformat, filein, read=True).load(filein, **kwargs)


def loads(
    data,
    dformat=None,
    **kwargs,
):
    """
    Loads data from a raw string of bytes and returns it as a lyncs object.

    Parameters
    ----------"""

    return Format(dformat, data=data).loads(data, **kwargs)


def dump(obj, fileout, dformat=None, **kwargs):
    """
    Saves a lyncs object into a file.

    Parameters
    ----------
    fileout: str
        The filename of the data file to write. It can also be a file-like object."""

    return Format(dformat, fileout, obj=obj).dump(obj, fileout, **kwargs)


def dumps(obj, dformat=None, **kwargs):
    """
    Dumps a lyncs object as a string of bytes.

    Parameters
    ----------"""

    return Format(dformat, obj=obj).dumps(obj, **kwargs)


load.__doc__ += DOC
loads.__doc__ += DOC
dump.__doc__ += DOC
dumps.__doc__ += DOC


@contextmanager
def fopen(filename, *args, **kwargs):
    "Opens filename if is a string otherwise consider it as a file-like object"

    if isinstance(filename, str):  # filename
        with open(filename, *args, **kwargs) as fin:
            yield fin
    else:  # file-like object
        yield filename


class Format(str):
    """
    Holder for file formats.
    Deduces the format from the input and returns functions from the respective module.
    See Format.s for the file formats available.
    The key is the module name and the value is a tuple of the aliases (lower case).
    """

    # module: (aliases,)  !!NOTE!! use only lower cases for aliases
    s = {
        "pickle": (
            "pkl",
            "pickle",
        ),
        "lyncs.io.json": (
            "json",
            "txt",
            "ascii",
        ),
        "lyncs.io.hdf5": ("hdf5", "h5"),
        "lyncs.io.lime": ("lime",),
    }

    def __new__(cls, value, filename=None, read=False, data=None, obj=None):
        "Multiple ways to deduce the file format"

        if isinstance(value, str):
            if value in Format.s:
                return super().__new__(cls, value)
            value = value.lower()
            for key, aliases in Format.s.items():
                if value in aliases:
                    return cls(key)
            raise ValueError("Could not deduce the format from %s" % value)

        if isinstance(filename, str):
            try:  # Deducing from the extension
                return cls(filename.split(".")[-1])
            except ValueError:
                pass

        if read:
            with fopen(filename, "r") as fin:
                return cls(value, data=fin.read(1024))

        if data is not None:
            for key in Format.s:
                try:
                    if cls(key).is_compatible(data):
                        return cls(key)
                except (ValueError, ImportError, AttributeError):
                    pass

        if obj is not None and hasattr("__lyncs_file_format__"):
            return cls(obj.__lyncs_file_format__())

        raise ValueError(
            "Not enough information has been given to deduce the file format."
        )

    def __init__(self, *args, **kwargs):
        assert self in Format.s

        try:
            import_module(self)
        except ImportError as err:
            raise err

        super().__init__()

    def __getattr__(self, key):
        return getattr(import_module(self), key)

    def __eq__(self, other):
        return super().__eq__(other) or other in Format.s[self]
