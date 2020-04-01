"""
Additional low-level functions to the one provided by cppyy
"""

from cppyy.ll import __all__
#from cppyy.ll import *
from cppyy.ll import cast
from cppyy import cppdef, gbl

__all__ = __all__ + [
    "assign",
]

def assign(ptr, val):
    try:
        return gbl._assign(ptr, val)
    except AttributeError:
        assert cppdef(
            """
            template<typename T>
            void _assign( T* ptr, T&& val ) {
              *ptr = val;
            }
            """
        ), "Couldn't define _assign"
        return assign(ptr, val)
