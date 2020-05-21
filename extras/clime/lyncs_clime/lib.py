__all__ = [
    "lib",
    "PATHS",
]

import os
from . import __path__

PATHS = list(__path__)

from lyncs_cppyy import Lib

lib = Lib(
    path=PATHS,
    header="lime.h",
    library="liblime.so",
    c_include=True,
    check="LimeRecordHeader",
)
