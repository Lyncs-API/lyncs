import codecs
import os
from itertools import product
from .data_files import add_to_data_files

__all__ = [
    "find_description",
]


def find_description(readme=None):
    base = ["README", "readme", "description"]
    ext = ["", ".txt", ".md", ".rst"]
    options = ["".join(parts) for parts in product(base, ext)]

    if readme:
        assert os.path.isfile(readme), "Given readme does not exist"
    else:
        for filename in options:
            if os.path.isfile(filename):
                readme = filename
                break

    assert readme, """
    Couldn't find a compatible filename.
    Options are %s""" % ", ".join(
        options
    )

    if readme.endswith(".md"):
        dtype = "text/markdown"
    elif readme.endswith(".rst"):
        dtype = "text/x-rst"
    else:
        dtype = "text/plain"

    with codecs.open(readme, encoding="utf-8") as _fp:
        add_to_data_files(readme, directory=".")
        dlong = _fp.read()

    dshort = ""
    for line in dlong.split("\n"):
        if line.split():
            dshort = line
            break

    if "markdown" in dtype:
        while dshort.startswith("#"):
            dshort = dshort[1:]

    return dshort.strip(), dlong, dtype
