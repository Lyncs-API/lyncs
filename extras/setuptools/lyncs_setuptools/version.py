__all__ = [
    "find_version",
]

import codecs
import os
import re
from itertools import product
from setuptools import find_packages
from .data_files import add_to_data_files


def find_version(filename=None):
    def get_version(filename):
        assert os.path.isfile(filename), "Given version does not exist"
        with codecs.open(filename, "r") as _fp:
            if filename.endswith(".py"):
                version_match = re.search(
                    r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", _fp.read(), re.M
                )
                assert version_match, "__version__ = not found in file"
                return version_match.group(1)
            return _fp.read()

    version = None
    if filename:
        version = get_version(filename)
    else:
        base = ["VERSION", "version"]
        ext = ["", ".txt"]
        options = ["".join(parts) for parts in product(base, ext)]

        pkgs = find_packages()
        files = ["__init__.py", "version.py", "_version.py"]
        options += [os.sep.join(parts) for parts in product(pkgs, files)]

        for _filename in options:
            try:
                version = get_version(_filename)
                filename = _filename
                break
            except AssertionError:
                pass

    assert version, """
    Couldn't find a compatible filename.
    Options are %s""" % ", ".join(
        options
    )

    add_to_data_files(filename)
    return version
