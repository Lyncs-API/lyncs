from lyncs_setuptools import setup

import os

install_requires = [
    "numpy",
    "xmltodict",
    "tuneit",
]

extras_require = {
    # lyncs/extras
    "dask": ["dask", "dask[array]"],
    "clime": ["lyncs-clime"],
    "DDalphaAMG": ["lyncs-DDalphaAMG"],
    "test": ["pytest", "pytest-cov", "pytest-benchmark"],
    # Groups
    "cpu": ["lyncs[DDalphaAMG]",],
    "gpu": [],
    "io": ["lyncs[clime]",],
}

setup(
    "lyncs",
    install_requires=install_requires,
    extras_require=extras_require,
)
