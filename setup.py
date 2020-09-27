from lyncs_setuptools import setup

import os

install_requires = [
    "numpy",
    "xmltodict",
    "tuneit",
]

# Extras
lyncs = {
    "dask": ["dask", "dask[array]"],
    "clime": ["lyncs_clime"],
    "DDalphaAMG": ["lyncs_DDalphaAMG"],
    "test": ["pytest", "pytest-cov"],
}

# Groups
lyncs["io"] = lyncs["clime"]

lyncs["mpi"] = [
    "lyncs_mpi",
] + lyncs["DDalphaAMG"]

lyncs["notebook"] = [
    "jupyterlab",
    "tuneit[graph]",
    "perfplot",
]

lyncs["all"] = lyncs["notebook"]

setup(
    "lyncs",
    install_requires=install_requires,
    extras_require=lyncs,
)
