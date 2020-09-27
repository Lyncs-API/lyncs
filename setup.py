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
    "clime": ["lyncs_clime"],
    "DDalphaAMG": ["lyncs_DDalphaAMG"],
    "test": ["pytest", "pytest-cov"],
    # Groups
    "cpu": [
        "lyncs[DDalphaAMG]",
    ],
    "gpu": [],
    "io": [
        "lyncs[clime]",
    ],
    "mpi": [
        "lyncs_mpi",
    ],
    "notebook": [
        "jupyterlab",
        "tuneit[graph]",
        "perfplot",
        "lyncs[mpi]",
    ],
    "all": [
        "lyncs[notebook]",
    ],
}

setup(
    "lyncs",
    install_requires=install_requires,
    extras_require=extras_require,
)
