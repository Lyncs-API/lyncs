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
    "test": ["pytest", "pytest-cov"],
    # Groups
    "cpu": [
        "lyncs[DDalphaAMG]",
    ],
    "gpu": [],
    "io": [
        "lyncs[clime]",
    ],
    "notebook": [
        "jupyterlab",
        "tuneit[graph]",
        "perfplot",
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
