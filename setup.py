from lyncs_setuptools import setup

install_requires = [
    "dask",
    "numpy",
    "xmltodict",
    "tunable",
]

extras_require = {
    # lyncs/extras
    "clime": ["lyncs-clime"],
    "DDalphaAMG": ["lyncs-DDalphaAMG"],
    # Groups
    "cpu": ["lyncs[DDalphaAMG]",],
    "gpu": [],
    "io": ["lyncs[clime]",],
}

setup(
    "lyncs",
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=["Python", "API", "Lattice", "Field", "QCD",],
)
