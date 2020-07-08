try:
    from lyncs_setuptools import setup
except ImportError:
    from .extras.setuptools.lyncs_setuptools import setup

from glob import glob

install_requires = [
    "numpy",
    "xmltodict",
    "tuneit",
]

extras_require = {
    # lyncs/extras
    "dask": ["dask"],
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
    data_files=[
        (
            "extras/setuptools",
            glob("extras/setuptools/lyncs_setuptools/**/*.py", recursive=True),
        )
    ],
)
