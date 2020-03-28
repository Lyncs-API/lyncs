from lyncs_setuptools import setup, CMakeExtension
from lyncs_clime import __path__ as lime_path

setup(
    'lyncs_DDalphaAMG',
    ext_modules = [CMakeExtension('lyncs_DDalphaAMG.lib', '.', ['-DLIME_PATH=%s'%lime_path[0]])],
    install_requires = [
        "lyncs-cppyy",
        "lyncs-clime",
    ],
    keywords = [
        "Lyncs",
        "DDalphaAMG",
        "Lattice QCD",
        "Multigrid",
        "Wilson",
        "Twisted-mass",
        "Clover",
        "Fermions",
    ],
)
