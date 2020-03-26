from lyncs_setuptools import setup, CMakeExtension

setup(
    'lyncs_clime',
    ext_modules = [CMakeExtension('lyncs_clime.clime', '.')],
    install_requires = [
        "lyncs-cppyy",
        ],
    keywords = [
        "Lyncs",
        "c-lime",
        "Lattice QCD",
        ],
    )
