from lyncs_setuptools import setup, CMakeExtension

setup(
    'lyncs_clime',
    ext_modules = [CMakeExtension('lyncs_clime.clime', '.')],
    keywords = [
        "Lyncs",
        "c-lime",
        "Lattice QCD",
        ],
    )
