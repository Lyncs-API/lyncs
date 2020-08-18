import sys

try:
    from lyncs_setuptools import setup, CMakeExtension
    from lyncs_clime import __path__ as lime_path

    # from lyncs_DDalphaAMG import __path__ as DDalphaAMG_path
except ImportError:
    print(
        """
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Install first the requirements:
    pip install -r requirements.txt
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    """
    )
    sys.exit(0)

setup(
    "lyncs_tmLQCD",
    exclude=["*.config"],
    ext_modules=[
        CMakeExtension("lyncs_tmLQCD.lib", ".", ["-DLIME_PATH=%s" % lime_path[0],])
    ],
    data_files=[(".", ["config.py.in"])],
    install_requires=["lyncs-cppyy", "lyncs-clime",],  # "lyncs-mpi","lyncs-DDalphaAMG"
    keywords=[
        "Lyncs",
        "tmLQCD",
        "Lattice QCD",
        "Wilson",
        "Twisted-mass",
        "Clover",
        "Fermions",
        "HMC",
        "Actions",
        "ETMC",
    ],
)
