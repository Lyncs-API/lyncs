from lyncs_setuptools import setup, CMakeExtension

setup(
    "lyncs_mpi",
    exclude=["*.config"],
    ext_modules=[CMakeExtension("lyncs_mpi.lib", ".")],
    data_files=[(".", ["config.py.in"])],
    install_requires=["mpi4py", "lyncs-cppyy", "dask-mpi", "sh"],
    keywords=["Lyncs", "MPI", "mpi4py", "dask-mpi",],
)
