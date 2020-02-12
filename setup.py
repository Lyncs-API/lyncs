from setuptools import find_packages, setup

try:
    import lyncs_config
except ModuleNotFoundError:
    # TODO: run cmake with default options and then import lyncs_config
    raise

requirements = [
    "cppyy>=1.6.2",
    "dask",
    "numpy",
    "xmltodict",
    ]

if lyncs_config.mpi_enabled:
    requirements.append("mpi4py")
    requirements.append("dask_mpi")


setup(name='lyncs',
      version=lyncs_config.version,
      packages=find_packages(),
      description='A python API for LQCD applications',
      long_description=open("./README.md", 'r').read(),
      long_description_content_type="text/markdown",
      keywords="LQCD, API, ...",
      classifiers=["Intended Audience :: Students",
                   "Intended Audience :: Researchers",
                   "License :: OSI Approved :: "
                   "The 3-Clause BSD License (BSD-3-Clause)",
                   "Natural Language :: English",
                   "Programming Language :: C",
                   "Programming Language :: C++",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3.6"],
      license='BSD-3-Clause',
      install_requires = requirements
    )
