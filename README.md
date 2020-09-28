# A python API for Lattice QCD applications

[![python](https://img.shields.io/pypi/pyversions/lyncs.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs/)
[![pypi](https://img.shields.io/pypi/v/lyncs.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs)
[![pylint](https://img.shields.io/badge/pylint%20score-7.4%2F10-yellow?logo=python&logoColor=white)](http://pylint.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)

![alt text](https://github.com/sbacchio/lyncs/blob/master/docs/source/_static/logo.png "Lyncs")

Lyncs is a Python API for Lattice QCD applications currently under development with a first
released version expected by the end of Q2 of 2020. Lyncs aims to bring several popular
libraries for Lattice QCD under a common framework. Lyncs will interface with libraries for
GPUs and CPUs in a way that can accommodate additional computing architectures as these
arise, achieving the best performance for the calculations while maintaining the same high-
level workflow. Lyncs is one of 10 applications supported by PRACE-6IP, WP8 "Forward
Looking Software Solutions".

Lyncs distributes calculations using Dask, with bindings to the libraries performed
automatically via Cppyy. Multiple distributed tasks can be executed in parallel and different
computing units can be used at the same time to fully exploit the machine allocation. The data
redistribution is efficiently managed by the API. We expect this model of distributing tasks to
be well suited for modular architectures, allowing to flexibly distribute
work between the different modules.
While Lyncs is designed to quite generally allow linking to multiple libraries, we will
focus on a set of targeted packages that include tmLQCD, DDalphaAMG, PLEGMA and QUDA.


## Installation:

The package can be installed via `pip`:

```
pip install [--user] lyncs
```

### Sub-modules and plugins

Sub-modules and plugins can also be installed via `pip` with:

```
pip install [--user] lyncs[NAME]
```

where NAME is the name of the sub-module. Hereafter the list of the available sub-modules.

#### Groups

- `all`: installs all the plugins enabling all Lyncs' functionalities (e.g. hmc, visualization, etc..).
  Note it does not install libraries with strong dependencies like MPI, GPUs, etc.
  Simple CPUs libraries may be installed.

- `mpi`: installs all MPI libraries.

- `cuda`: installs all NVIDIA GPUs libraries.

- `io`: installs all IO libraries for full support of IO formats (clime, HDF5, etc..).

#### LQCD libraires

- `DDalphaAMG`: multigrid solver library for Wilson and Twisted mass fermions.

- `QUDA`: NVIDIA GPUs library for LQCD.

- `clime`: IO library for c-lime format.

- `tmLQCD`: legacy code of the Extended Twisted Mass collaboration.

## Goals:

- Include several Lattice QCD libraries under a single framework
- Provide crosschecks and benchmarks of different libraries' implementations
- Handle memory distribution and mapping
- Allow for multitasking parallelization and unequal distribution


## Dependencies:

### Python utils:

- numpy: Multidimensional arrays in python
- dask: Utility for sceduling distributed tasks
- cppyy: Automatic binding to C/C++ libraries
- (optional) dask-mpi, mpi4py: MPI for python
- (under consideration) numba: JIT compilation of python code
- others: xmltodict, 

### LQCD libraries:

- QUDA: Lattice QCD operators and solvers on GPUs
- DDalphaAMG: Multigrid solver on CPUs
- tmLQCD: HMC routines on CPUs
- PLEGMA: contraction kernels on GPUs

### Extras requirements:

- Jupyter notebook/lab: for visualizing and running the avaialble notebooks
- dask-labextension: utils for profiling the task execution in Jupyter lab


## Fundings:

- PRACE-6IP, Grant agreement ID: 823767, Project name: LyNcs.


## Authors:

- Simone Bacchio, The Cyprus Institute
