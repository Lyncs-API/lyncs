
![alt text](https://github.com/sbacchio/lyncs/blob/master/docs/source/_static/logo.png "Lyncs")

# Lyncs

## A python API for Lattice QCD applications

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

For a default installation use either `conda install lyncs` or `pip install lyncs`.

### Advance setup:

For a development setup or for advanced options, build the package using cmake.
It can be done either by editing and running build.sh or by running the following commands
from the lyncs direction.

```
mkdir -p build && \
( cd build && ccmake ../ && make -j && make install ) && \
python3 setup.py develop --user
```


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
