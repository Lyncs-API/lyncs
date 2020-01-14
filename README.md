# Lyncs
A python API for LQCD applications

## Installing:
### Using cmake:

### Using pip wheels:


## Goals:
- Include several LQCD libraries within a single framework
- Provide crosschecks and benchmarks of different libraries' implementations
- Handle memory distribution and mapping
- Allow for multitasking parallelization and unequal distribution

## Dependencies:
(M) mandatory
(O) optional

### Python utils:
(M) numpy: Multidimensional arrays in python
(M) dask: Utility for sceduling distributed tasks
(M) cppyy: Automatic binding to C/C++ libraries
(O) dask-mpi, mpi4py: MPI for python

### LQCD libraries:
(O) Quda: LQCD on GPUs

### Extras:
(O) Jupyter lab
(O) dask-labextension

## Founding:
- PRACE-6IP, Grant agreement ID: 823767, Project name: LyNcs.

## Authors:
- Simone Bacchio, The Cyprus Institute
