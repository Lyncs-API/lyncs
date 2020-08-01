# Tools for Hybrid Monte Carlo simulations

This module offers an implementation of the Hybrid Monte Carlo algorithm.
The implementation is based on [Simpy](https://simpy.readthedocs.io/),
a process-based discrete-event simulation framework.

The user can easily add events in any step of the simulation.
These can be measurements steps, storage of the configuration or also
variations of the algorithm itself.