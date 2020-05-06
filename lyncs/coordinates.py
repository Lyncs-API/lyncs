"""
Coordinates on the Lattice
"""
# pylint: disable=C0303,C0330

import random
from itertools import chain


class Coordinates(dict):
    "Coordinates class"

    def __init__(self, lattice):
        self.lattice = lattice
        super().__init__()

    def random_source(self, label=None):
        "A random coordinate in the lattice dims"
        coord = {
            key: random.choice(range(val)) for key, val in self.lattice.dims.items()
        }

        if label is not None:
            self[label] = coord

        return coord

    def random(self, label=None):
        "A random coordinate in the lattice dims and dofs"
        coord = {
            key: random.choice(range(val))
            for key, val in chain(self.lattice.dims.items(), self.lattice.dofs.items())
        }

        if label is not None:
            self[label] = coord

        return coord

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.deduce(key)

    def __setitem__(self, key, value):
        # TODO: do checks
        super().__setitem__(key, value)

    def resolve(self, *keys, **coords):
        "Combines a set of coordinates"
        # TODO
        raise NotImplementedError

    def deduce(self, key):
        """
        Deduces the coordinates from the key.
        
        E.g.
        ----
        "random source"
        "color diagonal"
        "x=0"
        """
        # TODO
        raise NotImplementedError
