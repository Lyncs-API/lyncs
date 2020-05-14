"""
Coordinates on the Lattice
"""
# pylint: disable=C0303,C0330

import random
from itertools import chain
from collections.abc import Iterable
from .field.base import BaseField, index_to_axis
from .utils import compact_indeces, expand_indeces


class Coordinates(dict):
    "Coordinates class"

    @classmethod
    def add_coords(cls, coords=None, **kwargs):
        "Add kwargs to coords where coords is a dict"
        coords = coords or {}
        assert isinstance(coords, dict), "Coords is supposed to be a dict"
        for key, val in kwargs.items():
            if not isinstance(val, tuple):
                val = (val,)
            if key not in coords:
                coords[key] = val
            else:
                if not isinstance(coords[key], tuple):
                    coords[key] = (coords[key],)
                coords[key] += val
        return coords

    @classmethod
    def format_coords(cls, *coords, **kwargs):
        "Returns a list of args, kwargs from the given coords"
        args = []
        kwargs = cls.add_coords(**kwargs)
        for coord in coords:
            if coord is None:
                continue
            if isinstance(coord, str):
                args.append(coord)
            elif isinstance(coord, dict):
                kwargs = cls.add_coords(kwargs, **coord)
            else:
                if not isinstance(coord, Iterable):
                    raise TypeError(
                        "coords can be str, dict or iterables. %s not accepted." % coord
                    )
                _args, _kwargs = cls.format_coords(*coord)
                kwargs = cls.add_coords(kwargs, **_kwargs)
                args.extend(_args)
        return tuple(args), kwargs

    @classmethod
    def format_values(cls, *values, interval=None, compact=True):
        "Returns a list of values for the coordinate"
        vals = set()
        for value in values:
            if isinstance(value, (int, str, range)):
                tmp = tuple(expand_indeces(value))
                if interval is not None and not set(tmp).issubset(interval):
                    raise ValueError("Value %s out of interval" % value)
                vals.update(tmp)
            elif isinstance(value, slice):
                if interval is None:
                    raise ValueError("Slice requires an interval")
                vals.update(interval[value])
            else:
                vals.update(cls.format_values(*value, interval=interval, compact=False))
        assert not vals.difference(interval), "Trivial assertion"
        if compact:
            if vals == set(interval):
                return slice(None)
            vals = compact_indeces(*vals)
        return tuple(vals)

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
        super().__setitem__(key, self.resolve(value))

    def resolve(self, *keys, field=None, **coords):
        "Combines a set of coordinates"
        if field is not None and not isinstance(field, BaseField):
            raise ValueError("field must be a Field type")

        keys, coords = self.format_coords(*keys, **coords)
        if not keys and not coords:
            if field is not None:
                return field.coords
            return ()
        for key in keys:
            coords = self.add_coords(coords, **self.deduce(key, field=field))

        resolved = {}
        for key, val in coords.items():
            if field is not None:
                keys = field.get_indeces(key)
                if not keys:
                    raise ValueError("Index '%s' not in field" % key)
            else:
                keys = self.lattice.expand(key)
            resolved = self.add_coords(resolved, **{idx: val for idx in keys})

        for key, val in resolved.items():
            interval = self.lattice.get_axis_range(index_to_axis(key))
            resolved[key] = self.format_values(*val, interval=interval)

        if field is not None:
            for key, val in field.coords:
                if key in resolved:
                    if not set(expand_indeces(val)) >= set(
                        expand_indeces(resolved[key])
                    ):
                        raise ValueError(
                            "%s = %s not in field coordinates that has %s = %s"
                            % (key, resolved[key], key, val)
                        )
                else:
                    resolved[key] = val

        return tuple(resolved.items())

    def deduce(self, key, field=None):
        """
        Deduces the coordinates from the key.
        
        E.g.
        ----
        "random source"
        "color diagonal"
        "x=0"
        """
        if key in self:
            return dict(self[key])

        # Looking up in lattice labels
        for name, labels in self.lattice.labels.items():
            if field is not None and name not in field.labels:
                continue
            if key in labels:
                return {name: key}

        # TODO
        raise NotImplementedError
