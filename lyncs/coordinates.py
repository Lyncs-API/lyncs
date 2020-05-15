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
    def add_coords(cls, coords, **kwargs):
        "Add kwargs to coords where coords is a dict"
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

    @classmethod
    def format_coords(cls, *keys, field=None, **coords):
        "Returns a list of args, kwargs from the given keys and coords"
        args = set()
        kwargs = {}
        cls.add_coords(kwargs, **coords)
        for key in keys:
            if key is None:
                continue
            if isinstance(key, str):
                args.add(key)
            elif isinstance(key, dict):
                cls.add_coords(kwargs, **key)
            else:
                if not isinstance(key, Iterable):
                    raise TypeError(
                        "keys can be str, dict or iterables. %s not accepted." % key
                    )
                _args, _kwargs = cls.format_coords(*key)
                cls.add_coords(kwargs, **_kwargs)
                args.update(_args)
        return tuple(args), kwargs

    @classmethod
    def format_values(cls, *values, interval=None, compact=True):
        "Returns a list of values for the coordinate"
        vals = set()
        for value in values:
            if value is None:
                vals.add(value)
            elif isinstance(value, (int, str, range)):
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
        if None in vals:
            if len(vals) == 1:
                return None
            vals.remove(None)
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
                return ((key, val) for key, val in field.coords if key in field.indeces)
            return ()

        # Adding to resolved all the coordinates
        resolved = {}
        for axis, val in coords.items():
            if field is not None:
                indeces = field.get_indeces(axis)
                if not indeces:
                    raise ValueError("Index '%s' not in field" % axis)
            else:
                indeces = self.lattice.expand(axis)
            self.add_coords(resolved, **{idx: val for idx in indeces})

        for key in keys:
            coords = self.deduce(key)
            if field is not None:
                coords = {
                    index: val
                    for axis, val in coords.items()
                    for index in field.get_indeces(axis)
                }
                if not coords:
                    raise ValueError("'%s' not in field" % key)
            self.add_coords(resolved, **coords)

        # Checking the coordinates values
        for key, val in resolved.items():
            interval = self.lattice.get_axis_range(index_to_axis(key))
            resolved[key] = self.format_values(*val, interval=interval)

        if field is not None:
            for key, val in field.coords:
                if key in resolved:
                    if resolved[key] is None:
                        if len(set(expand_indeces(val))) > 1:
                            raise ValueError(
                                "None can only be assigned to an axis of size 1."
                            )
                    elif not set(expand_indeces(val)) >= set(
                        expand_indeces(resolved[key])
                    ):
                        raise ValueError(
                            "%s = %s not in field coordinates that has %s = %s"
                            % (key, resolved[key], key, val)
                        )
                elif key in field.indeces:
                    resolved[key] = val

        # Removing coordinates that are the whole axis
        for key, val in list(resolved.items()):
            if val == slice(None):
                del resolved[key]

        return tuple(resolved.items())

    def deduce(self, key):
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
            if key in labels:
                return {name: key}

        # TODO
        raise NotImplementedError
