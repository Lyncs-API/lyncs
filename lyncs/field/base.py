"""
Basic utils for the Field class.
- implements the interface to the Lattice
"""


from logging import warning
from types import MappingProxyType
from collections import OrderedDict
from .types.base import FieldType
from ..utils import default_repr


class BaseField:
    __repr__ = default_repr

    def __init__(self, axes, lattice=None, types=None, **kwargs):
        from ..lattice import Lattice, default_lattice

        assert lattice is None or isinstance(
            lattice, Lattice
        ), "lattice must be of Lattice type"
        self._lattice = (lattice or default_lattice()).freeze()
        self._axes = self.lattice.expand(axes)

        types = types or FieldType.s
        self._types = OrderedDict()
        for name, ftype in types.items():
            if isinstance(self, ftype):
                self._types[name] = ftype
                try:
                    getattr(ftype, "__init__").__get__(self)(**kwargs)
                except AttributeError:
                    continue
        self._types = MappingProxyType(self._types)

    @property
    def lattice(self):
        return self._lattice

    @property
    def axes(self):
        return self._axes

    @property
    def types(self):
        return self._types

    def __getattr__(self, key):
        "Looks up for methods in the field types"
        for ftype in self.types:
            if isinstance(self, ftype):
                try:
                    return getattr(ftype, key).__get__(self)
                except AttributeError:
                    continue

        raise AttributeError


FieldType.Field = BaseField
