"""
Base class of the Field type that implements
the interface to the Lattice class.
"""

from collections import OrderedDict
from .types.base import FieldType
from ..utils import default_repr, FrozenDict


class BaseField:
    """
    Base class of the Field type that implements
    the interface to the Lattice class and deduce
    the list of Field types from the field axes.
    
    The list of types are accessible via field.types
    """

    __repr__ = default_repr

    def __init__(self, field=None, axes=None, lattice=None, **kwargs):
        """
        The base field class.
        
        Parameters
        ----------
        field: instance of BaseField
            If given, then the missing parameters are deduced from it.
        axes: list(str)
            List of axes of the field.
        lattice: Lattice
            The lattice on which the field is defined.
        kwargs: dict
           Extra parameters that will be passed to the field types.
        """

        from ..lattice import Lattice, default_lattice

        assert lattice is None or isinstance(
            lattice, Lattice
        ), "lattice must be of Lattice type"

        if isinstance(field, BaseField):
            types = field.types
            self._lattice = (lattice or field.lattice).freeze()
            self._axes = self.lattice.expand(axes or field.axes)
        else:
            types = FieldType.s
            self._lattice = (lattice or default_lattice()).freeze()
            self._axes = self.lattice.expand(axes or ())

        self._types = OrderedDict()
        for name, ftype in types.items():
            if isinstance(self, ftype):
                self._types[name] = ftype
                try:
                    getattr(ftype, "__init__").__get__(self)(**kwargs)
                except AttributeError:
                    continue
        self._types = FrozenDict(self._types)

    @property
    def lattice(self):
        "The lattice on which the field is defined."
        return self._lattice

    @property
    def axes(self):
        "List of axes of the field. Order is not significant. See field.axes_order."
        return self._axes

    @property
    def types(self):
        "List of field types that the field is instance of."
        return self._types

    def __getattr__(self, key):
        "Looks up for methods in the field types"
        for ftype in self.types.values():
            if isinstance(self, ftype):
                try:
                    return getattr(ftype, key).__get__(self)
                except AttributeError:
                    continue

        raise AttributeError

    @property
    def name(self):
        "Name of the Field. Equivalent to the most relevant field type."
        return [
            name
            for name, ftype in sorted(
                self.types.items(),
                key=lambda item: self.lattice.expand(item[1].axes.expand),
            )
        ][-1]


FieldType.Field = BaseField
