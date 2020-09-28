"""
Base tools for defining field types.
"""

__all__ = [
    "FieldType",
    "Axes",
]

import re
from types import MappingProxyType
from collections import OrderedDict


class FieldType(type):
    """
    Metaclass for the field types.
    Field types are special classes used to define properties of fields
    depending on the axes.

    Field types have the following restrictions
    - The name of the class must be **unique**. All the field
      types are stored in dictionary (FieldType.s) and this list is
      looked up for searching properties of the field.
    - The FieldType needs to define an attribute __axes__ to specify
      which axes are needed for the properties of the type.


    Behaviour of field type special attributes:
    __axes__: list of axes that identify the field type.
        One can use generic names as "dims", "dofs" or properties
        like "space" or any of the special dimensions that may
        be defined on the lattice. If the current lattice does not
        have dimensions with these names, the field type will be simply
        ignored.
        Special characters may follow the name of the dimension as
         "+", "!"...
        The "+" means that the dimension can appear more than once.
        If a dimension is not followed by any special character then
        the "+" behavious is applied; e.g. "dims" -> "dims+".
        The "!" means that the specific dimension can be repeated only
        once. E.g. "spin!" means that only one spin dimension must be
        present to be of this type. Repetition of the dimension with
        "!" increase the counter, e.g. ["spin!", "spin!"] = "spin!!"
        means that the spin dimension must appear twice to be of the type.
        When "!" is used for a group of dimensions, i.e. "dofs!" then
        means that all the dofs must appear and only once. Then
        ["dofs!", "dofs"] = "dofs!+" means that all the dofs must appear
        but repetitions are allowed.

    - __init__ =
    """

    __types__ = OrderedDict()
    s = MappingProxyType(__types__)
    BaseField = None
    Field = None

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        """
        Checks that the name of the class is unique and construct from
        the axes in the base classes (bases) the axes for the new class.
        """
        assert not kwargs, "kwargs not used"
        assert name not in cls.__types__, "A FieldType named %s already exists" % name

        axes = Axes()
        for base in bases:
            if isinstance(base, FieldType):
                axes += base.axes

        return {"__axes__": axes}

    def __new__(cls, name, bases, attrs, **kwargs):
        "Checks that __axes__ is a valid Axes"
        assert not kwargs, "kwargs not used"

        assert "__axes__" in attrs
        attrs["__axes__"] = Axes(attrs["__axes__"])
        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs, **kwargs):
        """
        Adds the class to the list of all FieldType and checks
        that is subclass of bases.
        """
        assert not kwargs, "kwargs not used"

        for base in bases:
            assert issubclass(
                cls, base
            ), """
            The axes defined in the class %s are not compatible
            with the parent class %s.
            Axes of %s: %s
            Axes of %s: %s
            """ % (
                cls,
                base,
                cls,
                cls.axes,
                base,
                base.axes,
            )

        FieldType.__types__[name] = cls
        FieldType.__types__.move_to_end(name, last=False)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        "Returns a Field with the correct axes"
        return cls.Field(*args, axes=kwargs.pop("axes", cls.axes.expand), **kwargs)

    def __subclasscheck__(cls, child):
        "Checks if child is subclass of class"
        return isinstance(child, FieldType) and cls.axes in child.axes

    def __instancecheck__(cls, field):
        "Checks if field is compatible with the class"
        if not isinstance(field, cls.BaseField):
            return False
        if cls.axes.labels not in field.lattice:
            return False
        axes = list(field.axes)
        for axis in field.lattice.expand(cls.axes.must):
            if axis not in axes:
                return False
            axes.remove(axis)
        axes = set(axes)
        for axis in cls.axes.may:
            if not axes.intersection(field.lattice.expand(axis)):
                return False
        return True

    @property
    def axes(cls):
        return cls.__axes__


class Axes(tuple):
    """
    Functionalities to parse the axes.
    """

    _get_label = re.compile(r"[a-zA-Z]([a-zA-Z0-9]|_[0-9]*[a-zA-Z])*")

    @classmethod
    def get_label(cls, key):
        return cls._get_label.match(key)[0]

    _get_count = re.compile(
        # r"([\+\!\?\*]|({([0-9]+,?)+(,...)?}))?$"
        r"[\!]*[\+]?$"
    )

    @classmethod
    def get_count(cls, key):
        return cls._get_count.search(key)[0]

    _check_key = re.compile(_get_label.pattern + _get_count.pattern)

    @classmethod
    def check_keys(cls, keys):
        for key in keys:
            if not cls._check_key.match(key):
                raise KeyError("Invalid key: %s." % key)

    def __new__(cls, axes=()):
        if isinstance(axes, cls):
            return axes

        if isinstance(axes, str):
            axes = (axes,)

        cls.check_keys(axes)

        tmp = dict()
        for axis in axes:
            clean = cls.get_label(axis)
            sym = "!" * axis.count("!") + tmp.get(clean, "")
            if axis == clean or axis[-1] == "+":
                if sym.endswith("+"):
                    sym = sym[:-1] + "!"
                sym += "+"
            tmp[clean] = sym

        axes = tuple((key + val for key, val in tmp.items()))

        return super().__new__(Axes, axes)

    def __add__(self, axes):
        return Axes(super().__add__(Axes(axes)))

    def __contains__(self, axes):
        axes = Axes(axes)
        axes = dict(zip(axes.labels, axes.counts))
        this = dict(zip(self.labels, self.counts))
        return all((axis in this for axis in axes)) and all(
            (
                len(count) <= len(this[axis])
                if count[-1] == "+"
                else count == this[axis]
                for axis, count in axes.items()
            )
        )

    @property
    def expand(self):
        axes = []
        for axis, count in zip(self.labels, self.counts):
            axes += [axis] * len(count)
        return tuple(axes)

    @property
    def must(self):
        axes = []
        for axis, count in zip(self.labels, self.counts):
            axes += [axis] * count.count("!")
        return tuple(axes)

    @property
    def may(self):
        axes = []
        for axis, count in zip(self.labels, self.counts):
            axes += [axis] * count.count("+")
        return tuple(axes)

    @property
    def labels(self):
        return tuple((self.get_label(axis) for axis in self))

    @property
    def counts(self):
        return tuple((self.get_count(axis) for axis in self))
