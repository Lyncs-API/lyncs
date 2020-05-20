"""
Some recurring utils used all along the package
"""
# pylint: disable=C0303,C0330

__all__ = [
    "to_list",
    "default_repr",
    "add_parameters_to_doc",
    "add_kwargs_of",
    "compute_property",
    "single_true",
    "isiterable",
]

from collections.abc import Iterable
from types import MethodType
from copy import copy
from inspect import signature, _empty


def isiterable(obj, types=None):
    if types is None:
        return isinstance(obj, Iterable)
    return isiterable(obj) and all((isinstance(val, types) for val in obj))


def single_true(iterable):
    i = iter(iterable)
    return any(i) and not any(i)


def default_repr(self):
    """
    Default repr used by lyncs
    """

    ret = (self.type if hasattr(self, "type") else type(self).__name__) + "("
    pad = " " * (len(ret))
    found_first = False
    for key, val in signature(self.__init__).parameters.items():
        if key in dir(self) and val.kind in [
            val.POSITIONAL_ONLY,
            val.POSITIONAL_OR_KEYWORD,
            val.KEYWORD_ONLY,
        ]:
            if val.kind == val.POSITIONAL_ONLY:
                arg_eq = ""
            elif val.kind == val.POSITIONAL_OR_KEYWORD and val.default == _empty:
                arg_eq = ""
            else:
                arg_eq = key + " = "

            val = getattr(self, key)
            if isinstance(val, MethodType):
                continue
            val = repr(getattr(self, key)).replace(
                "\n", "\n" + pad + " " * (len(arg_eq))
            )

            if found_first:
                ret += ",\n" + pad
            else:
                found_first = True
            ret += arg_eq + val
    ret += ")"
    return ret


def add_parameters_to_doc(doc, doc_params):
    """
    Inserts doc_params in the first empty line after Parameters if possible.
    """
    doc = doc.split("\n")
    found = False
    for i, line in enumerate(doc):
        words = line.split()
        if not found and len(words) == 1 and words[0].startswith("Parameter"):
            found = True
        elif found and not words:
            doc.insert(i, doc_params)
            return "\n".join(doc)

    return "\n".join(doc) + doc_params


def get_parameters_doc(doc):
    """
    Extracts the documentation of the parameters
    """
    found = False
    parameters = []
    for line in doc.split("\n"):
        words = line.split()
        if not found and len(words) == 1 and words[0].startswith("Parameter"):
            found = True
        elif found and words:
            parameters.append(line)
        elif found and not words:
            break

    if found and parameters:
        return "\n".join(parameters[1:])
    return doc


def add_kwargs_of(fnc):
    """
    Decorator for adding kwargs of a function to another
    """

    def decorator(fnc2):
        args = []
        var_kwargs = False
        kwargs = []

        for key, val in signature(fnc2).parameters.items():
            if val.kind == val.POSITIONAL_ONLY or (
                val.kind == val.POSITIONAL_OR_KEYWORD and val.default == _empty
            ):
                args.append(key)
            elif val.kind == val.VAR_POSITIONAL:
                args.append("*" + key)
            elif val.kind == val.VAR_KEYWORD:
                var_kwargs = key
            else:
                kwargs.append((key, val.default))

        assert (
            var_kwargs is not False
        ), "Cannot append kwargs to a function without **kwargs."

        keys = [key for key, val in kwargs]
        kwargs += [
            (key, val.default)
            for key, val in signature(fnc).parameters.items()
            if val.kind in [val.POSITIONAL_OR_KEYWORD, val.KEYWORD_ONLY]
            and val.default != _empty
            and key not in keys
        ]

        args.extend(("%s=%s" % (key, val) for key, val in kwargs))
        args.append("**" + var_kwargs)

        args = ", ".join(args)
        fnc2.__dict__["__wrapped__"] = eval("lambda %s: None" % (args))
        fnc2.__doc__ = add_parameters_to_doc(
            fnc.__doc__, get_parameters_doc(fnc2.__doc__)
        )
        return fnc2

    return decorator


def to_list(*args):
    from ..tunable import computable, Delayed

    @computable
    def to_list(*args):
        return list(args)

    lst = to_list(*args)
    if isinstance(lst, Delayed):
        lst._length = len(args)

    return lst


class compute_property(property):
    """
    Computes a property once and store the result in key
    """

    @property
    def key(self):
        return getattr(self, "_key", "_" + self.fget.__name__)

    @key.setter
    def key(self, value):
        self._key = value

    def __get__(self, obj, owner):
        try:
            return copy(getattr(obj, self.key))
        except AttributeError:
            setattr(obj, self.key, super().__get__(obj, owner))
            return self.__get__(obj, owner)
