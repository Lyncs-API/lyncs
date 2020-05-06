"""
Some recurring utils used all along the package
"""

__all__ = [
    "to_list",
    "default_repr",
    "add_parameters_to_doc",
    "compute_property",
]

from copy import copy


def default_repr(self):
    """
    Default repr used by lyncs
    """
    from inspect import signature, _empty

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

            if found_first:
                ret += ",\n" + pad
            else:
                found_first = True

            val = repr(getattr(self, key)).replace(
                "\n", "\n" + pad + " " * (len(arg_eq))
            )
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
        if words and "Parameters" == line.split()[0]:
            found = True
        if found and not words:
            doc.insert(i, doc_params)
            return "\n".join(doc)

    return "\n".join(doc) + doc_params


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
            return copy(getattr(self, self.key))
        except AttributeError:
            setattr(self, self.key, super().__get__(obj, owner))
            return self.__get__(obj, owner)
