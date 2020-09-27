from dataclasses import dataclass
from functools import wraps
import re
from numpy import random
from tuneit import Function
from .base import BaseField
from .array import backend_method


@dataclass(frozen=True)
class RandomFieldGenerator:
    field: BaseField
    seed: int = None

    @property
    def backend(self):
        "Returns the computational backend of the field (numpy.random)."
        return RandomBackend(self.field)

    @property
    def backend_kwargs(self):
        "Returns the list of field variables to be passed to the backend function"
        if self.seed is None:
            return {}
        kwargs = dict(seed=self.seed, indexes_order=self.field.indexes_order)
        kwargs.update(dict(self.field.labels_order))
        return kwargs

    def shuffle(self, *axes):
        "Randomly shuffles the content of the field along the axes"
        return self.field.copy(self.backend.shuffle(*axes, **self.backend_kwargs))

    def bytes(self):
        "Fills up the field with random bytes"
        return self.field.copy(self.backend.bytes(**self.backend_kwargs))


def random_method(fnc):
    "Wraps numpy random methods"

    @wraps(fnc)
    def method(self, *args, **kwargs):

        if "size" in kwargs:
            raise KeyError("The parameter 'size' has been disabled")
        if "out" in kwargs:
            raise KeyError("The parameter 'out' has been disabled")

        # Getting dtype of the resulting field
        dtype = kwargs.get("dtype", self.field.dtype)
        try:
            out = fnc(random.default_rng(), *args, dtype=dtype, size=1, **kwargs)
            kwargs["dtype"] = dtype
        except TypeError:
            out = fnc(random.default_rng(), *args, size=1, **kwargs)
            if dtype != self.field.dtype:
                raise TypeError(
                    "Cannot change dtype using '%s' function" % fnc.__name__
                )

        dtype = out.dtype
        return self.field.copy(
            getattr(self.backend, fnc.__name__)(*args, **kwargs, **self.backend_kwargs),
            dtype=dtype,
        )

    # Editing doc
    doc = method.__doc__.split("\n")
    assert doc[1].strip().startswith(method.__name__ + "(")
    assert "size=None" in doc[1]

    params = []
    for param in "size", "out":
        if param in doc[1]:
            params.append(param)
            doc[1] = re.sub("(, )?" + param + "=?['a-zA-Z_0-9']*", "", doc[1])
    doc.insert(1, "")
    if params:
        doc.insert(
            1,
            " " * 8
            + "The parameter(s) '%s' is disabled since deduced from the field."
            % "', '".join(params),
        )
    doc.insert(1, " " * 8 + "Note: this documentation has been copied from numpy.")

    method.__doc__ = "\n".join(doc)

    return method


@dataclass
class RandomBackend:
    field: BaseField

    def generate(self, fnc, *args, seed=None, indexes_order=None, **kwargs):
        if seed is None:
            return getattr(random.default_rng(), fnc)(*args, **kwargs, size=field.shape)
        raise NotImplementedError("Reproducible random number not implemented")

    @backend_method
    def shuffle(self, *axes, **kwargs):
        "Randomly shuffles the content of the field along the axes"
        pass

    @backend_method
    def bytes(self, **kwargs):
        "Fills up the field with random bytes"
        pass

    def __getattr__(self, key):
        if hasattr(random.Generator, key):
            return Function(self.generate, args=(key), label=key)
        return super().__getattr__(key)

    def __getstate__(self):
        return self.field


for _fnc in dir(random.Generator):
    if not _fnc.startswith("_") and _fnc not in (
        "bit_generator",
        "bytes",
        "shuffle",
        "permutation",
    ):
        setattr(
            RandomFieldGenerator, _fnc, random_method(getattr(random.Generator, _fnc))
        )
