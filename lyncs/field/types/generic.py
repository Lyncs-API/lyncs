from .base import FieldType

__all__ = [
    "Scalar",
    "Degrees",
    "Sites",
    "Links",
    "Vector",
    "Propagator",
]


class Scalar(metaclass=FieldType):
    __axes__ = []

    # scalar methods


class Degrees(Scalar):
    __axes__ = ["dofs"]

    # dofs operations, i.e. trace, dot


class Sites(Scalar):
    __axes__ = ["dims"]

    # volume methods, i.e. reductions


class Links(Sites):
    __axes__ += ["dirs"]
    # Oriented links


class Vector(Sites):
    __axes__ += ["dofs!"]


class Propagator(Sites):
    __axes__ += ["dofs!!"]
