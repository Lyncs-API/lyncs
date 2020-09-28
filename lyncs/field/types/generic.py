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
    "Scalar field, no axes specified"
    __axes__ = []

    # scalar methods


class Degrees(Scalar):
    "Field with degrees of freedom (dofs)"
    __axes__ = ["dofs"]

    # dofs operations, i.e. trace, dot


class Sites(Scalar):
    "Field on all the sites volume"
    __axes__ = ["dims"]

    # volume methods, i.e. reductions


class Links(Sites):
    "Field on all the links between sites"
    __axes__ += ["dirs"]
    # Oriented links


class Vector(Sites):
    "Vector Field on all the sites volume"
    __axes__ += ["dofs!"]


class Propagator(Sites):
    "Propagator Field on all the sites volume"
    __axes__ += ["dofs!!"]
