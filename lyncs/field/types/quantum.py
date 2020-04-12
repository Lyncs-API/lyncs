from .generic import Scalar, Links

__all__ = [
    "Gauge",
    "GaugeLinks",
    "Spinor",
    "SpinMatrix",
]


class Gauge(Scalar):
    __axes__ = ["gauge"]


class GaugeLinks(Links, Gauge):
    pass


class Spinor(Scalar):
    __axes__ = ["spin"]


class SpinMatrix(Spinor):
    __axes__ = ["spin!", "spin!"]
