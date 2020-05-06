from .generic import Scalar, Links

__all__ = [
    "Gauge",
    "GaugeLinks",
    "Spinor",
    "SpinMatrix",
]


class Gauge(Scalar):
    __axes__ = ["gauge"]


class GaugeMatrix(Gauge):
    __axes__ = ["gauge!", "gauge!"]


class GaugeLinks(Links, GaugeMatrix):
    pass


class Spinor(Scalar):
    __axes__ = ["spin"]


class SpinMatrix(Spinor):
    __axes__ = ["spin!", "spin!"]
