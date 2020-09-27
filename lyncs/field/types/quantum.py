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
    def plaquette(self, dirs=None):
        dirs = dirs or self.get_range("dirs")
        if not set(dirs).issubset(self.get_range("dirs")):
            raise ValueError("Dirs not part of the field")
        if not len(dirs) > 1:
            raise ValueError("At least two dirs needed for computing plaquette")

        plaq = tuple(
            self[dir1].dot(
                self[dir2].roll(-1, dir1),
                self[dir1].roll(-1, dir2).H,
                self[dir2].H,
                trace=True,
                axes="all",
                mean=True,
            )
            for i, dir1 in enumerate(dirs[:-1])
            for dir2 in dirs[i + 1 :]
        ).real
        return plaq[0].add(plaq[1:]) / len(plaq)


class Spinor(Scalar):
    __axes__ = ["spin"]


class SpinMatrix(Spinor):
    __axes__ = ["spin!", "spin!"]
