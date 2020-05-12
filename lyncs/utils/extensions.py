"""
Extensions of Python standard functions
"""

__all__ = ["count"]

from itertools import count as _count


class count(_count):
    def __call__(self, num):
        for _ in range(num):
            yield next(self)
