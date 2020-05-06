"""
Aliasing or definition of base types
"""
# pylint: disable=C0103

__all__ = [
    "FrozenDict",
]

from types import MappingProxyType

FrozenDict = MappingProxyType
