"""
Extensions of Python standard functions
"""

__all__ = ["count", "FrozenDict"]

from functools import wraps
from itertools import count as _count


class count(_count):
    "Extension of itertools.count. Add __call__ method"

    def __call__(self, num):
        for _ in range(num):
            yield next(self)


class FrozenDict(dict):
    "Frozable dictionary"

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self._allows_new = True
        self._allows_changes = True
        return self

    @property
    def frozen(self):
        """
        Returns if the current instance is frozen, i.e. cannot be changed anymore.
        To unfreeze it use .copy().
        """
        return not self._allows_new or not self._allows_changes

    @frozen.setter
    def frozen(self, value):
        if value != self.frozen:
            if not value is True:
                raise ValueError(
                    "Frozen can only be changed to True. To unfreeze do a copy."
                )
            self.allows_new = False
            self.allows_changes = False

    @property
    def allows_new(self):
        "Returns if the current instance allows new keys"
        return self._allows_new

    @allows_new.setter
    def allows_new(self, value):
        if value != self.allows_new:
            if not value is False:
                raise ValueError(
                    "Allows_new can only be changed to False. To unfreeze do a copy."
                )
            self._allows_new = value

    @property
    def allows_changes(self):
        "Returns if the current instance allows changes to the values"
        return self._allows_new

    @allows_changes.setter
    def allows_changes(self, value):
        if value != self.allows_changes:
            if not value is False:
                raise ValueError(
                    "Allows_changes can only be changed to False. To unfreeze do a copy."
                )
            self._allows_changes = value

    def freeze(self, allows_new=False, allows_changes=False):
        "Returns a frozen copy of the dictionary"
        if self.allows_new == allows_new and self.allows_changes == allows_changes:
            return self
        copy = self.copy()
        copy.allows_new = allows_new
        copy.allows_changes = allows_changes
        return copy

    def __delitem__(self, key):
        if self.frozen:
            raise RuntimeError(
                "The dict has been frozen and %s cannot be deleted." % key
            )
        super().__delitem__(key)

    def __setitem__(self, key, val):
        if key in self:
            if not self.allows_changes:
                raise RuntimeError(
                    "The dict has been frozen and %s cannot be changed." % key
                )
        elif not self.allows_new:
            raise RuntimeError("The dict has been frozen and %s cannot be added." % key)
        super().__setitem__(key, val)

    @wraps(dict.copy)
    def copy(self):
        return type(self)(self)

    @wraps(dict.update)
    def update(self, val=None):
        if not val:
            return
        for _k, _v in dict(val).items():
            self[_k] = _v

    @wraps(dict.setdefault)
    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
