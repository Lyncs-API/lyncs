"""
Reductions for the fields
"""
# pylint: disable=C0303,C0330

__all__ = [
    "trace",
]

import numpy as np
from lyncs_utils import count
from .array import ArrayField, NumpyBackend, backend_method
from .contractions import einsum, trace_indexes, dot_prepare


def trace(self, *axes, **kwargs):
    """
    Performs the trace over repeated axes contracting the outer-most index with the inner-most.

    Parameters
    ----------
    axes: str
        If given, only the listed axes are traced.
        If the axes are two indexes of the field, then those two indexes are traced.
    """
    tmp, kwargs = ArrayField.get_input_axes(*axes, **kwargs)
    if kwargs:
        raise ValueError("Unknown kwargs %s" % kwargs)
    _, axes, _ = dot_prepare(self, axes=tmp)

    counts = dict(self.axes_counts)
    axes = tuple(axis for axis in axes if counts[axis] > 1)

    if not axes:
        return self

    if len(axes) == 1:
        if (
            len(tmp) == 2
            and set(tmp) <= set(self.indexes)
            and self.index_to_axis(tmp[0]) == self.index_to_axis(tmp[1])
        ):
            indexes = tmp
        else:
            indexes = sorted(self.get_indexes())
            indexes = (indexes[0], indexes[-1])
        axes = tuple(
            self.index_to_axis(idx) for idx in self.indexes if idx not in indexes
        )
        return self.copy(self.backend.trace(indexes, self.indexes_order), axes=axes)

    counter = count()
    indexes = {}
    for axis, num in counts.items():
        indexes[axis] = tuple(counter(num))

    indexes = trace_indexes(indexes, indexes, axes=axes)
    return einsum(self, indexes=indexes)


ArrayField.trace = trace

_ = backend_method(
    lambda self, indexes, indexes_order: self.trace(
        axis1=indexes_order.index(indexes[0]), axis2=indexes_order.index(indexes[1])
    )
)
_.__name__ = "trace"
NumpyBackend.trace = _


def reduction_method(key, fnc=None, doc=None):
    """
    Default implementation for field reductions

    Parameters
    ----------
    key: str
        The key of the method
    fnc: callable
        Fallback for the method in case self it is not a field
    """

    def method(self, *axes, **kwargs):
        if not isinstance(self, ArrayField):
            if fnc is None:
                raise TypeError(
                    "First argument of %s must be of type Field. Given %s"
                    % (key, type(self).__name__)
                )

            return fnc(self, *axes, **kwargs)

        axes, kwargs = self.get_input_axes(*axes, **kwargs)
        indexes = self.get_indexes(*axes) if axes else self.indexes
        axes = tuple(
            self.index_to_axis(idx) for idx in self.indexes if idx not in indexes
        )

        # Deducing the dtype of the output
        if fnc is not None:
            trial = fnc(np.ones((1), dtype=self.dtype), **kwargs)
        else:
            trial = getattr(np.ones((1), dtype=self.dtype), key)(**kwargs)

        if axes:
            result = getattr(self.backend, key)(indexes, self.indexes_order, **kwargs)
        else:
            result = getattr(self.backend, key)(**kwargs)

        if isinstance(trial, tuple):
            return tuple(
                (
                    self.copy(result[i], dtype=trial.dtype, axes=axes)
                    for i, trial in enumerate(trial)
                )
            )
        return self.copy(result, dtype=trial.dtype, axes=axes)

    method.__name__ = key

    if doc:
        method.__doc__ = doc
    elif fnc:
        method.__doc__ = fnc.__doc__

    return method


def backend_reduction_method(key, fnc=None, doc=None):
    """
    Returns a method for the backend that calls
    the given reduction (key) of the field value.
    """

    def method(self, indexes=None, indexes_order=None, **kwargs):
        if indexes is not None:
            kwargs["axis"] = tuple(indexes_order.index(idx) for idx in indexes)
        if fnc is None:
            return getattr(self, key)(**kwargs)
        return fnc(self, **kwargs)

    method.__name__ = key
    if doc is not None:
        method.__doc__ = doc
    elif fnc is not None:
        method.__doc__ = fnc.__doc__

    return method


REDUCTIONS = (
    ("any",),
    ("all",),
    ("min",),
    ("max",),
    ("argmin",),
    ("argmax",),
    ("sum",),
    ("prod",),
    ("mean",),
    ("std",),
    ("var",),
)

for (reduction,) in REDUCTIONS:
    __all__.append(reduction)
    globals()[reduction] = reduction_method(reduction, fnc=getattr(np, reduction))
    setattr(ArrayField, reduction, globals()[reduction])
    if hasattr(np.ndarray, reduction):
        fnc = backend_reduction_method(reduction, doc=getattr(np, reduction).__doc__)
    else:
        fnc = backend_reduction_method(reduction, fnc=getattr(np, reduction))
    backend_method(fnc, NumpyBackend)
