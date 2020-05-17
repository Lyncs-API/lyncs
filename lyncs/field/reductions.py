"""
Reductions for the fields
"""

__all__ = []

from functools import wraps
import numpy as np
from tunable import function
from .ufuncs import default_method
from .array import ArrayField, NumpyBackend, uniform_input_axes


def wrap_reduction(fnc):
    "Wrapper for reduction functions"

    @wraps(fnc)
    def wrapped(self, *axes, **kwargs):

        # Extracting the axes to reduce
        axes, kwargs = uniform_input_axes(*axes, **kwargs)
        dtype = fnc(np.ones((1,), dtype=self.field.dtype), **kwargs).dtype
        if axes:
            reduce = self.field.get_indeces(*axes)
            indeces = list(self.field.indeces)
            for idx in set(reduce):
                indeces.remove(idx)

            axes = self.field.indeces_to_axes(*indeces)
            indeces_order = self.field.get_indeces_order(indeces)
            kwargs["axis"] = self.field.get_indeces_index(*reduce)
        else:
            axes = ()
            indeces_order = ()

        return dict(
            axes=axes,
            value=function(fnc, self.field.value, **kwargs),
            dtype=dtype,
            indeces_order=indeces_order,
        )

    return wrapped


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
    globals()[reduction] = default_method(reduction, fnc=getattr(np, reduction))
    setattr(ArrayField, reduction, globals()[reduction])
    setattr(NumpyBackend, reduction, wrap_reduction(getattr(np, reduction)))
