"""
Set of contraction functions for array fields
"""
# pylint: disable=C0303,C0330

__all__ = [
    "dot",
    "einsum",
]

from collections import defaultdict
import numpy as np
from tuneit import Permutation
from ..utils import count
from .array import ArrayField, NumpyBackend, backend_method


def prepare_fields(*fields):
    "Auxiliary function for preparing the fields"

    fields = list(fields)
    fields[0], fields[1:] = fields[0].prepare(*fields[1:], elemwise=False)
    return tuple(fields)


def dot_indeces(*fields, closed_indeces=None, open_indeces=None):
    "Auxiliary function for formatting the indeces of the dot product"

    axes = set()
    for field in fields:
        axes.update(field.axes)

    if closed_indeces is not None:
        if isinstance(closed_indeces, str):
            closed_indeces = [closed_indeces]
        tmp = set()
        for axis in closed_indeces:
            for field in fields:
                tmp.update(field.get_axes(axis))

        closed_indeces = tmp
        assert closed_indeces.issubset(axes), "Trivial assertion."
        axes = axes.difference(closed_indeces)
    else:
        closed_indeces = set()

    if open_indeces is not None:
        if isinstance(open_indeces, str):
            open_indeces = [open_indeces]
        tmp = set()
        for axis in open_indeces:
            for field in fields:
                tmp.update(field.get_axes(axis))

        open_indeces = tmp
        if open_indeces.intersection(closed_indeces):
            raise ValueError("Close and open indeces cannot have axes in common.")
        assert open_indeces.issubset(axes), "Trivial assertion."
        axes = axes.difference(open_indeces)
    else:
        open_indeces = set()

    return axes, closed_indeces, open_indeces


def dot_prepare(*fields, axes=None, axis=None, closed_indeces=None, open_indeces=None):
    "Auxiliary function that prepares for a dot product checking the input"

    if (axis, axes, closed_indeces).count(None) < 2:
        raise KeyError(
            """
            Only one between axis, axes or closed_indeces can be used. They are the same parameters.
            """
        )

    if closed_indeces is None and open_indeces is None:
        closed_indeces = "dofs"

    closed_indeces = (
        axis if axis is not None else axes if axes is not None else closed_indeces
    )

    axes, closed_indeces, open_indeces = dot_indeces(
        *fields, closed_indeces=closed_indeces, open_indeces=open_indeces
    )

    counts = {}
    for field in fields:
        for key, num in field.axes_counts:
            if key in axes:
                if key not in counts:
                    counts[key] = num
                    continue

                if counts[key] != num:
                    raise ValueError(
                        """
                        Axis %s has count %s and %s for different fields.
                        Axes that are neither closes or open indeces,
                        must have the same count between all fields.
                        """
                        % (key, num, counts[key])
                    )

    return axes, closed_indeces, open_indeces


def trace_indeces(*field_indeces, axes=None):
    "Auxiliary function that traces field_indeces given in a einsum style"

    if not axes:
        return (field_indeces,)

    for key, val in tuple(field_indeces[-1].items()):
        if key in axes and len(val) > 1:
            idx = val[-1]
            if len(val) > 2:
                field_indeces[-1][key] = val[:-1] + (val[0],)
            else:
                del field_indeces[-1][key]
            for field in reversed(field_indeces[:-1]):
                if key in field and idx in field[key]:
                    assert idx == field[key][-1], "Trivial Assertion"
                    field[key] = field[key][:-1] + (val[0],)

    return field_indeces


def dot(
    *fields,
    axes=None,
    axis=None,
    closed_indeces=None,
    open_indeces=None,
    reduced_indeces=None,
    trace=False,
    average=False,
    debug=False
):
    """
    Performs the dot product between fields.

    Default behaviors:
    ------------------

    Contractions are performed between only degree of freedoms of the fields, e.g. field.dofs.
    For each field, indeces are always contracted in pairs combining the outer-most free index
    of the left with the inner-most of the right.

    I.e. dot(*fields) = dot(*fields, axes="dofs")

    Parameters:
    -----------
    fields: Field
        List of fields to perform dot product between.
    axes: str, list
        Axes where the contraction is performed on.
        Indeces are contracted in pairs combining the outer-most free index
        of the left with the inner-most of the right.
    axis: str, list
        Same as axes.
    closed_indeces: str, list
        Same as axes.
    open_indeces: str, list
        Opposite of close indeces, i.e. the axes that are left open.
    reduced_indeces: str, list
        List of indeces to sum over and not available in the output field.
    average: bool
        If True, the reduced_indeces are averaged, i.e. result/prod(reduced_indeces.size).
    trace: bool
        If True, then the closed indeces are also traced
    debug: bool
        If True, then the output are the contraction indeces

    Examples:
    ---------
    dot(vector, vector, axes="color")
      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,spin]
      [X,Y,Z,T, mu , c_0 ] x [X,Y,Z,T, mu , c_0 ] -> [X,Y,Z,T, mu ]

    dot(vector, vector, closed_indeces="color", open_indeces="spin")
      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,spin,spin]
      [X,Y,Z,T, mu , c_0 ] x [X,Y,Z,T, nu , c_0 ] -> [X,Y,Z,T, mu , nu ]

    dot(gauge, gauge, closed_indeces="color", trace=True)
      [x,y,z,t,color,color] x [x,y,z,t,color,color] -> [x,y,z,t]
      [X,Y,Z,T, c_0 , c_1 ] x [X,Y,Z,T, c_1 , c_0 ] -> [X,Y,Z,T]
    """
    fields = prepare_fields(*fields)
    axes, closed_indeces, open_indeces = dot_prepare(
        *fields,
        axis=axis,
        axes=axes,
        closed_indeces=closed_indeces,
        open_indeces=open_indeces,
    )

    counter = count()
    field_indeces = []
    new_field_indeces = defaultdict(tuple)
    for field in fields:
        field_indeces.append({})
        for key, num in field.axes_counts:

            if key in axes:
                if key not in new_field_indeces:
                    new_field_indeces[key] = tuple(counter(num))
                field_indeces[-1][key] = tuple(new_field_indeces[key])

            elif key in open_indeces:
                field_indeces[-1][key] = tuple(counter(num))
                new_field_indeces[key] += field_indeces[-1][key]

            else:
                assert key in closed_indeces, "Trivial assertion."
                if key not in new_field_indeces:
                    new_field_indeces[key] = tuple(counter(num))
                    field_indeces[-1][key] = tuple(new_field_indeces[key])
                else:
                    assert len(new_field_indeces[key]) > 0, "Trivial assertion."
                    field_indeces[-1][key] = (new_field_indeces[key][-1],) + tuple(
                        counter(num - 1)
                    )
                    new_field_indeces[key] = (
                        new_field_indeces[key][:-1] + field_indeces[-1][key][1:]
                    )
                    if len(new_field_indeces[key]) == 0:
                        del new_field_indeces[key]

    field_indeces.append(dict(new_field_indeces))

    if trace:
        field_indeces = trace_indeces(*field_indeces, axes=closed_indeces)

    if average:
        pass

    return einsum(*fields, indeces=field_indeces, debug=debug)


ArrayField.dot = dot
ArrayField.__matmul__ = dot


def einsum(*fields, indeces=None, debug=False):
    """
    Performs the einsum product between fields.

    Parameters:
    -----------
    fields: Field
        List of fields to perform the einsum between.
    indeces: list of dicts of indeces
        List of dictionaries for each field plus one for output field if not scalar.
        Each dictionary should have a key per axis of the field.
        Every key should have a list of indeces for every repetition of the axis in the field.
        Indeces must be integers.

    Examples:
    ---------
    einsum(vector, vector, indeces=[{'x':0,'y':1,'z':2,'t':3,'spin':4,'color':5},
                                    {'x':0,'y':1,'z':2,'t':3,'spin':4,'color':6},
                                    {'x':0,'y':1,'z':2,'t':3,'color':(5,6)} ])

      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,color,color]
      [0,1,2,3, 4  ,  5  ] x [0,1,2,3, 4  ,  6  ] -> [0,1,2,3,  5  ,  6  ]
    """
    fields = prepare_fields(*fields)
    if isinstance(indeces, dict):
        indeces = (indeces,)
    indeces = tuple(indeces)

    if not len(indeces) in (len(fields), len(fields) + 1):
        raise ValueError("A set of indeces per field must be given.")

    if not all((isinstance(idxs, dict) for idxs in indeces)):
        raise TypeError("Each set of indeces list must be a dictionary")

    for idxs in indeces:
        for key, val in list(idxs.items()):
            if isinstance(val, int) or len(val) == 1:
                new_key = fields[0].axes_to_indeces(key)[0]
                idxs[new_key] = val if isinstance(val, int) else val[0]
                if new_key != key:
                    del idxs[key]
                continue
            for i, _id in enumerate(val):
                new_key = key + "_%d" % i
                assert new_key not in idxs
                idxs[new_key] = _id
            del idxs[key]

    for (i, field) in enumerate(fields):
        if not set(indeces[i].keys()) == set(field.indeces):
            raise ValueError(
                """
                Indeces must be specified for all the field axes/indeces.
                For field %d,
                Got indeces: %s
                Field indeces: %s
                """
                % (i, tuple(indeces[i].keys()), field.indeces)
            )

    if debug:
        return indeces

    indeces_order = Permutation(
        list(indeces[-1].keys()), label="indeces_order", uid=True
    ).value
    # TODO: coords
    return fields[0].copy(
        fields[0].backend.contract(
            *(field.value for field in fields[1:]),
            *(field.indeces_order for field in fields),
            indeces=indeces,
            indeces_order=indeces_order,
        ),
        axes=fields[0].indeces_to_axes(*indeces[-1].keys()),
        indeces_order=indeces_order,
    )


ArrayField.einsum = einsum

SYMBOLS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


@backend_method
def contract(*fields_orders, indeces=None, indeces_order=None):
    "Implementation of contraction via einsum"
    assert len(fields_orders) % 2 == 0
    fields = fields_orders[: len(fields_orders) // 2]
    orders = fields_orders[len(fields_orders) // 2 :]

    symbols = []
    for order, idxs in zip(orders + (indeces_order,), indeces):
        symbols.append("")
        for idx in order:
            symbols[-1] += SYMBOLS[idxs[idx]]

    string = ",".join(symbols[:-1]) + "->" + symbols[-1]
    return np.einsum(string, *fields)


NumpyBackend.contract = contract
