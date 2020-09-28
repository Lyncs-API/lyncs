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
from lyncs_utils import count
from .array import ArrayField, NumpyBackend, backend_method


def prepare_fields(*fields):
    "Auxiliary function for preparing the fields"

    fields = list(fields)
    fields[0], fields[1:] = fields[0].prepare(*fields[1:], elemwise=False)
    return tuple(fields)


def dot_indexes(*fields, closed_indexes=None, open_indexes=None):
    "Auxiliary function for formatting the indexes of the dot product"

    axes = set()
    for field in fields:
        axes.update(field.axes)

    if closed_indexes is not None:
        if isinstance(closed_indexes, str):
            closed_indexes = [closed_indexes]
        tmp = set()
        for axis in closed_indexes:
            for field in fields:
                tmp.update(field.get_axes(axis))

        closed_indexes = tmp
        assert closed_indexes.issubset(axes), "Trivial assertion."
        axes = axes.difference(closed_indexes)
    else:
        closed_indexes = set()

    if open_indexes is not None:
        if isinstance(open_indexes, str):
            open_indexes = [open_indexes]
        tmp = set()
        for axis in open_indexes:
            for field in fields:
                tmp.update(field.get_axes(axis))

        open_indexes = tmp
        if open_indexes.intersection(closed_indexes):
            raise ValueError("Close and open indexes cannot have axes in common.")
        assert open_indexes.issubset(axes), "Trivial assertion."
        axes = axes.difference(open_indexes)
    else:
        open_indexes = set()

    return axes, closed_indexes, open_indexes


def dot_prepare(*fields, axes=None, axis=None, closed_indexes=None, open_indexes=None):
    "Auxiliary function that prepares for a dot product checking the input"

    if (axis, axes, closed_indexes).count(None) < 2:
        raise KeyError(
            """
            Only one between axis, axes or closed_indexes can be used. They are the same parameters.
            """
        )

    if closed_indexes is None and open_indexes is None:
        closed_indexes = "dofs"

    closed_indexes = (
        axis if axis is not None else axes if axes is not None else closed_indexes
    )

    axes, closed_indexes, open_indexes = dot_indexes(
        *fields, closed_indexes=closed_indexes, open_indexes=open_indexes
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
                        Axes that are neither closes or open indexes,
                        must have the same count between all fields.
                        """
                        % (key, num, counts[key])
                    )

    return axes, closed_indexes, open_indexes


def trace_indexes(*field_indexes, axes=None):
    "Auxiliary function that traces field_indexes given in a einsum style"

    if not axes:
        return (field_indexes,)

    for key, val in tuple(field_indexes[-1].items()):
        if key in axes and len(val) > 1:
            idx = val[-1]
            if len(val) > 2:
                field_indexes[-1][key] = val[:-1] + (val[0],)
            else:
                del field_indexes[-1][key]
            for field in reversed(field_indexes[:-1]):
                if key in field and idx in field[key]:
                    assert idx == field[key][-1], "Trivial Assertion"
                    field[key] = field[key][:-1] + (val[0],)

    return field_indexes


def dot(
    *fields,
    axes=None,
    axis=None,
    closed_indexes=None,
    open_indexes=None,
    reduced_indexes=None,
    trace=False,
    average=False,
    debug=False
):
    """
    Performs the dot product between fields.

    Default behaviors:
    ------------------

    Contractions are performed between only degree of freedoms of the fields, e.g. field.dofs.
    For each field, indexes are always contracted in pairs combining the outer-most free index
    of the left with the inner-most of the right.

    I.e. dot(*fields) = dot(*fields, axes="dofs")

    Parameters:
    -----------
    fields: Field
        List of fields to perform dot product between.
    axes: str, list
        Axes where the contraction is performed on.
        Indexes are contracted in pairs combining the outer-most free index
        of the left with the inner-most of the right.
    axis: str, list
        Same as axes.
    closed_indexes: str, list
        Same as axes.
    open_indexes: str, list
        Opposite of close indexes, i.e. the axes that are left open.
    reduced_indexes: str, list
        List of indexes to sum over and not available in the output field.
    average: bool
        If True, the reduced_indexes are averaged, i.e. result/prod(reduced_indexes.size).
    trace: bool
        If True, then the closed indexes are also traced
    debug: bool
        If True, then the output are the contraction indexes

    Examples:
    ---------
    dot(vector, vector, axes="color")
      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,spin]
      [X,Y,Z,T, mu , c_0 ] x [X,Y,Z,T, mu , c_0 ] -> [X,Y,Z,T, mu ]

    dot(vector, vector, closed_indexes="color", open_indexes="spin")
      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,spin,spin]
      [X,Y,Z,T, mu , c_0 ] x [X,Y,Z,T, nu , c_0 ] -> [X,Y,Z,T, mu , nu ]

    dot(gauge, gauge, closed_indexes="color", trace=True)
      [x,y,z,t,color,color] x [x,y,z,t,color,color] -> [x,y,z,t]
      [X,Y,Z,T, c_0 , c_1 ] x [X,Y,Z,T, c_1 , c_0 ] -> [X,Y,Z,T]
    """
    fields = prepare_fields(*fields)
    axes, closed_indexes, open_indexes = dot_prepare(
        *fields,
        axis=axis,
        axes=axes,
        closed_indexes=closed_indexes,
        open_indexes=open_indexes,
    )

    counter = count()
    field_indexes = []
    new_field_indexes = defaultdict(tuple)
    for field in fields:
        field_indexes.append({})
        for key, num in field.axes_counts:

            if key in axes:
                if key not in new_field_indexes:
                    new_field_indexes[key] = tuple(counter(num))
                field_indexes[-1][key] = tuple(new_field_indexes[key])

            elif key in open_indexes:
                field_indexes[-1][key] = tuple(counter(num))
                new_field_indexes[key] += field_indexes[-1][key]

            else:
                assert key in closed_indexes, "Trivial assertion."
                if key not in new_field_indexes:
                    new_field_indexes[key] = tuple(counter(num))
                    field_indexes[-1][key] = tuple(new_field_indexes[key])
                else:
                    assert len(new_field_indexes[key]) > 0, "Trivial assertion."
                    field_indexes[-1][key] = (new_field_indexes[key][-1],) + tuple(
                        counter(num - 1)
                    )
                    new_field_indexes[key] = (
                        new_field_indexes[key][:-1] + field_indexes[-1][key][1:]
                    )
                    if len(new_field_indexes[key]) == 0:
                        del new_field_indexes[key]

    field_indexes.append(dict(new_field_indexes))

    if trace:
        field_indexes = trace_indexes(*field_indexes, axes=closed_indexes)

    if average:
        pass

    return einsum(*fields, indexes=field_indexes, debug=debug)


ArrayField.dot = dot
ArrayField.__matmul__ = dot


def einsum(*fields, indexes=None, debug=False):
    """
    Performs the einsum product between fields.

    Parameters:
    -----------
    fields: Field
        List of fields to perform the einsum between.
    indexes: list of dicts of indexes
        List of dictionaries for each field plus one for output field if not scalar.
        Each dictionary should have a key per axis of the field.
        Every key should have a list of indexes for every repetition of the axis in the field.
        Indexes must be integers.

    Examples:
    ---------
    einsum(vector, vector, indexes=[{'x':0,'y':1,'z':2,'t':3,'spin':4,'color':5},
                                    {'x':0,'y':1,'z':2,'t':3,'spin':4,'color':6},
                                    {'x':0,'y':1,'z':2,'t':3,'color':(5,6)} ])

      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,color,color]
      [0,1,2,3, 4  ,  5  ] x [0,1,2,3, 4  ,  6  ] -> [0,1,2,3,  5  ,  6  ]
    """
    fields = prepare_fields(*fields)
    if isinstance(indexes, dict):
        indexes = (indexes,)
    indexes = tuple(indexes)

    if not len(indexes) in (len(fields), len(fields) + 1):
        raise ValueError("A set of indexes per field must be given.")

    if not all((isinstance(idxs, dict) for idxs in indexes)):
        raise TypeError("Each set of indexes list must be a dictionary")

    for idxs in indexes:
        for key, val in list(idxs.items()):
            if isinstance(val, int) or len(val) == 1:
                new_key = fields[0].axes_to_indexes(key)[0]
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
        if not set(indexes[i].keys()) == set(field.indexes):
            raise ValueError(
                """
                Indexes must be specified for all the field axes/indexes.
                For field %d,
                Got indexes: %s
                Field indexes: %s
                """
                % (i, tuple(indexes[i].keys()), field.indexes)
            )

    if debug:
        return indexes

    indexes_order = Permutation(
        list(indexes[-1].keys()), label="indexes_order", uid=True
    ).value
    # TODO: coords
    return fields[0].copy(
        fields[0].backend.contract(
            *(field.value for field in fields[1:]),
            *(field.indexes_order for field in fields),
            indexes=indexes,
            indexes_order=indexes_order,
        ),
        axes=fields[0].indexes_to_axes(*indexes[-1].keys()),
        indexes_order=indexes_order,
    )


ArrayField.einsum = einsum

SYMBOLS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


@backend_method
def contract(*fields_orders, indexes=None, indexes_order=None):
    "Implementation of contraction via einsum"
    assert len(fields_orders) % 2 == 0
    fields = fields_orders[: len(fields_orders) // 2]
    orders = fields_orders[len(fields_orders) // 2 :]

    symbols = []
    for order, idxs in zip(orders + (indexes_order,), indexes):
        symbols.append("")
        for idx in order:
            symbols[-1] += SYMBOLS[idxs[idx]]

    string = ",".join(symbols[:-1]) + "->" + symbols[-1]
    return np.einsum(string, *fields)


NumpyBackend.contract = contract
