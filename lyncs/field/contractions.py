"""
Set of contraction functions for array fields
"""
# pylint: disable=C0303,C0330

__all__ = [
    "dot",
]

from collections import defaultdict
from ..utils import count
from .array import ArrayField, NumpyBackend


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

    if not all((isinstance(field, ArrayField) for field in fields)):
        raise ValueError("All fields must be of type field.")

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
                        % (key, num, counts[key],)
                    )

    return axes, closed_indeces, open_indeces


def trace_indeces(*field_indeces, axes=None):
    "Auxiliary function that traces field_indeces given in a einsum style"

    if not axes:
        return (field_indeces,)

    for key, val in tuple(field_indeces[-1]):
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
    trace=False,
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

    axes, closed_indeces, open_indeces = dot_prepare(
        *fields,
        axis=axis,
        axes=axes,
        closed_indeces=closed_indeces,
        open_indeces=open_indeces
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

    field_indeces.append(new_field_indeces)

    if trace:
        field_indeces = trace_indeces(*field_indeces, axes=closed_indeces)

    if debug:
        return field_indeces

    return einsum(*fields, indeces=field_indeces)
