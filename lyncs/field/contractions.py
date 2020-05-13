"""
Set of contraction functions for array fields
"""
# pylint: disable=C0303,C0330

__all__ = [
    "dot",
    "trace",
]

from collections import defaultdict
from ..utils import count
from .array import ArrayField, NumpyBackend
from .base import index_to_axis


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

    field_indeces.append(dict(new_field_indeces))

    if trace:
        field_indeces = trace_indeces(*field_indeces, axes=closed_indeces)

    return einsum(*fields, indeces=field_indeces, debug=debug)


def trace(field, *axes):
    """
    Performs the trace over repeated axes contracting the outer-most index with the inner-most.
    
    Parameters
    ----------
    axes: str
        If given, only the listed axes are traced.
    """

    if (
        len(axes) == 2
        and set(axes) <= set(field.indeces)
        and index_to_axis(axes[0]) == index_to_axis(axes[1])
    ):
        return field.copy(**field.backend.trace(*axes))

    _, axes, _ = dot_prepare(field, axes=axes)

    counts = dict(field.axes_counts)
    axes = tuple(axis for axis in axes if counts[axis] > 1)

    if not axes:
        return field

    counter = count()
    indeces = {}
    for axis, num in counts.items():
        indeces[axis] = tuple(counter(num))

    indeces = trace_indeces(indeces, indeces, axes=axes)
    return einsum(field, indeces=indeces)


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
    if not all((isinstance(field, ArrayField) for field in fields)):
        raise ValueError("All fields must be of type field.")

    if isinstance(indeces, dict):
        indeces = (indeces,)
    indeces = tuple(indeces)

    if not len(indeces) in (len(fields), len(fields) + 1):
        raise ValueError("A set of indeces per field must be given.")

    if not all((isinstance(idxs, dict) for idxs in indeces)):
        raise TypeError("Each set of indeces list must be a dictionary")

    for idxs in indeces:
        for key, val in list(idxs.items()):
            if isinstance(val, int):
                continue
            if len(val) == 1:
                idxs[key] = val[0]
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

    return field[0].copy(**field[0].backend.einsum(*fields[1:], indeces=indeces))