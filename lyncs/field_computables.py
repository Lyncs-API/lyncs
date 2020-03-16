from .tunable import computable

@computable
def field_shape(shape, axes_order):
    keys, vals = zip(*shape)
    keys, vals = list(keys), list(vals)
    shape = []
    for key in axes_order:
        idx = keys.index(key)
        shape.append(vals[idx])
        keys.pop(idx)
        vals.pop(idx)
    return tuple(shape)


@computable
def field_chunks(shape, chunks, axes_order):
    keys, vals = zip(*shape)
    Skeys, Svals = list(keys), list(vals)
    keys, vals = zip(*chunks)
    Ckeys, Cvals = list(keys), list(vals)
    chunks = []
    for key in axes_order:
        if key in Ckeys:
            idx = Ckeys.index(key)
            chunks.append(Cvals[idx])
            Ckeys.pop(idx)
            Cvals.pop(idx)
            idx = Skeys.index(key)
            Skeys.pop(idx)
            Svals.pop(idx)
        else:
            idx = Skeys.index(key)
            chunks.append(Svals[idx])
            Skeys.pop(idx)
            Svals.pop(idx)

    return tuple(chunks)


@computable
def getitem(field, axes, axes_order, **coords):
    mask = [slice(None) for i in axes]
    for key,val in coords.items():
        mask[axes_order.index(key)] = val

    return field[tuple(mask)]


@computable
def setitem(field, setitem, axes, axes_order, **coords):
    mask = [slice(None) for i in axes]
    for key,val in coords.items():
        mask[axes_order.index(key)] = val

    field[tuple(mask)] = setitem
    return field


@computable
def squeeze(field, new_axes, old_axes_order, old_field_shape):
    from collections import Counter
    axes = []
    for i, (axis, size) in enumerate(zip(list(old_axes_order), old_field_shape)):
        if axis in new_axes and size>1:
            continue
        elif axis in new_axes and new_axes.count(axis) == old_axes_order.count(axis):
            continue

        assert size==1, "Trying to squeeze axis (%s) with size (%s) larger than one" % (axis,size)
        old_axes_order.remove(axis)
        axes.append(i)
    assert Counter(new_axes) == Counter(old_axes_order), "This should not happen"
    return field.squeeze(axis=tuple(axes))


@computable
def rechunk(field, field_chunks):
    return field.rechunk(field_chunks)
        

@computable
def reorder(field, new_axes_order, old_axes_order):
    from collections import Counter
    assert Counter(new_axes_order) == Counter(old_axes_order), """
    Got not compatible new_ and old_axes_order:
    new_axes_order = %s
    old_axes_order = %s
    """ % (new_axes_order, old_axes_order)
    old_indeces = list(range(len(old_axes_order)))
    axes = []
    for key in new_axes_order:
        idx = old_indeces[old_axes_order.index(key)]
        axes.append(idx)
        old_axes_order.remove(key)
        old_indeces.remove(idx)

    return field.transpose(*axes)


@computable
def roll(field, to_roll, axes_order, **kwargs):
    from dask.array import roll
    indeces = []
    shifts = []
    for axis, shift in to_roll.items():
        axis_indeces = []
        last_index = -1
        count = axes_order.count(axis)
        for i in range(count):
            axis_indeces.append(axes_order.index(axis, last_index+1))
            last_index = axis_indeces[-1]
        if count>1:
            axis_order = kwargs[key+"_order"]
        else:
            axis_order = [0]
        for idx,pos in zip(axis_indeces, axis_order):
            indeces.append(idx)
            shifts.append(shift[pos])

    return roll(field, tuple(shifts), axis=tuple(indeces))


@computable
def transpose(field, axis, axes_order, new_order, old_order):
    assert axes_order.count(axis) == len(new_order) and len(new_order) == len(old_order) and \
        len(new_order) == len(set(new_order)) and set(new_order) == set(old_order), """
    Got wrong parameters for performing transpose.
        axis: %s
        axes_order: %s
        new_order: %s
        old_order: %s
    """ % (axis, axes_order, new_order, old_order)
    old_axes_order = list(axes_order)
    new_axes_order = list(axes_order)
    for new, old in zip(new_order,old_order):
        idx = old_axes_order.index(axis)
        old_axes_order[idx] = axis+str(old)
        new_axes_order[idx] = axis+str(new)

    axes = []
    indeces = list(range(len(axes_order)))
    for axis in new_axes_order:
        idx = indeces[old_axes_order.index(axis)]
        axes.append(idx)
        old_axes_order.remove(axis)
        indeces.remove(idx)

    return field.transpose(*axes)
