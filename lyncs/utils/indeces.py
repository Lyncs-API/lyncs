"""
Utils for indeces
"""

__all__ = [
    "compact_indeces",
]

from .class_utils import isiterable


def compact_indeces(indeces):
    """
    Returns a list of ranges or integers
    as they occur sequentially in the list

    Examples
    --------
    >>> list(compact_indeces([1, 2, 4, 6, 7, 8, 10, 12, 13]))
    [1, range(2, 7, 2), 7, range(8, 13, 2), 13]
    """
    if not isiterable(indeces, int):
        raise ValueError("Compact_indeces requires a list of integers")
    tmp = []
    step = 0
    for idx in indeces:
        if len(tmp) < 2:
            tmp.append(idx)
        else:
            if step == 0:
                step = tmp[1] - tmp[0]
                if step == 0 or idx - tmp[1] != step:
                    yield tmp.pop(0)
                    tmp.append(idx)
                    step = 0
                else:
                    tmp[1] = idx
            else:
                if idx - tmp[1] == step:
                    tmp[1] = idx
                else:
                    yield range(tmp[0], tmp[1] + (1 if step > 0 else -1), step)
                    tmp = [
                        idx,
                    ]
                    step = 0
    if step == 0:
        yield from tmp
    else:
        yield range(tmp[0], tmp[1] + (1 if step > 0 else -1), step)
