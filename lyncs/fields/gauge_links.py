"""
Methods for gauge_links type
"""

__all__ = [
    "__init__",
    "plaquette",
    ]


def __init__(self, **kwargs):
    
    from ..tunable import Permutation

    if "dirs_order" not in self.tunable_options:
        self.add_tunable_option("dirs_order", Permutation(list(self.dims.keys())))

    for dir in self.dims.keys():
        self.label(dir, n_dims=[self.dirs_order.index(dir)])
    


def plaquette(self, dirs=None):
    dirs = set(dirs or self.dims.keys())
    assert dirs.issubset(self.dims), "Dims not part of the field"
    assert len(dirs) > 1, "At least two dims needed for computing plaquette"

    plaq = 0
    for dir1 in dirs:
        for dir2 in dirs:
            if dir1!=dir2:
                plaq += self[dir1].dot(self[dir2].shift(dir1, 1), "gauge_dofs") \
                                  .dot(self[dir1].shift(dir2, 1).conj(), "gauge_dofs") \
                                  .dot(self[dir2].conj(), "gauge_dofs").sum()
    return plaq
