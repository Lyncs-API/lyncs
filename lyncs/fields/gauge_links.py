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
        self.add_option("dirs_order", Permutation(list(self.dims)))

    for dir in self.dims:
        self.label(dir, n_dims=self.dirs_order.index(dir))
    


def plaquette(self, dirs=None):
    dirs = dirs or self.dims
    assert set(dirs).issubset(self.dims), "Dims not part of the field"
    assert len(dirs) > 1, "At least two dims needed for computing plaquette"

    plaq = 0
    count = 0
    for i,dir1 in enumerate(dirs[:-1]):
        for dir2 in dirs[i+1:]:
            plaq += self[dir1].dot(self[dir2].roll(dir1, -1),  self[dir1].roll(dir2, -1).H,
                                   self[dir2].H).trace().real
            count+=1
            
    return plaq.mean()/count/self.lattice["color"]
