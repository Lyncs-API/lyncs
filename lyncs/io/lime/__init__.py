
# List of available implementations
engines = [
    "pylime",
    ]

from . import lime as pylime
default_engine = "pylime"

# from lyncs import config
# if config.clime_enabled:
#     import .clime
#     engines.append("clime")

# TODO: add more, e.g. lemon

def get_engine(engine=None):
    import sys
    engine = engine or default_engine
    self = sys.modules[__name__]
    return getattr(self,engine)


def is_compatible(filename):
    try:
        return pylime.is_lime_file(filename)
    except:
        return False
   

def get_lattice(records):
    assert "ildg-format" in records, "ildg-format not found"

    from lyncs import Lattice
    import xmltodict
    info = xmltodict.parse(records["ildg-format"])["ildgFormat"]

    return Lattice(dims={'t': int(info["lt"]),
                         'x': int(info["lx"]),
                         'y': int(info["ly"]),
                         'z': int(info["lz"])},
                   dofs = "QCD",
                   dtype = "complex%d"%(int(info["precision"])*2))


def get_field_type(records):
    assert "ildg-format" in records, "ildg-format not found"
    
    import xmltodict
    info = xmltodict.parse(records["ildg-format"])["ildgFormat"]

    if info["field"] in ["su3gauge",]:
        return "gauge_links"
    else:
        # TODO
        assert False, "To be implemented"
    

def get_shape_order(field_type):
    shape_order = ['x', 'y', 'z', 't']
    
    if field_type == "gauge_links":
        shape_order.extend(['n_dims', 'color', 'color'])
    else:
        # TODO
        assert False, "To be implemented"
        
    return shape_order
    

def get_type(filename, lattice=None, field_type=None, **kwargs):
    records = pylime.scan_file(filename)
    records = {r["lime_type"]: (r["data"] if "data" in r else r["data_length"]) for r in records}
    
    assert "ildg-binary-data" in records, "ildg-binary-data not found"

    read_lattice = None
    try:
        read_lattice = get_lattice(records)
    except AssertionError:
        if not lattice: raise

    if lattice and read_lattice:
        assert lattice == read_lattice, "Given lattice not compatible with the one read from file"
    else:
        lattice = read_lattice

    read_field_type = None
    try:
        read_field_type = get_field_type(records)
    except AssertionError:
        if not field_type: raise

    if field_type and read_field_type:
        assert field_type == read_field_type, "Given field_type not compatible with the one read from file"
    else:
        field_type = read_field_type

    from lyncs import Field
    
    field = Field(
        lattice=lattice,
        field_type=field_type,
        shape_order = get_shape_order(field_type),
    )
    
    assert field.byte_size == records["ildg-binary-data"], """
        Size of deduced field (%s) is not compatible with size of data (%s).
        """ %(field.byte_size, records["ildg-binary-data"])
    
    return field


from ...tunable import Tunable, tunable_property, delayed

class file_manager(Tunable):
    
    def __init__(self, filename, **kwargs):
        from ...tunable import Choice
        self.filename = filename
        
        Tunable.__init__(
            self,
            lime_engine=Choice(engines),
        )

    @tunable_property
    def engine(self):
        return get_engine(self.lime_engine)

    
    def read(self, chunk_id=None):
        return delayed(self.engine.read)(chunk_id = chunk_id)

    def __dask_tokenize__(self):
        from dask.base import normalize_token
            
        return normalize_token(
            (type(self), self.filename)
        )
