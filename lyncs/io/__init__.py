# Formats must be either a file .py or a submodule in a directory
formats = [
    "lime",
    ]

for format in formats:
    try:
        exec("from . import %s"%format)
    except AssertionError:
        formats.remove(format)


_required_callables=[
    "is_compatible",
    "get_lattice",
    "get_field_type",
]


    
def deduce_format(filename):
    import sys
    self = sys.modules[__name__]
    for format in formats:
        if getattr(self,format).is_compatible(filename):
            return format
    assert False, "Impossible to deduce format"


def deduce_lattice(filename, format=None, field_type=None):
    import sys
    self = sys.modules[__name__]
    format  = format or deduce_format(filename)
    return getattr(self,format).get_lattice(filename, field_type=field_type)


def deduce_field_type(filename, format=None, lattice=None):
    import sys
    self = sys.modules[__name__]
    format  = format or deduce_format(filename)
    lattice = lattice or deduce_lattice(filename, format=format)
    return getattr(self,format).get_field_type(filename, field_type=field_type)


def get_reading_info(filename, format, **kwargs):
    import sys
    self = sys.modules[__name__]
    return getattr(self,format).get_reading_info(filename, **kwargs)


def read_data(info):
    import sys
    self = sys.modules[__name__]
    return getattr(self,info["format"]).read_data(info)
    
