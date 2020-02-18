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
    "get_type",
    "file_manager",
]


def get_module(filename, format=None):
    import sys
    self = sys.modules[__name__]
    format = deduce_format(filename, format=format)
    return getattr(self,format)


def deduce_format(filename, format=format):
    if format:
        # TODO: implement alias for formats
        assert format in formats, "Unknown format %s" % format
        return format
    else:
        for format in formats:
            if get_module(filename, format=format).is_compatible(filename):
                return format
        assert False, "Impossible to deduce format"

    
def deduce_type(filename, format=None, **kwargs):
    return get_module(filename, format=format).get_type(filename, **kwargs)


def file_manager(filename, format=None, **kwargs):
    return get_module(filename, format=format).file_manager(filename, **kwargs)
    
