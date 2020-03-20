# Formats must be either a file .py or a submodule in a directory
formats = [
    "lime",
    ]

from importlib import import_module

modules = {}
for format in formats:
    try:
        modules[format.lower()] = import_module(".io.%s"%format, package="lyncs")
    except AssertionError:
        formats.remove(format)


_required_callables=[
    "is_compatible",
    "get_type",
    "file_manager",
]


def get_module(format, filename=None):
    import sys
    self = sys.modules[__name__]
    format = deduce_format(filename, format=format)
    return modules[format]


def deduce_format(filename, format=None):
    assert format or filename, "Impossible to deduce format" 
    if format:
        assert isinstance(format, str), "Format must be a string"
        format = format.lower()
        assert format in modules, "Unknown format %s" % format
        return format
    else:
        for format, module in modules.items():
            if module.is_compatible(filename):
                return format
        assert False, "Impossible to deduce format"

    
def deduce_type(filename, format=None, **kwargs):
    return get_module(format, filename=filename).get_type(filename, **kwargs)


def file_manager(obj, format=None, **kwargs):
    return get_module(format, filename=kwargs.get("filename", None)).file_manager(obj, **kwargs)
    
