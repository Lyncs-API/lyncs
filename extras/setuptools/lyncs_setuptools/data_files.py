__all__ = [
    "get_data_files",
    "add_to_data_files",
]

_data_files = {}

def is_subdir(path):
    """returns true if the path is a subdirectory"""
    import os
    p1 = os.getcwd()
    p2 = os.path.realpath(path)
    return p1 == p2 or p2.startswith(p1+os.sep)


def _add_to_data_files(directory, filename):
    assert is_subdir(directory), "Given directory is not a subdir %s" % directory
    if directory in _data_files:
        _data_files[directory].append(filename)
    else:
        _data_files[directory] = [filename]
    

def add_to_data_files(*files, directory=None):
    import os
    for filename in files:
        if directory:
            _add_to_data_files(directory, filename)
        else:
            assert is_subdir(filename), "If directory is not given, then the file must be in a subdir"
            filename = os.path.realpath(filename)[len(os.getcwd())+1:].split(os.sep)
            if len(filename)==1:
                _add_to_data_files(".", filename[0])
            else:
                _add_to_data_files(os.sep.join(filename[:-1]), os.sep.join(filename))


def get_data_files():
    print(_data_files)
    return list(_data_files.items())
 
