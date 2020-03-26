from .version import *
from .data_files import *
from .description import *
from .classifiers import *
from .cmake import *

__version__ = "0.0.4"


# TODO: authomatize the following
AUTHOR='Simone Bacchio'
AUTHOR_EMAIL='s.bacchio@gmail.com'
URL='https://github.com/sbacchio'


def setup(name, **kwargs):
    from setuptools import setup, find_packages

    kwargs.setdefault('author', 'Simone Bacchio')
    kwargs.setdefault('author_email', 's.bacchio@gmail.com')
    kwargs.setdefault('url', 'https://github.com/sbacchio')
    kwargs.setdefault('version', find_version())
    kwargs.setdefault('packages', find_packages())
    kwargs.setdefault('classifiers', classifiers)

    if 'long_description' not in kwargs:
        dshort, dlong, dtype = find_description()
        kwargs.setdefault('description', dshort)
        kwargs.setdefault('long_description', dlong)
        kwargs.setdefault('long_description_content_type', dtype)
    
    if 'ext_modules' in kwargs:
        kwargs.setdefault('cmdclass', dict())
        kwargs['cmdclass'].setdefault("build_ext", CMakeBuild)

    kwargs.setdefault('install_require', [])
    kwargs['install_require'].append("lyncs_setuptools")

    kwargs.setdefault('extras_require', {})
    if kwargs['extras_require'] and "all" not in kwargs['extras_require']:
        _all = set()
        for val in kwargs['extras_require'].values():
            _all = _all.join(val)
        kwargs['extras_require']['all'] = list(_all)
        
    kwargs.setdefault('data_files', [])
    kwargs['data_files'] += get_data_files()

    setup(name=name, **kwargs)
    
    
switcher = {
    "author": AUTHOR,
    "author_email": AUTHOR_EMAIL,
    "url": URL,
    "version": find_version,
    "description": lambda: find_description()[0],
    "long_description_content_type": lambda: find_description()[2],
    "long_description": lambda: find_description()[1],
    "classifiers": classifiers,
    "data_files": get_data_files,
}


def main(arg="all"):

    def run(arg):
        if callable(arg):
            return arg()
        else:
            return arg

    if arg in switcher:
        return run(switcher[sys.argv[1]])
    else:
        assert arg=="all", "Allowed options are 'all', '%s'" % ("', '".join(switcher.keys()))
        accum = ""
        for key, fnc in switcher.items():
            res = run(fnc)
            
            if type(res) is str and "\n" in res:
                res = "\"\"\"\n" + res + "\n\"\"\""
            elif type(res) is str:
                res = "\"" + res + "\""
            elif type(res) is list and res:
                res = "[\n" + ",\n".join((repr(i) for i in res)) + "\n]"                
            else:
                res = repr(res)
                
            res = res.replace("\n", "\n |  ")
            accum += "%s: %s\n\n" % (key, res)
        return accum
            
