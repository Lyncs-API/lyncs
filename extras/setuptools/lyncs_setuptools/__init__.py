from .version import *
from .data_files import *
from .description import *
from .classifiers import *
from .cmake import *

__version__ = "0.0.6"


def setup(*args, **kwargs):
    from setuptools import find_packages
    
    if args:
        assert len(args)==1, "Only one arg allowed and it will be threated as name."
        assert "name" not in kwargs, "Repeated name parameter"
        kwargs["name"] = args[0]

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

    kwargs.setdefault('install_requires', [])
    if "name" in kwargs and kwargs["name"] != "lyncs_setuptools":
        kwargs['install_requires'].append("lyncs-setuptools")

    kwargs.setdefault('extras_require', {})
    if kwargs['extras_require'] and "all" not in kwargs['extras_require']:
        _all = set()
        for val in kwargs['extras_require'].values():
            _all = _all.join(val)
        kwargs['extras_require']['all'] = list(_all)
        
    kwargs.setdefault('data_files', [])
    kwargs['data_files'] += get_data_files()

    if __name__ != "__main__":
        from setuptools import setup, find_packages
        setup(**kwargs)
    else:
        import sys
        global __argv__
        
        if len(__argv__)==1:
            assert __argv__[0] in kwargs, "Allowed options are '%s'" % ("', '".join(kwargs))
            print(kwargs[__argv__[0]])
        else:
            for key, res in kwargs.items():
                if type(res) is str and "\n" in res:
                    res = "\"\"\"\n" + res + "\n\"\"\""
                elif type(res) is str:
                    res = "\"" + res + "\""
                elif type(res) is list and res:
                    res = "[\n" + ",\n".join((repr(i) for i in res)) + "\n]"                
                else:
                    res = repr(res)
                    
                res = res.replace("\n", "\n |  ")
                if not __argv__ or key in __argv__:
                    print("%s: %s\n" % (key, res))

__argv__=[]
def main(argv=None):
    global __name__
    __name__ = "__main__"

    import sys, os
    sys.path.insert(0, os.getcwd())
    
    global __argv__
    if argv is None:
        __argv__ = sys.argv[1:]
    else:
        __argv__ = argv
        
    try:
        import setup
    except:
        global setup
        setup()
            
