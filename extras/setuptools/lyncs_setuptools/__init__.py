from .version import *
from .data_files import *
from .description import *
from .classifiers import *
from .cmake import *

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
        
    kwargs.setdefault('data_files', [])
    kwargs['data_files'] += get_data_files()

    setup(name=name, **kwargs)
    
