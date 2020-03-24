__all__ = [
    "CMakeExtension",
    "CMakeBuild",
    ]
    

from setuptools import Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='.', cmake_args=[]):
        import os
        
        sourcedir = sourcedir or '.'
        
        sources = [sourcedir+"/CMakeLists.txt"]
        if os.path.exists(sourcedir+"/patches"):
            for filename in os.listdir(sourcedir+"/patches"):
                sources += [sourcedir+"/patches/"+filename]
                
        Extension.__init__(self, name, sources=sources)
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_args = cmake_args


class CMakeBuild(build_ext):
    def run(self):
        import subprocess
        
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import os, subprocess
        
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DEXTERNAL_INSTALL_LOCATION=' + extdir]
        cmake_args += ext.cmake_args
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
