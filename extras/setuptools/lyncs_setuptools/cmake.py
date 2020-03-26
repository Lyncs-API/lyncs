__all__ = [
    "CMakeExtension",
    "CMakeBuild",
    ]
    

from setuptools import Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, source_dir='.', cmake_args=[]):
        import os
        
        source_dir = source_dir or '.'
        
        sources = [source_dir+"/CMakeLists.txt"]
        if os.path.exists(source_dir+"/patches"):
            for filename in os.listdir(source_dir+"/patches"):
                sources += [source_dir+"/patches/"+filename]
                
        Extension.__init__(self, name, sources=sources)
        self.source_dir = os.path.abspath(source_dir)
        self.cmake_args = cmake_args


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_extension(ext)
                self.extensions.remove(ext)
        if self.extensions:
            return build_ext.run(self)

    def build_extension(self, ext):
        import os, subprocess

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " + ext.name)
        
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
            
        subprocess.check_call(['cmake', ext.source_dir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
