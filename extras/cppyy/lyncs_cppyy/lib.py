__all__ = [
    "Lib"
    ]

class Lib:
    __slots__ = ["_cwd", "_path", "_header", "_library", "_check", "_c_include"]
    
    def __init__(
            self,
            path = '.',
            header = [],
            library = [],
            check = [],
            c_include = False,
    ):
        """
        Initializes a library class that can be pickled.
        
        Parameters
        ----------
        path: str or list
          Path(s) where to look for headers and libraries. 
          Headers are searched in path+"/include" and libraries in path+"/lib".
        header: str or list
          Header(s) file to include.
        library: str or list
          Library(s) file to include. Also absolute paths are accepted.
        check: str or list
          Check function(s) to look for in the library to test if it has been loaded.
        c_include: bool
          Whether the library is a c library (False means it is a c++ library).
        """
        import os
        assert check, "No checks given."
        self._cwd = os.getcwd()
        self._path = [path] if isinstance(path, str) else path
        self._header = [header] if isinstance(header, str) else header
        self._library = [library] if isinstance(library, str) else library
        self._check = [check] if isinstance(check, str) else check
        self._c_include = c_include
        
    @property
    def path(self):
        return self._path
    
    @property
    def header(self):
        return self._header
    
    @property
    def library(self):
        return self._library
    
    @property
    def check(self):
        return self._check
    
    @property
    def c_include(self):
        return self._c_include
    
    @property
    def lib(self):
        import cppyy
        if all((hasattr(cppyy.gbl, check) for check in self.check)):
            return cppyy.gbl
        
        import os
        for header in self.header:
            for path in self.path:
                if not path.startswith(os.sep): path = self._cwd + "/" + path
                if os.path.isfile(path+"/include/"+header):
                    cppyy.add_include_path(path+"/include")
                    break
            if self.c_include:
                cppyy.c_include(header)
            else:
                cppyy.include(header)

        for library in self.library:
            tmp = library
            if not tmp.startswith(os.sep): tmp = self._cwd + "/" + tmp
            if os.path.isfile(tmp):
                cppyy.load_library(tmp)
            else:
                found=False
                for path in self.path:
                    if not path.startswith(os.sep): path = self._cwd + "/" + path
                    if os.path.isfile(path+"/lib/"+library):
                        cppyy.load_library(path+"/lib/"+library)
                        found=True
                        break
                assert found, "Library %s not found in paths %s" % (library, self.path)

        assert all((hasattr(cppyy.gbl, check) for check in self.check)), "Given checks not found."
        return cppyy.gbl
    
            
    def __getattr__(self, key):
        return getattr(self.lib, key)
