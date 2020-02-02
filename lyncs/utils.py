"""
Some recurring utils used all along the package
"""

def default_repr(self):
    """
    Default repr used by lyncs
    """
    ret = type(self).__name__+"("
    pad = " "*(len(ret))
    nvars = len(self.__init__.__code__.co_varnames)
    for i in range(1,nvars):
        arg = self.__init__.__code__.co_varnames[i]
        if hasattr(self, arg):
            arg_eq = arg+" = "
            val = repr(getattr(self,arg)).replace("\n", "\n"+pad+" "*(len(arg_eq)))
            ret += arg+" = "+val
            if i < nvars-1: ret += ",\n"+pad
    ret += ")"
    return ret
