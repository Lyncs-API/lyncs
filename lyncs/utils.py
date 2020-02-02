"""
Some recurring utils used all along the package
"""

def default_repr(self):
    """
    Default repr used by lyncs
    """
    ret = type(self).__name__+"("
    pad = " "*(len(ret))
    for arg in self.__init__.__code__.co_varnames[1:]:
        if hasattr(self, arg):
            arg_eq = arg+" = "
            val = repr(getattr(self,arg)).replace("\n", "\n"+pad+" "*(len(arg_eq)))
            ret += arg+" = "+val+",\n"+pad
    ret+=")"
    return ret
