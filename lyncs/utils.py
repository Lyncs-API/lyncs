"""
Some recurring utils used all along the package
"""

def default_repr(self):
    """
    Default repr used by lyncs
    """
    ret = type(self).__name__+"("
    pad = " "*(len(ret))
    found_first = False
    for arg in self.__init__.__code__.co_varnames:
        if arg in dir(self):
            if found_first: ret += ",\n"+pad
            else: found_first = True
            arg_eq = arg+" = "
            val = repr(getattr(self,arg)).replace("\n", "\n"+pad+" "*(len(arg_eq)))
            ret += arg+" = "+val
    ret += ")"
    return ret
