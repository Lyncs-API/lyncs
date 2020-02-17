"""
Some recurring utils used all along the package
"""

def default_repr(self):
    """
    Default repr used by lyncs
    """
    from inspect import signature
    ret = type(self).__name__+"("
    pad = " "*(len(ret))
    found_first = False
    for key,val in signature(self.__init__).parameters.items():
        if key in dir(self) and val.kind == val.POSITIONAL_OR_KEYWORD:
            if found_first: ret += ",\n"+pad
            else: found_first = True
            arg_eq = key+" = "
            val = repr(getattr(self,key)).replace("\n", "\n"+pad+" "*(len(arg_eq)))
            ret += key+" = "+val
    ret += ")"
    return ret
