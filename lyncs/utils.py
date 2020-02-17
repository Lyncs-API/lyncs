"""
Some recurring utils used all along the package
"""

def default_repr(self):
    """
    Default repr used by lyncs
    """
    from inspect import signature, _empty
    ret = type(self).__name__+"("
    pad = " "*(len(ret))
    found_first = False
    for key,val in signature(self.__init__).parameters.items():
        if key in dir(self) and val.kind in [val.POSITIONAL_ONLY,
                                             val.POSITIONAL_OR_KEYWORD,
                                             val.KEYWORD_ONLY]:
            if val.kind == val.POSITIONAL_ONLY:
                arg_eq = ""
            elif val.kind == val.POSITIONAL_OR_KEYWORD and val.default == _empty:
                arg_eq = ""
            else:
                arg_eq = key+" = "
                
            if found_first: ret += ",\n"+pad
            else: found_first = True
            
            val = repr(getattr(self,key)).replace("\n", "\n"+pad+" "*(len(arg_eq)))
            ret += arg_eq+val
    ret += ")"
    return ret
