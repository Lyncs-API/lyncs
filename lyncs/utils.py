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


def add_parameters_to_doc(doc, doc_params):
    """
    Inserts doc_params in the first empty line after Parameters if possible.
    """
    doc = doc.split("\n")
    found=False
    for i,line in enumerate(doc):
        words = line.split()
        if words and "Parameters" == line.split()[0]:
            found=True
        if found and not words:
            doc.insert(i,doc_params)
            return "\n".join(doc)
        
    return "\n".join(doc)+doc_params
            
