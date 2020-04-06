"""
Some recurring utils used all along the package
"""

__all__ = [
    "default_repr",
    "add_parameters_to_doc",
    "compute_property",
    "simple_property",
]

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
            

def to_list(*args):
    from ..tunable import computable, Delayed
    
    @computable
    def to_list(*args):
        return list(args)
    
    lst = to_list(*args)
    if isinstance(lst, Delayed):
        lst._length = len(args)
        
    return lst
    

def compute_property(key):
    """
    Computes a property once and then store it in self.*key*
    """
    from functools import wraps
    from ..tunable import Delayed
    from copy import copy
    
    def decorator(fnc):
    
        @property
        @wraps(fnc)
        def wrapped_property(self):
            try:
                value = getattr(self, key)
           
            except AttributeError:
                value = fnc(self)
                
                if isinstance(value, Delayed):
                    return value
                setattr(self, key, value)
                
            return copy(value)
            
        return wrapped_property
    
    return decorator


def simple_property(key, value, copy=True):
    """
    Creates a simple property, i.e. returns self.key if exists of the given value.
    """

    from functools import wraps
    if copy:
        from copy import copy
    else:
        copy = lambda value: value
    
    def decorator(fnc):
    
        @property
        @wraps(fnc)
        def wrapped_property(self):
            assert fnc(self) is None, "A default property should return None"
            return copy(self.__dict__.get(key, value))
        
        return wrapped_property
    
    return decorator
