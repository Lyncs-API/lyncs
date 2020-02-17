"""
Extension of dask delayed with a tunable aware Delayed object.
"""

__all__ = [
    "delayed",
    "Tunable"
]


def to_delayed(obj, **attrs):
    """
    Converts a dask.Delayed object into a lyncs.Delayed object with parent the type of the object.
    """
    from dask.delayed import Delayed as DaskDelayed
    if isinstance(obj, Delayed):
        return obj

    elif isinstance(obj, DaskDelayed):

        def wrap_func(func):
            def wrapped(*args, **kwargs):
                return to_delayed(func(*args, **kwargs), **attrs)
            return wrapped

        obj_attrs = {}
        for attr in dir(type(obj)):
            if attr not in dir(Tunable):
                val = getattr(type(obj), attr)
                if callable(val):
                    obj_attrs[attr] = wrap_func(val)
        obj_attrs["__slots__"] = type(obj).__slots__

        obj_attrs.update(attrs)
        return type(type(obj).__name__, (Delayed, type(obj)), obj_attrs)(obj)

    else:
        return obj
    

    
def delayed(*args, **kwargs):
    """
    Equivalent to dask.delayed, but returns a lyncs.Delayed instead of a dask.Delayed object. 
    For help see dask.delayed.
    """
    from dask import delayed as dask_delayed
    return to_delayed(dask_delayed(*args, **kwargs))



class Delayed:
    """
    A lyncs.Delayed object is the same as a dask.Delayed object that also implements a tuning step.
    If in the dask graph there is an object of type Tunable, this will be tuned.
    The tune step is performed before any graph optimization or calculation.
    """
    def __init__(self, obj):
        self.__setstate__(obj.__getstate__())
                          
    
    def tune():
        if not self.tunable: return
        pass


    @property
    def tunable(self):
        def is_tunable(val):
            if isinstance(val, (list, tuple)):
                if len(val) == 0: return False
                else: return is_tunable(val[0]) or is_tunable(val[1:])
            else:
                if isinstance(val, Tunable): return val.tunable is True
                else: return False
        
        return is_tunable(self.dask.values())


    
class Tunable:
    __slots__ = ["_tunable_options", "_tuned_options", "_tuning"]
    
    def __init__(self, **kwargs):
        self._tunable_options = {}
        self._tuned_options = {}
        self._tuning = False
        
        for key,val in kwargs.items():
            self.add_tunable_option(key,val)

        
    @property
    def tunable(self):
        return bool(self.tunable_options)

    
    @property
    def tuned(self):
        return not self.tunable


    @property
    def tuning(self):
        return self._tuning

    
    @property
    def tunable_options(self):
        if hasattr(self, "_tunable_options"):
            return self._tunable_options.copy()
        else:
            return {}

    
    @property
    def tuned_options(self):
        if hasattr(self, "_tuned_options"):
            return self._tuned_options.copy()
        else:
            return {}


    def add_tunable_option(self, key, val):
        "Adds a tunable option where key is the name and val is the default value."
        if key in self.tuned_options:
            assert False, "A tuned options with the given name already exist."

        self._tunable_options[key] = val


    def tune(self, key=None, **kwargs):
        """
        Tunes a tunable option.
        
        Parameters
        ----------
        key: the name of the tunable option to tune.
           If key is None then all the tunable options are tuned.
        callback: a function to call for the tuning. 
           The function will be called as (key=key, value=value, **kwargs)

        """
        if key is None:
            for key in self.tunable_options:
                self.tune(key, **kwargs)
            return
        
        elif key in self.tuned_options:
            return

        assert key in self.tunable_options, "Option %s not found" % key
        self._tuned_options[key] = self._tunable_options.pop(key)

        self._tuning = True
        try:
            callback = kwargs.get("callback", None)
            if callback is not None:
                self._tuned_options[key] = callback(key=key,value=value,**kwargs)
        except:
            self._tuning = False
            raise

        
    def __getattr__(self, key):
        if key not in Tunable.__slots__:
            if key in self.tunable_options:
                self.tune(key=key)
                
            if key in self.tuned_options:
                return self._tuned_options[key]
        else:
            raise AttributeError("Not a tunable option %s"%key)


    def __setattr__(self, key, value):
        if key not in Tunable.__slots__ and (key in self.tunable_options or key in self.tuned_options):
            if key in self.tunable_options:
                del self._tunable_options[key]
                self._tuned_options[key] = value
            else:
                assert False, "The value of a tuned option cannot be changed."
        else:
            super().__setattr__(key, value)
