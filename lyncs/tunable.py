"""
Extension of dask delayed with a tunable aware Delayed object.
"""

__all__ = [
    "add_lyncs_methods",
    "delayed",
    "Tunable"
]


from dask.base import DaskMethodsMixin
from .utils import add_parameters_to_doc

def add_lyncs_methods(obj, **attrs):
    """
    Replaces the DaskMethodsMixin with LyncsMethodsMixin
    """
    if isinstance(obj, DaskMethodsMixin) and not isinstance(obj, LyncsMethodsMixin):
        from inspect import ismethod
        
        def wrap_method(method):
            def wrapped(*args, **kwargs):
                return add_lyncs_methods(method(*args, **kwargs), **attrs)
            wrapped.__name__ = method.__name__
            wrapped.__doc__ = method.__doc__
            return wrapped

        obj_attrs = {}
        for attr in dir(type(obj)):
            if attr not in dir(LyncsMethodsMixin):
                if hasattr(obj, attr) and ismethod(getattr(obj, attr)):
                    obj_attrs[attr] = wrap_method(getattr(type(obj), attr))
                    
        obj_attrs.update(attrs)
        object.__setattr__(obj, "__class__",
                           type(type(obj).__name__, (LyncsMethodsMixin, type(obj)), obj_attrs))

    return obj
    

    
def delayed(*args, **kwargs):
    """
    Equivalent to dask.delayed, but returns a tunable-aware Delayed object. 
    For help see dask.delayed.
    """
    from dask import delayed as dask_delayed
    return add_lyncs_methods(dask_delayed(*args, pure=True, **kwargs))


def is_tunable(obj):
    "Tells whether an object is tunable"
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0: return False
        else: return is_tunable(obj[0]) or is_tunable(obj[1:])
    else:
        return isinstance(obj, Tunable) and obj.tunable

        
class LyncsMethodsMixin(DaskMethodsMixin):
    """
    LyncsMethodsMixin makes DaskMethodsMixin aware of tunable objects.
    DaskMethodsMixin implements the compute, optimize, visualize and persist function.
    Here we add also a tune function and we call it appropriately in the others.
    """
    
    def tune(self, **kwargs):
        if not self.tunable: return
        
        from dask.delayed import DelayedAttr
        if isinstance(self, DelayedAttr) and \
           isinstance(self.dask[self._obj.key], Tunable) and \
           self._attr in self.dask[self._obj.key].tunable_options:
            # Special case for tunable options
            return self.dask[self._obj.key].tune(key=self._attr)

        for key,val in self.tunable_items:
            val.tune(**kwargs)
        return self


    @property
    def tunable(self):
        return is_tunable(list(self.dask.values()))

    @property
    def tunable_items(self):
        return {key:val for key,val in self.dask.items() if is_tunable(val)}.items()

    
    def compute(self, *args, tune=True, tune_kwargs={}, **kwargs):
        if tune: self.tune(**tune_kwargs)
        return super().compute(*args, **kwargs)

    compute.__doc__ = add_parameters_to_doc(DaskMethodsMixin.compute.__doc__, """
        tune: bool
            Whether to perform tuning before computing.
        tune_kwargs: dict
            Kwargs that will be passed to the tune function.
        """)

    
    def visualize(self, *args, mark_tunable="red", **kwargs):
        if mark_tunable:
            kwargs["data_attributes"] = { k: {"color": mark_tunable,
                                              "label": ", ".join(v.tunable_options.keys()),
                                              "fontcolor": mark_tunable,
                                              "fontsize": "12",
                                             }
                                          for k,v in self.tunable_items if isinstance(v,Tunable) }
        return super().visualize(*args, **kwargs)

    visualize.__doc__ = add_parameters_to_doc(DaskMethodsMixin.visualize.__doc__, """
        mark_tunable: color
            Marks the tunable object with the given color. Skips if None.
        """)
    
    
    def __repr__(self):
        ret = super().__repr__()
        if self.tunable: ret = "Tunable"+ret
        return ret


class NotTuned(Exception):
    pass

    
class Tunable:
    """
    A base class for tunable objects.
    """
    
    __slots__ = [
        "_tunable_options",
        "_tuned_options",
        "_tuning",
        "_raise_not_tuned",
    ]
    
    def __init__(
            self,
            tunable_options = {},
            tuned_options = {},
            **kwargs
    ):
        self._tunable_options = {}
        self._tuned_options = {}
        self._tuning = False
        
        for key,val in tunable_options.items():
            self.add_tunable_option(key,val)

        for key,val in kwargs.items():
            if isinstance(val,TunableOption):
                self.add_tunable_option(key,val)
            else:
                self.add_tuned_option(key,val)
                
        for key,val in tuned_options.items():
            self.add_tuned_option(key,val)

            
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
        assert key not in self.tunable_options, "A tunable options with the given name already exist."
        assert key not in self.tuned_options, "A tuned options with the given name already exist."
            
        self._tunable_options[key] = val if isinstance(val, TunableOption) else TunableOption(val)


    def add_tuned_option(self, key, val):
        "Adds a tunde option where key is the name and val is the value."
        assert key not in self.tuned_options, "A tuned options with the given name already exist."
        
        if key in self.tunable_options:
            setattr(self,key,val)
        else:
            self._tuned_options[key] = val

        

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
            return self.tuned_options[key]

        assert key in self.tunable_options, "Option %s not found" % key

        self._tuning = True
        try:
            callback = kwargs.get("callback", None)
            if callback is not None:
                setattr(self, key, callback(key=key,value=value,**kwargs))
            else:
                setattr(self, key, self.tunable_options[key].get())
            
        except:
            self._tuning = False
            raise
        return self.tuned_options[key]

        
    def __getattr__(self, key):
        if key not in Tunable.__slots__ and (key in self.tunable_options or key in self.tuned_options):
            if key in self.tunable_options:
                if self._raise_not_tuned:
                    raise NotTuned
                else:
                    return delayed(self).__getattr__(key)
                
            if key in self.tuned_options:
                return self._tuned_options[key]
        else:
            raise AttributeError("Not a tunable option %s"%key)


    def __setattr__(self, key, value):
        if key not in Tunable.__slots__ and (key in self.tunable_options or key in self.tuned_options):
            if key in self.tunable_options:
                assert self.tunable_options[key].compatible(value), """
                Value not compatible with %s""" % self.tunable_options[key]
                del self._tunable_options[key]
                self._tuned_options[key] = value
            else:
                assert False, "The value of a tuned option cannot be changed."
        else:
            super().__setattr__(key, value)

            
    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)
        


class tunable_property(property):
    def __init__(self,func):
        def getter(cls):
            cls._raise_not_tuned = True
            try:
                return func(cls)
            except NotTuned:
                return delayed(getter)(delayed(cls))
            finally:
                cls._raise_not_tuned = False
        getter.__name__=func.__name__
        super().__init__(getter)
            

class TunableOption:
    "Base class for tunable options"
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self.get()
            
    def get(self):
        return self._value

    def compatible(self, value):
        return value == self.get()

    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)


class Permutation(TunableOption):
    "A permutation of the given list/tuple"
    def __init__(self, value):
        assert isinstance(value, (tuple,list)), "A permutation must be initialized with a list/tuple"
        super().__init__(value)
    
    def compatible(self, value):
        from collections import Counter
        return len(self.get()) == len(value) and Counter(self.get()) == Counter(value)


class Choice(TunableOption):
    "One element of list/tuple"
    def __init__(self, value):
        assert isinstance(value, (tuple,list)), "A choice must be initialized with a list/tuple"
        assert len(value) > 0, "List cannot be empty"
        super().__init__(value)
    
    def compatible(self, value):
        return value in self.get()

    def get(self):
        return self._value[0]

class ChunksOf(TunableOption):
    "Chunks of a given shape"
    def __init__(self, value):
        if isinstance(value, (tuple,list)):
            assert all(isinstance(v, tuple) and len(v)==2 for v in value)
            shape = {key:val for key,val in value}
        elif isinstance(value, dict):
            shape = value
        super().__init__(shape)
    
    def compatible(self, value):
        chunks = ChunksOf(value)
        shape = self.get()
        # Here we ask for uniform distribution. Consider to allow for not uniform
        return all(key in shape and val<=shape[key] and shape[key]%val == 0 for key,val in chunks.get().items())
    
