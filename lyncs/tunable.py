"""
Extension of dask delayed with a tunable aware Delayed object.
"""

__all__ = [
    "delayed",
    "Delayed",
    "RaiseNotTuned",
    "Tunable",
    "tunable_property",
    "LyncsMethodsMixin",
]

from dask.delayed import delayed as dask_delayed
from dask.delayed import Delayed

def delayed(*args,**kwargs):
    kwargs.setdefault("pure", True)
    return dask_delayed(*args, **kwargs)

delayed.__doc__ = dask_delayed.__doc__


class NotTuned(Exception):
    pass

    
raise_not_tuned = False
not_tuned_error = NotTuned

class RaiseNotTuned:
    def __init__(self, *args, **kwargs):
        self.error = NotTuned(*args, **kwargs)
    
    def __enter__(self):
        global raise_not_tuned, not_tuned_error
        raise_not_tuned = True
        tmp = not_tuned_error
        not_tuned_error = self.error
        self.error = not_tuned_error
        return type(not_tuned_error)

    def __exit__(self, type, value, traceback):
        global raise_not_tuned, not_tuned_error
        raise_not_tuned = False
        tmp = not_tuned_error
        not_tuned_error = self.error
        self.error = not_tuned_error

    
class Tunable:
    """
    A base class for tunable objects.
    """
    
    __slots__ = [
        "_tunable_options",
        "_tuned_options",
        "_tuning",
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

        val = val if isinstance(val, TunableOption) else TunableOption(val)
        
        if key in self.tuned_options:
            assert val.compatible(self.tuned_options[key]), "A tuned options with the given name already exist and it is not compatible."
        else:
            self._tunable_options[key] = val


    def add_tuned_option(self, key, val):
        "Adds a tuned option where key is the name and val is the value."
        assert key not in self.tuned_options, "A tuned options with the given name already exist."
        
        if key in self.tunable_options:
            setattr(self,key,val)
        else:
            self._tuned_options[key] = val


    def add_option(self, key, val):
        "Adds a tunable/tuned option (accordingly to val) where key is the name and val is the value."
        if isinstance(val, TunableOption): self.add_tunable_options(key, val)
        else: self.add_tuned_option(key, val)

        
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
                global raise_not_tuned
                if raise_not_tuned:
                    raise not_tuned_error
                else:
                    return getattr(delayed(self),key)
                
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
            with RaiseNotTuned("In tunable property") as err:
                try:
                    return func(cls)
                except err:
                    return delayed(getter)(delayed(cls))
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
    

# In the following we monkey patch DaskMethodsMixin in order to make it tunable-aware

from dask.base import DaskMethodsMixin as LyncsMethodsMixin
from .utils import add_parameters_to_doc


LyncsMethodsMixin.__name__ = "LyncsMethodsMixin"


def _is_tunable(obj):
    "Tells whether an object is tunable"
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0: return False
        else: return _is_tunable(obj[0]) or _is_tunable(obj[1:])
    else:
        return isinstance(obj, Tunable) and obj.tunable


    
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

LyncsMethodsMixin.tune = tune



@property
def tunable(self):
    return _is_tunable(list(self.dask.values()))

LyncsMethodsMixin.tunable = tunable



@property
def tunable_items(self):
    return {key:val for key,val in self.dask.items() if _is_tunable(val)}.items()

LyncsMethodsMixin.tunable_items = tunable_items


# In compute, we tune and then compute

dask_compute = LyncsMethodsMixin.compute

def compute(self, *args, tune=True, tune_kwargs={}, **kwargs):
    if tune: self.tune(**tune_kwargs)
    return dask_compute(self, *args, **kwargs)

compute.__doc__ = add_parameters_to_doc(dask_compute.__doc__, """
    tune: bool
        Whether to perform tuning before computing.
    tune_kwargs: dict
        Kwargs that will be passed to the tune function.
    """)

LyncsMethodsMixin.compute = compute


# In visualize, we mark in red tunable objects

dask_visualize = LyncsMethodsMixin.visualize

def visualize(self, *args, mark_tunable="red", **kwargs):
    if mark_tunable:
        kwargs["data_attributes"] = { k: {"color": mark_tunable,
                                          "label": ", ".join(v.tunable_options.keys()),
                                          "fontcolor": mark_tunable,
                                          "fontsize": "12",
                                         }
                                      for k,v in self.tunable_items if isinstance(v,Tunable) }
    return dask_visualize(self, *args, **kwargs)

visualize.__doc__ = add_parameters_to_doc(dask_visualize.__doc__, """
    mark_tunable: color
        Marks the tunable object with the given color. Skips if None.
    """)

LyncsMethodsMixin.visualize = visualize
