"""
Extension of dask delayed with a tunable aware Delayed object.
"""

__all__ = [
    "delayed",
    "Delayed",
    "RaiseNotTuned",
    "Tunable",
    "computable",
    "LyncsMethodsMixin",
]

from dask.delayed import delayed as dask_delayed
from dask.delayed import Delayed

def delayed(*args,**kwargs):
    kwargs.setdefault("pure", True)
    return dask_delayed(*args, **kwargs)

delayed.__doc__ = dask_delayed.__doc__


class Tunable:
    """
    A base class for tunable objects.
    """
    
    __slots__ = [
        "_options",
    ]
            
    @property
    def tunable(self):
        return any((val.tunable for val in self.tunable_options.values()))

    
    @property
    def options(self):
        if hasattr(self, "_options"): return self._options.copy()
        else: return {}

        
    @property
    def tunable_options(self):
        return {key:val for key,val in self.options.items() if val.tunable}

    
    @property
    def fixed_options(self):
        return {key:val.value for key,val in self.options.items() if not val.tunable}


    def add_option(self, key, opt, transformer = None):
        "Adds a tunable option where key is the name and opt is a TunableOption"
        assert key not in self.tunable_options, "A tunable options with the given name already exist."
        assert isinstance(opt, TunableOption), "The options must be a TunableOptions."

        from collections import OrderedDict
        if not hasattr(self, "_options"): self._options = OrderedDict()
        opt.__name__ = key
        opt._transformer = transformer
        self._options[key] = opt

                
    def transform(self, key, obj, old_value):
        assert key in self.options, "Unknown option %s" % key
        if self.options[key]._transformer is not None:
            option = self.options[key]
            if isinstance(old_value, TunableOption):
                old_value = old_value.value
            old_value = computable(option.format)(old_value)
            return option._transformer(key, obj, option.value, old_value)
        else:
            return obj

        
    def tune(self, key=None, **kwargs):
        """
        Tunes one or all tunable options.
        
        Parameters
        ----------
        key: str 
            The name of the tunable option to tune.
            If key is None then all the tunable options are tuned.
        kwargs: dict
            The list of parameters to pass to the tune function
        """
        if key is None:
            return tune(self, **kwargs)
        
        elif key in self.fixed_options:
            return self.fixed_options[key]
        
        else:
            assert key in self.tunable_options, "Option %s not found" % key
            
            if hasattr(self, "dask"):
                kwargs.setdefault("graph", getattr(self,"dask"))
                
            return tune(getattr(self,key), **kwargs)

        
    def __getattr__(self, key):
        if key not in Tunable.__slots__ and key in self.options:
            val = self.options[key]
            val.__name__ = key
            return val.value
        else:
            raise AttributeError("Not a tunable option %s"%key)


    def __setattr__(self, key, value):
        if key not in Tunable.__slots__ and key in self.options:
            self.options[key].value = value
        else:
            super().__setattr__(key, value)

            
    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)
        
def wrap_array(array):
    class wrapped:
        def __init__(self, array):
            self.array = array
            self.__name__ = array.name
        def __call__(self):
            return self.array
    return delayed(wrapped(array))()


def computable(fnc):
    from functools import wraps
    @wraps(fnc)
    def _fnc(*args, **kwargs):
        if any((isinstance(v, Delayed) for v in args)) or \
           any((isinstance(v, Delayed) for v in kwargs.values())):
            
            from dask.array import Array
            
            args = list(args)
            for i,val in enumerate(args):
                if isinstance(val, Array):
                    args[i] = wrap_array(val)
            
            return delayed(_fnc)(*args, **kwargs)
        else:
            return fnc(*args, **kwargs)
        
    _fnc.__name__=fnc.__name__
    return _fnc
    

class TunableOption:
    "Base class for tunable options"
    def __init__(self, source, length=None):
        self._source = source
        self._length = length
    
        
    @property
    def value(self):
        from copy import copy
        if self.tunable:
            value = delayed(self)()
            value._length = self._length
            return value
        elif isinstance(self._value, Delayed):
            self._value = self._value.compute(tune=False)
        return copy(self._value)
    
        
    @value.setter
    def value(self, value):
        self.set(value)

    def set(self, value, force=False):
        if not force:
            assert not hasattr(self,"_value") or self._value == format(value), """
            The value of a fixed option cannot be changed.
            """
        if type(value) == type(self) and self.source == value.source:
            self._value = value.value
            
        else:
            if isinstance(value, TunableOption):
                value = value.value
                
            self._value = computable(self.format)(value)


    def __call__(self, value=None):
        if value is None:
            return self.value
        else:
            self.value = value


    @property
    def source(self):
        return self._source
    
    
    @property
    def tunable(self):
        return not hasattr(self,"_value")
    
    
    def __repr__(self):
        from .utils import default_repr
        return default_repr(self)

    
    def __eq__(self, value):
        return self.value == format(value)

    
    # Functions you may want to overload
    def format(self, value):
        assert value in self.source, """
        Given value not in list.
        %s not in %s
        """ (value, self.source)
        return value

        
    def __iter__(self):
        return iter(self._source)

    
    def __len__(self):
        return len(self._source)

    
    def __dask_tokenize__(self):
        import uuid
        from dask.base import normalize_token
        if not hasattr(self, "_unique_id"): self._unique_id = str(uuid.uuid4())
            
        return normalize_token(
            (type(self), self.source, self._unique_id)
        )
    

class Choice(TunableOption):
    "One element of iterable"
    def __init__(self, source):
        assert hasattr(source, "__iter__"), "A choice must be iterable"
        source = list(source)
        lengths = (len(value) if hasattr(value, "__len__") else None  for value in source)
        length = next(lengths)
        if any((_len != length for _len in lengths)): length = None
        super().__init__(source, length)

    

class Permutation(TunableOption):
    "A permutation of iterable"
    def __init__(self, source):
        assert hasattr(source, "__iter__"), "A permutation must be iterable"
        source = list(source)
        super().__init__(source, length = len(source))

        
    def format(self, value):
        from collections import Counter
        value = list(value)
        assert all((value.count(key)==val for key,val in Counter(self.source).items())), """
            Permutation: %s
            Given value: %s
            The given value is not compatible with the permutation.
            Are compatible all the lists that include the permutation 
            and the counts of the element of the permutation are not higher.
            """ % (self.source, value)
        source = list(self.source)
        ret = []
        for v in value:
            if v in source:
                source.remove(v)
                ret.append(v)
        return ret
    
        
    def __iter__(self):
        from itertools import permutations
        return permutations(self.source)

    
    def __len__(self):
        from math import factorial
        return factorial(self._length)

    

class ChunksOf(TunableOption):
    "Chunks of a given shape"
    def __init__(self, source):
        self._type = type(source)
        
        if isinstance(source, (tuple,list)):
            assert all(isinstance(v, tuple) and len(v)==2 and isinstance(v[1],int) for v in source), """
            Chunks supports list of pairs, where the second element of each pair must be a integer,
            e.g. [("a", 2), ("b", 4), ...]
            """
        elif isinstance(source, dict):
            assert all(isinstance(v, int) for v in source.values()),"""
            Each element of the dictionary must be a integer.
            """
            source = list(source.items())
            
        super().__init__(source, length=len(source))
        
        from math import ceil, sqrt
        self._factors = [[ val//n for n in range(1, int(sqrt(val))+1) if val%n==0 ] + [1] for key,val in self.source]

        
    def format(self, value):
        if not self.source and not value:
            return value
        if isinstance(value, dict):
            value = list(value.items())
        keys,vals = zip(*value)
        keys = list(keys)
        vals = list(vals)
        chunks = []
        for key,val in self.source:
            if key not in keys:
                chunks.append((key,val))
            else:
                idx = keys.index(key)
                assert val % vals[idx] == 0, """
                Not compatible chunk: %s=%s does not divide exactly %s.
                """ %(key,val,vals[idx])
                chunks.append((key,vals[idx]))
                keys.pop(idx)
                vals.pop(idx)
        return self._type(chunks)

    
    def __iter__(self):
        from itertools import product
        for chunk in product(*self._factors):
            yield self._type([(key,ch) for (key,val),ch in zip(self.source, chunk)])

            
    def __len__(self):
        from numpy import prod
        return prod([len(factors) for factors in self._factors])

    
# In the following we monkey patch DaskMethodsMixin in order to make it tunable-aware

from dask.base import DaskMethodsMixin as LyncsMethodsMixin
from .utils import add_parameters_to_doc


LyncsMethodsMixin.__name__ = "LyncsMethodsMixin"


def _is_tunable(obj):
    "Tells whether an object is tunable"
    if isinstance(obj, (list, tuple)):
        return any((_is_tunable(i) for i in obj))
    else:
        return isinstance(obj, TunableOption) and obj.tunable


    
def tune(self, **kwargs):
    kwargs.setdefault("graph", self.dask if hasattr(self, "dask") else None)

    if len(self.tunable_options)==0:
        return self
    
    elif len(self.tunable_options)>1:
        for val in self.tunable_options.values():
            tune(val.value, **kwargs)
        return self

    tunable_option = next(iter(self.tunable_options.values()))
    
    #TODO
    
    tunable_option.value = next(iter(tunable_option))
    return tunable_option.value

LyncsMethodsMixin.tune = tune



@property
def tunable(self):
    return _is_tunable(list(self.dask.values()))

LyncsMethodsMixin.tunable = tunable



@property
def tunable_options(self):
    return {key:val[0] for key,val in self.dask.items() if _is_tunable(val)}

LyncsMethodsMixin.tunable_options = tunable_options


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


# In perist, we tune and then persist

dask_persist = LyncsMethodsMixin.persist

def persist(self, *args, tune=True, tune_kwargs={}, **kwargs):
    if tune: self.tune(**tune_kwargs)
    return dask_persist(self, *args, **kwargs)

persist.__doc__ = add_parameters_to_doc(dask_persist.__doc__, """
        tune: bool
            Whether to perform tuning before computing.
        tune_kwargs: dict
            Kwargs that will be passed to the tune function.
    """)

LyncsMethodsMixin.persist = persist


# In visualize, we mark in red tunable objects

dask_visualize = LyncsMethodsMixin.visualize

def visualize(self, *args, mark_tunable="red", **kwargs):
    kwargs.setdefault("rankdir","LR")
    kwargs.setdefault("labelfontsize","12")
    if mark_tunable:
        kwargs["function_attributes"] = { k: {"color": mark_tunable,
                                              "fontcolor": mark_tunable,
                                              "fontsize": "14",
                                              "shape":"diamond",
                                              }
                                          for k in self.tunable_options.keys() }
    return dask_visualize(self, *args, **kwargs)

visualize.__doc__ = add_parameters_to_doc(dask_visualize.__doc__, """
        mark_tunable: color
            Marks the tunable object with the given color. Skips if None.
    """)

LyncsMethodsMixin.visualize = visualize
