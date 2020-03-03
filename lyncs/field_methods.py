"""
Dynamic interface to dask array methods
"""

from importlib import import_module
from functools import wraps
from dask import array as dask
import numpy


__all__ = [
    "FieldMethods",
]


class FieldMethods:
    def __pos__(self):
        return self
    
    
    def __matmul__(self, other):
        return self.dot(other)
    
    
    def __rmatmul__(self, other):
        return other.dot(self)
    
    
    def reshape(self, *axes_order):
        """
        Changes the axes_order of the field.
        
        *NOTE*: this is conceptually different from numpy.reshape.
        """
        from .field import Field
        return Field(self, axes_order=axes_order)
    
    
    def rechunk(self, **chunks):
        """
        Changes the chunks of the field.
        """
        from .field import Field
        return Field(self, chunks=chunks)
    
    
    def squeeze(self, axis=None):
        """
        Removes axes with size one. (E.g. axes where a coordinate has been selected)
        """
        if axis is None:
            new_axes = [axis for axis,size in self.shape if size>1]
        else:
            shape = {axis:size for axis,size in self.shape}
            assert axis in shape
            assert shape[axis]==1
            new_axes = [ax for ax,size in self.shape if ax!=axis]
            
        return Field(self, field_type=new_axes)
    
    
    def __getitem__(self, coords):
        from .field import Field
        return Field(field=self, coords=coords)

    
    def get(self, *labels, **coords):
        from .field import Field
        coords = list(labels) + [coords]
        return Field(field=self, coords=coords)


    def __setitem__(self, coords, value):
        if isinstance(coords, tuple):
            return self.set(value, *coords)
        else:
            return self.set(value, coords)

    
    def set(self, value, *labels, **coords):
        return Field(field=self, coords=coords)    

        
    @property
    def real(self):
        return real(self)

    
    @property
    def imag(self):
        return imag(self)

    
    @property
    def T(self):
        "Transposes the field."
        return self.transpose()
    
    
    def transpose(self, *axes, **axes_order):
        """
        Transposes the matrix/tensor indeces, i.e. the dimensions that are repeated.
        
        *NOTE*: this is conceptually different from numpy.transpose.
        
        Parameters
        ----------
        axes: str
            If given, only the listed axes are transposed, otherwise all the tensorial axes are changed.
            By default the order of the indeces is inverted.
        axes_order: dict
            Same as axes, but specifying the reordering of the indeces. The keys must be "*axis*_order".
        """
        
        axes = list(axes)
        if not axes and not axes_order:
            axes = self.axes
        elif axes_order:
            for key in axes_order:
                assert key.endswith("_order"), "Keys in axes_order must end with \"_order\""
                axes.append(key[:-6])

        new_order = {}
        for key in set(axes):
            if key not in self.axes:
                assert key in self.dimensions, "Unknown axes %s" % key
                keys = self._expand(key)
            else:
                keys = [key]

            for key2 in keys:
                count = self.axes.count(key2)
                if key+"_order" in axes_order:
                    from .tunable import Permutation
                    assert Permutation(list(range(count))).compatible(axes_order[key+"_order"]), """
                    The axes_order of %s must be a permutation of %s.
                    """ %(key, list(range(count)))

                    if count > 1:
                        new_order[key2+"_order"] = axes_order[key+"_order"]
                elif count > 1:
                    new_order[key2+"_order"] = reversed(getattr(self, key2+"_order"))

        if new_order:
            return Field(self, **new_order)
        else:
            return self

    @property
    def H(self):
        "Conjugate transpose of the field."
        return self.dagger()
    
    
    def dagger(self, *axes, **axes_order):
        """
        Conjugate and transposes the matrix/tensor indeces, i.e. the dimensions that are repeated.
        
        Parameters
        ----------
        axes: str
            If given, only the listed axes are transposed, otherwise all the tensorial axes are changed.
            By default the order of the indeces is inverted.
        axes_order: dict
            Same as axes, but specifying the reordering of the indeces. The keys must be "*axis*_order".
        """
        return self.transpose(*axes, **axes_order).conj()

    
    def trace(self, *axes):
        """
        Performs the trace over the matrix axes.
        
        Parameters
        ----------
        axes: str
            If given, only the listed axes are traced.
        """
        axes = list(axes)
        if not axes:
            for axis in set(self.axes):
                if self.axes.count(axis) == 2:
                    axes.append(axis)
        else:
            # TODO should work also for non-fundamental axes
            for	axis in	axes:
                assert self.axes.count(axis) == 2, "Only matrix indeces can be traced"
                
        new_axes = self.axes
        for axis in axes:
            while axis in new_axes:
                new_axes.remove(axis)
            
        new_field = Field(self, field_type = new_axes)
        
        if len(axes) == 1:
            new_field.field = self.field #TODO
        else:
            #TODO
            pass

        return new_field
    
    
    def dot(self, *axes):
        pass
        


    
# The following are simple universal functions and they are dynamically wrapped

ufuncs = (
    # math operations
    ("add", True, ),
    ("subtract", True, ),
    ("multiply", True, ),
    ("divide", True, ),
    ("logaddexp", False, ),
    ("logaddexp2", False, ),
    ("true_divide", True, ),
    ("floor_divide", True, ),
    ("negative", True, ),
    ("power", True, ),
    ("float_power", True, ),
    ("remainder", True, ),
    ("mod", True, ),
    ("fmod", True, ),
    ("conj", True, ),
    ("exp", False, ),
    ("exp2", False, ),
    ("log", False, ),
    ("log2", False, ),
    ("log10", False, ),
    ("log1p", False, ),
    ("expm1", False, ),
    ("sqrt", True, ),
    ("square", True, ),
    ("cbrt", False, ),
    ("reciprocal", True, ),
    # trigonometric functions
    ("sin", False, ),
    ("cos", False, ),
    ("tan", False, ),
    ("arcsin", False, ),
    ("arccos", False, ),
    ("arctan", False, ),
    ("arctan2", False, ),
    ("hypot", False, ),
    ("sinh", False, ),
    ("cosh", False, ),
    ("tanh", False, ),
    ("arcsinh", False, ),
    ("arccosh", False, ),
    ("arctanh", False, ),
    ("deg2rad", False, ),
    ("rad2deg", False, ),
    # comparison functions
    ("greater", True, ),
    ("greater_equal", True, ),
    ("less", True, ),
    ("less_equal", True, ),
    ("not_equal", True, ),
    ("equal", True, ),
    ("isneginf", False, ),
    ("isposinf", False, ),
    ("logical_and", False, ),
    ("logical_or", False, ),
    ("logical_xor", False, ),
    ("logical_not", False, ),
    ("maximum", False, ),
    ("minimum", False, ),
    ("fmax", False, ),
    ("fmin", False, ),
    # floating functions
    ("isfinite", True, ),
    ("isinf", True, ),
    ("isnan", True, ),
    ("signbit", False, ),
    ("copysign", False, ),
    ("nextafter", False, ),
    ("spacing", False, ),
    ("modf", False, ),
    ("ldexp", False, ),
    ("frexp", False, ),
    ("fmod", False, ),
    ("floor", True, ),
    ("ceil", True, ),
    ("trunc", False, ),
    ("round", True, ),
    # more math routines
    ("degrees", False, ),
    ("radians", False, ),
    ("rint", True, ),
    ("fabs", True, ),
    ("sign", True, ),
    ("absolute", True, ),
    # non-ufunc elementwise functions
    ("clip", True, ),
    ("isreal", True, ),
    ("iscomplex", True, ),
    ("real", False, ),
    ("imag", False, ),
    ("fix", False, ),
    ("i0", False, ),
    ("sinc", False, ),
    ("nan_to_num", True, ),
)


operators = (
    ("__abs__", ),
    ("__add__", ),
    ("__radd__", ),
    ("__eq__", ),
    ("__gt__", ),
    ("__ge__", ),
    ("__lt__", ),
    ("__le__", ),
    ("__mod__", ),
    ("__rmod__", ),
    ("__mul__", ),
    ("__rmul__", ),
    ("__ne__", ),
    ("__neg__", ),
    ("__pow__", ),
    ("__rpow__", ),
    ("__sub__", ),
    ("__rsub__", ),
    ("__truediv__", ),
    ("__rtruediv__", ),
    ("__floordiv__", ),
    ("__rfloordiv__", ),
    ("__divmod__", ),
    ("__rdivmod__", ),
)

reductions = (
    ("any", ),
    ("all", ),
    ("min", ),
    ("max", ),
    ("argmin", ),
    ("argmax", ),
    ("sum", ),
    ("prod", ),
    ("mean", ),
    ("std", ),
    ("var", ),
)


def prepare(*fields):
    from .field import Field
    from builtins import all
    assert all([isinstance(field, Field) for field in fields])
    if len(fields)==1:
        return fields, Field(fields[0])
    
    # TODO
    return fields, Field(fields[0])


def wrap_ufunc(ufunc):
    @wraps(ufunc)
    def wrapped(*args, **kwargs):
        from .field import Field
        from .tunable import Delayed, tunable_function
        from dask import array
        assert len(args)>0 and isinstance(args[0], Field), "First argument must be a field type"
        args = list(args)
        # Deducing the number of outputs and the output dtype
        tmp_args = [array.ones((1), dtype=arg.dtype) if isinstance(arg, Field) else arg for arg in args]
        trial = ufunc(*tmp_args, **kwargs)

        # Uniforming the fields involved
        fields = [(i,arg) for i,arg in enumerate(args) if isinstance(arg, Field)]
        new_fields, out_field = prepare(*[field for i,field in fields])
        for (i,old), new in zip(fields, new_fields):
            assert args[i] is old, "Trivial assertion" % (old, args[i])
            args[i] = new.field
        
        # Calling ufunc
        if isinstance(trial, tuple):
            fields = tuple(Field(out_field, dtype=val.dtype) for val in trial)
            res = tunable_function(ufunc)(*args, **kwargs)
            if isinstance(res, Delayed): res._length = len(fields)
            for i,field in enumerate(fields):
                field.field = res[i] 
            return fields
        else:
            out_field.dtype = trial.dtype
            out_field.field = tunable_function(ufunc)(*args, **kwargs)
            return out_field
    return wrapped


def wrap_reduction(reduction):
    @wraps(reduction)
    def wrapped(field, *axes, **kwargs):
        from .field import Field
        from .tunable import tunable_function
        from dask import array
        
        assert isinstance(field, Field), "First argument must be a field type"


        # Extracting the axes to reduce
        axes=list(axes)
        tmp=kwargs.pop("axis",[])
        axes.extend([tmp] if isinstance(tmp,str) else tmp)
        tmp=kwargs.pop("axes",[])
        axes.extend([tmp] if isinstance(tmp,str) else tmp)

        dtype = reduction(array.ones((1,), dtype=field.dtype), **kwargs).dtype
        if axes:
            assert set(axes).issubset(field.dimensions)
            axes = field._expand(axes)
            new_axes = field.axes
            for axis in set(axes):
                while axis in new_axes:
                    new_axes.remove(axis)

            @tunable_function
            def get_axes(old_axes):
                axes = list(range(len(old_axes)))
                for axis in new_axes:
                    axes.pop(old_axes.index(axis))
                    old_axes.remove(axis)
                return tuple(axes)
                    
            kwargs["axis"] = get_axes(field.axes_order)
            out = Field(field, field_type=new_axes, dtype=dtype, zeros_init=True)
        else:
            out = Field(field, field_type=[], dtype=dtype, zeros_init=True)
        print(out)
        out.field = tunable_function(reduction)(field.field, **kwargs)
        return out
        
    return wrapped



from functools import partial, update_wrapper

for name, is_member in ufuncs:
    ufunc = getattr(dask,name)
    if isinstance(ufunc, partial): update_wrapper(ufunc, ufunc.func)
    globals()[name] = wrap_ufunc(ufunc)
    __all__.append(name)
    if is_member:
        setattr(FieldMethods, name, globals()[name])


for name, in operators:
    ufunc = getattr(dask.Array,name)
    if isinstance(ufunc, partial): update_wrapper(ufunc, ufunc.func)
    setattr(FieldMethods, name, wrap_ufunc(ufunc))

for name, in reductions:
    ufunc = getattr(dask,name)
    if isinstance(ufunc, partial): update_wrapper(ufunc, ufunc.func)
    globals()[name] = wrap_reduction(ufunc) 
    ufunc = getattr(dask.Array,name)
    if isinstance(ufunc, partial): update_wrapper(ufunc, ufunc.func)
    setattr(FieldMethods, name, wrap_reduction(ufunc))
