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

    
    def astype(self, dtype):
        """
        Changes the dtype of the field.
        """
        from .field import Field
        return Field(self, dtype=dtype)
    
    
    def reshape(self, *args, **kwargs):
        """
        *WARNING*: reshape not implemented.

        Does it make sense to reshape a field?
        If yes, please open a detailed issue on github.com/sbacchio/lyncs
        and it will be discussed. Thanks! <sbacchio>
        """
        assert False, FieldMethods.reshape.__doc__

        
    def _reorder(self, key, field, new_axes_order, old_axes_order):
        "Transformation for changing axes_order"
        from .tunable import computable
        assert key == "axes_order", "Got wrong key! %s" % key
        
        @computable
        def reorder(field, new_axes_order, old_axes_order):
            from collections import Counter
            assert Counter(new_axes_order) == Counter(old_axes_order), """
            Got not compatible new_ and old_axes_order:
            new_axes_order = %s
            old_axes_order = %s
            """ % (new_axes_order, old_axes_order)
            old_indeces = list(range(len(old_axes_order)))
            axes = []
            for key in new_axes_order:
                idx = old_indeces[old_axes_order.index(key)]
                axes.append(idx)
                old_axes_order.remove(key)
                old_indeces.remove(idx)
                
            return field.transpose(*axes)
        
        return reorder(field, new_axes_order, old_axes_order)

        
    def reorder(self, *axes_order):
        """
        Changes the axes_order of the field.
        """
        from .field import Field
        return Field(self, axes_order=axes_order)
    
    
    def _rechunk(self, key, field, new_chunks, old_chunks):
        "Transformation for changing chunks"
        from .tunable import computable
        assert key == "chunks", "Got wrong key! %s" % key
        
        @computable
        def rechunk(field, field_chunks):
            return field.rechunk(field_chunks)
        
        return rechunk(field, self.field_chunks)

        
    def rechunk(self, **chunks):
        """
        Changes the chunks of the field.
        """
        from .field import Field
        return Field(self, chunks=chunks)
    
    
    def _squeeze(self, field, new_axes, old_axes_order, old_field_shape):
        "Transformation for squeezing axes"
        from .tunable import computable
        
        @computable
        def squeeze(field, new_axes, old_axes_order, old_field_shape):
            from collections import Counter
            axes = []
            for i, (axis, size) in enumerate(zip(list(old_axes_order), old_field_shape)):
                if axis in new_axes and size>1:
                    continue
                elif axis in new_axes and new_axes.count(axis) == old_axes_order.count(axis):
                    continue
            
                assert size==1, "Trying to squeeze axis (%s) with size (%s) larger than one" % (axis,size)
                old_axes_order.remove(axis)
                axes.append(i)
            assert Counter(new_axes) == Counter(old_axes_order), "This should not happen"
            return field.squeeze(axis=tuple(axes))
        
        return squeeze(field, new_axes, old_axes_order, old_field_shape)
            
    
    def squeeze(self, axis=None):
        """
        Removes axes with size one. (E.g. axes where a coordinate has been selected)
        """
        from .field import Field
        if axis is None:
            new_axes = [axis for axis,size in self.shape if size>1]
        else:
            axes = self._expand(axis)
            new_axes = [axis for axis,size in self.shape if size>1 or axis not in axes]
            
        return Field(self, field_type=new_axes)


    def _getitem(self, field, new_coords, old_coords, old_axes_order, setitem=None):
        "Transformation for getitem"
        from .tunable import computable
        coords = {key:val for key,val in new_coords.items() if key not in old_coords or val is not old_coords[key]}
        if not coords: return field
        
        @computable
        def getitem(field, setitem, axes_order, **coords):
            mask = [slice(None) for i in self.axes]
            for key,val in coords.items():
                mask[axes_order.index(key)] = val
                
            if setitem is not None:
                field[tuple(mask)] = setitem
                return field
            
            return field[tuple(mask)]

        if setitem is not None: getitem.__name__ = "setitem"
        return getitem(field, setitem, old_axes_order, **coords)
    
    
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
        from .field import Field
        coords = list(labels) + [coords]
        tmp = Field(field=self, coords=coords)
        tmp.field = value
        self.field = self._getitem(self.field, coords, self.coords, self.axes_order, setitem = tmp.field)

        
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
    
    
    def _transpose(self, key, field, new_order, old_order):
        "Transformation for transposing"
        from .tunable import computable
        assert key.endswith("_order") and key[:-6] in self.axes, "Got wrong key! %s" % key
        
        @computable
        def transpose(field, axis, axes_order, new_order, old_order):
            assert axes_order.count(axis) == len(new_order) and len(new_order) == len(old_order) and \
                len(new_order) == len(set(new_order)) and set(new_order) == set(old_order), """
            Got wrong parameters for performing transpose.
                axis: %s
                axes_order: %s
                new_order: %s
                old_order: %s
            """ % (axis, axes_order, new_order, old_order)
            old_axes_order = list(axes_order)
            new_axes_order = list(axes_order)
            for new, old in zip(new_order,old_order):
                idx = old_axes_order.index(axis)
                old_axes_order[idx] = axis+str(old)
                new_axes_order[idx] = axis+str(new)
                
            axes = []
            indeces = list(range(len(axes_order)))
            for axis in new_axes_order:
                idx = indeces[old_axes_order.index(axis)]
                axes.append(idx)
                old_axes_order.remove(axis)
                indeces.remove(idx)
                
            return field.transpose(*axes)
        
        return transpose(field, key[:-6], self.axes_order, new_order, old_order)
        
        
        
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
        from .field import Field
        from .tunable import computable
        
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
                    new_order[key2+"_order"] = computable(reversed)(getattr(self, key2+"_order"))

        if new_order:
            ret = Field(self, **new_order)
            # restoring previous order
            for key in new_order:
                ret.options[key].set(self.options[key], force=True)
            return ret
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
        return self.conj().transpose(*axes, **axes_order)

    
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
            # should it work also for non-fundamental axes?
            for	axis in	axes:
                assert self.axes.count(axis) == 2, "Only matrix indeces can be traced"
                
        new_axes = self.axes
        for axis in axes:
            while axis in new_axes:
                new_axes.remove(axis)
            
        new_field = Field(self, field_type=new_axes, zeros_init=True)
        
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


def prepare(*fields, **kwargs):
    from .field import Field
    from builtins import all
    assert all([isinstance(field, Field) for field in fields])
    if len(fields)==1:
        return fields, Field(fields[0], **kwargs)
    
    # TODO
    return fields, Field(fields[0], **kwargs)


def wrap_ufunc(ufunc):
    @wraps(ufunc)
    def wrapped(*args, **kwargs):
        from .field import Field
        from .tunable import Delayed, computable
        from dask import array
        assert len(args)>0 and isinstance(args[0], Field), "First argument must be a field type"
        args = list(args)
        # Deducing the number of outputs and the output dtype
        tmp_args = (array.ones((1), dtype=arg.dtype) if isinstance(arg, Field) else arg for arg in args)
        trial = ufunc(*tmp_args, **kwargs)

        if isinstance(trial, tuple):
            dtype = trial[0].dtype
        else:
            dtype = trial.dtype
            
        # Uniforming the fields involved
        fields = (arg for arg in args if isinstance(arg, Field))
        idxs = (i for i,arg in enumerate(args) if isinstance(arg, Field))
        new_fields, out_field = prepare(*fields, dtype=dtype)
        
        for i,new in zip(idxs, new_fields):
            args[i] = new.field
            
        # Calling ufunc
        if isinstance(trial, tuple):
            fields = (out_field,) + tuple(Field(out_field, dtype=val.dtype) for val in trial[1:])
            res = computable(ufunc)(*args, **kwargs)
            if isinstance(res, Delayed): res._length = len(fields)
            for i,field in enumerate(fields):
                field.field = res[i] 
            return fields
        else:
            out_field.field = computable(ufunc)(*args, **kwargs)
            return out_field
    return wrapped


def wrap_reduction(reduction):
    @wraps(reduction)
    def wrapped(field, *axes, **kwargs):
        from .field import Field
        from .tunable import computable
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

            @computable
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
            
        out.field = computable(reduction)(field.field, **kwargs)
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
