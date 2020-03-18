"""
Dynamic interface to dask array methods
"""

from importlib import import_module
from functools import wraps
from dask import array as dask
import numpy


__all__ = [
    "FieldMethods",
    "dot",
    "einsum",
    "trace",
    "transpose",
    "dagger",
]


def dot(*fields, axes=None, close_indeces=None, open_indeces=None):
    """
    Performs the dot product between fields.
    
    Default behaviors:
    ------------------
    
    Contractions are performed between only degree of freedoms of the fields, e.g. field.dofs.
    For each field, indeces are always contracted in pairs combining the outer-most free index
    of the left with the inner-most of the right.
    
    I.e. dot(*fields) = dot(*fields, axes="dofs")

    Parameters:
    -----------
    fields: Field
        List of fields to perform dot product between.
    axes: str, list
        Axes where the contraction is performed on. 
        Indeces are contracted in pairs combining the outer-most free index
        of the left with the inner-most of the right.
    close_indeces: str, list
        Same as axes.
    open_indeces: str, list
        Opposite of close indeces, i.e. the indeces of these axes are left open.
    
    Examples:
    ---------
    dot(vector, vector, axes="color")
      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,spin]
      [X,Y,Z,T, mu , c_0 ] x [X,Y,Z,T, mu , c_0 ] -> [X,Y,Z,T, mu ]

    dot(vector, vector, close_indeces="color", open_indece="spin")
      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,spin,spin]
      [X,Y,Z,T, mu , c_0 ] x [X,Y,Z,T, nu , c_0 ] -> [X,Y,Z,T, mu , nu ]    
    """
    from builtins import all
    from .field import Field
    assert not (axes is not None and close_indeces is not None), """
    Only one between axes or close_indeces can be used. They are the same parameter."""
    assert all((isinstance(field, Field) for field in fields)), "All fields must be of type field."

    close_indeces = axes if close_indeces is None else close_indeces
    
    if close_indeces is None and open_indeces is None:
        close_indeces = "dofs"

    same_indeces = set()
    for field in fields:
        same_indeces.update(field.axes)
    
    if close_indeces is not None:
        if isinstance(close_indeces, str):
            close_indeces = [close_indeces]
        tmp = set()
        for axis in close_indeces:
            for field in fields:
                tmp.update(field._expand(axis))
            
        close_indeces = tmp
        assert close_indeces.issubset(same_indeces), "Trivial assertion."
        same_indeces = same_indeces.difference(close_indeces)
    else:
         close_indeces = set()

    if open_indeces is not None:
        if isinstance(open_indeces, str):
            open_indeces = [open_indeces]
        tmp = set()
        for axis in open_indeces:
            for field in fields:
                tmp.update(field._expand(axis))
            
        open_indeces = tmp
        assert open_indeces.issubset(same_indeces), "Close and open indeces cannot have axes in common."
        same_indeces = same_indeces.difference(open_indeces)
    else:
         open_indeces = set()

    _i=0
    field_indeces = []
    new_field_indeces = {}
    for field in fields:
        field_indeces.append({})
        for key, count in field.axes_counts.items():
            
            if key in same_indeces:
                if key not in new_field_indeces:
                    new_field_indeces[key] = tuple(_i+i for i in range(count))
                    _i+=count
                else:
                    assert len(new_field_indeces[key]) == count, """
                    Axes %s has count %s while was found %s for other field(s).
                    Axes that are neither close or open, must have the same count between all fields.
                    """ % (key, count, new_field_indeces[key])
                field_indeces[-1][key] = tuple(new_field_indeces[key])
                
            elif key in open_indeces:
                field_indeces[-1][key] = tuple(_i+i for i in range(count))
                _i+=count
                if key not in new_field_indeces:
                    new_field_indeces[key] = field_indeces[-1][key]
                else:
                    new_field_indeces[key] += field_indeces[-1][key]
                
            else:
                assert key in close_indeces, "Trivial assertion."
                if key not in new_field_indeces:
                    new_field_indeces[key] = tuple(_i+i for i in range(count))
                    _i+=count
                    field_indeces[-1][key] = tuple(new_field_indeces[key])
                else:
                    assert len(new_field_indeces[key]) > 0, "Trivial assertion."
                    field_indeces[-1][key] = (new_field_indeces[key][-1],) + tuple(_i+i for i in range(count-1))
                    new_field_indeces[key] = new_field_indeces[key][:-1] + tuple(_i+i for i in range(count-1))
                    _i+=count-1
                    if len(new_field_indeces[key]) == 0:
                        del new_field_indeces[key]
                    
    field_indeces.append(new_field_indeces)

    return einsum(*fields, indeces=field_indeces)


einsum_symbols = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

def einsum(*fields, indeces=None):
    """
    Performs the einsum product between fields.
    
    Parameters:
    -----------
    fields: Field
        List of fields to perform the einsum between.
    indeces: list of dicts of indeces
        List of dictionaries for each field plus the output field if not scalar.
        Each dictionary should have a key per axis of the field.
        Every key should have a list of indeces for every repetition of the axis in the field.
        Indeces must be integers.

    Examples:
    ---------
    einsum(vector, vector, indeces=[{'x':0,'y':1,'z':2,'t':3,'spin':4,'color':5},
                                    {'x':0,'y':1,'z':2,'t':3,'spin':4,'color':6},
                                    {'x':0,'y':1,'z':2,'t':3,'color':(5,6)} ])

      [x,y,z,t,spin,color] x [x,y,z,t,spin,color] -> [x,y,z,t,color,color]
      [0,1,2,3, 4  ,  5  ] x [0,1,2,3, 4  ,  6  ] -> [0,1,2,3,  5  ,  6  ]
    """
    from .field import Field
    from .tunable import computable
    from builtins import all
    from dask.array import einsum

    fields = tuple(fields)
    
    if isinstance(indeces, dict):
        indeces = [indeces]
    elif isinstance(indeces, tuple):
        indeces = list(indeces)
    if len(indeces) == len(fields):
        indeces.append({})
        
    assert isinstance(indeces, list), "Indeces must be either list, tuple or dict (for one field only)."
    assert len(indeces) == len(fields)+1, "The length of indeces must be equal or plus one to the number of fields."
    assert all((isinstance(field, Field) for field in fields)), "All the fields must be of Field type."
    assert all((isinstance(idxs, dict) for idxs in indeces)), "Each element of the indeces list must be a dictionary"

    for idxs, field in zip(indeces, fields):
        assert set(idxs.keys()) == set(field.axes), "Indeces must be specified for all the axes."
        for key,val in idxs.items():
            if isinstance(val, int):
                idxs[key] = (val,)
            idxs[key] = tuple(val)
            assert len(idxs[key])==field.axes.count(key), "Indeces must be given for all the repetion of the axis"

    for key,val in indeces[-1].items():
        if isinstance(val, int):
            indeces[-1][key] = (val,)
        indeces[-1][key] = tuple(val)
        
    out_axes = []
    for key, idxs in indeces[-1].items():
        out_axes += [key]*len(idxs)

    raw_fields, out = prepare(*fields, elemwise=False, field_type = out_axes)

    @computable
    def get_indeces(indeces, axes_order, **kwargs):
        field_indeces = list(axes_order)
        for key, val in indeces.items():
            if len(val)>1:
                axis_order = kwargs[key+"_order"]
            else:
                axis_order = [0]
            for idx in axis_order:
                field_indeces[field_indeces.index(key)] = einsum_symbols[val[idx]]
        return "".join(field_indeces)
    
    field_indeces = []
    for field, idxs in zip(fields+(out,), indeces):
        kwargs = {key+"_order": getattr(field, key+"_order") for key,val in field.axes_counts.items() if val>1}
        field_indeces.append(get_indeces(idxs, field.axes_order, **kwargs))

    @computable
    def to_string(*field_indeces):
        field_indeces = list(field_indeces)
        return ",".join(field_indeces[:-1]) + "->"+ field_indeces[-1]
        
    out.field = computable(einsum)(to_string(*field_indeces), *raw_fields, optimize="optimal")
    
    return out


class FieldMethods:
    def __pos__(self):
        return self
    
    
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
        from .field_computables import reorder
        assert key == "axes_order", "Got wrong key! %s" % key
        return reorder(field, new_axes_order, old_axes_order)

        
    def reorder(self, *axes_order):
        """
        Changes the axes_order of the field.
        """
        from .field import Field
        return Field(self, axes_order=axes_order)
    
    
    def _rechunk(self, key, field, new_chunks, old_chunks):
        "Transformation for changing chunks"
        from .field_computables import rechunk
        assert key == "chunks", "Got wrong key! %s" % key
        return rechunk(field, self.field_chunks)
        
    def rechunk(self, **chunks):
        """
        Changes the chunks of the field.
        """
        from .field import Field
        return Field(self, chunks=chunks)
    
    
    def _squeeze(self, field, new_axes, old_axes_order, old_field_shape):
        "Transformation for squeezing axes"
        from .field_computables import squeeze
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


    def _getitem(self, field, new_coords, old_coords, old_axes_order):
        "Transformation for getitem"
        from .field_computables import getitem
        coords = {key:val for key,val in new_coords.items() if key not in old_coords or val is not old_coords[key]}
        if not coords: return field
        
        return getitem(field, self.axes, old_axes_order, **coords)
    
    
    def __getitem__(self, coords):
        from .field import Field
        return Field(field=self, coords=coords).squeeze()

    
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
        coords = {key:val for key,val in tmp.coords.items() if key not in self.coords or val is not self.coords[key]}
        self.field = setitem(self.field, tmp.field, self.axes, self.axes_order, **coords)

        
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
        from .field_computables import transpose
        assert key.endswith("_order") and key[:-6] in self.axes, "Got wrong key! %s" % key
                
        return transpose(field, key[:-6], self.axes_order, new_order, old_order)
        
        
        
    def transpose(self, *axes, **axes_order):
        """
        Transposes the matrix/tensor indeces, i.e. the dimensions that are repeated.
        
        *NOTE*: this is conceptually different from numpy.transpose where all the axes are transposed.
        
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
        Performs the trace over repeated axes contracting the outer-most index with the inner-most.
        
        Parameters
        ----------
        axes: str
            If given, only the listed axes are traced.
        """
        from .tunable import computable
        axes = list(axes)
        if not axes:
            axes = "all"
            
        axes = [axis for axis in set(self._expand(axes)) if self.axes.count(axis) > 1]
            
        if len(axes) == 1:

            axis = axes[0]
            new_axes = self.axes
            new_axes.remove(axis)
            new_axes.remove(axis)
            count = self.axes.count(axis)

            @computable
            def indeces_order(indeces_order):
                indeces_order = list(indeces_order)
                indeces_order.remove(axis+"_0")
                indeces_order.remove(axis+"_%d" % (count-1))
                return indeces_order

            indeces_order = indeces_order(self.indeces_order)
            raw_fields, out = prepare(self, elemwise=False, field_type=new_axes,
                                      indeces_order=indeces_order)

            axis1 = self.indeces_order.index(axis+"_0")
            axis2 = self.indeces_order.index(axis+"_%d" % (count-1))

            from dask.array import trace
            out.field = computable(trace)(raw_fields[0], axis1=axis1, axis2=axis2)
            
            return out
        
        else:
            _i=0
            indeces = [{}, {}]
            for axis in set(self.axes):
                count = self.axes.count(axis) 
                if axis in axes:
                    indeces[0][axis] = tuple(_i+i for i in range(count-1)) + (_i,)
                    _i+=count-1
                    if len(indeces[0][axis]) > 2:
                        indeces[-1][axis] = indeces[0][axis][1:-1]
                else:
                    indeces[0][axis] = tuple(_i+i for i in range(count))
                    _i+=count
                    indeces[-1][axis] = tuple(indeces[0][axis])
                    
            return einsum(self, indeces=indeces)
        
        
    @wraps(dot)
    def dot(self, *fields, **kwargs):
        return dot(self, *fields, **kwargs)

    
    def __matmul__(self, other):
        return self.dot(other)
    
    
    def __rmatmul__(self, other):
        return other.dot(self)


    def roll(self, axis, shift):
        """
        Rolls axis of shift.
        
        Parameters:
        -----------
        axis: str or list of str
            Axis/axes to roll of shift amount.
        shift: int or list of int
            The number of places by which elements are shifted.
        """
        from .field import Field
        from .field_computables import roll
        
        if isinstance(shift, (list,tuple)):
            assert isinstance(axis, (list,tuple)) and len(shift) == len(axis), """
            If shift is a list then also axis must be a list and the length must match.
            """
            axes, shifts = list(zip(*[(ax,sh) for ax, sh in zip(axis, shift) for ax in self._expand(ax)]))
        else:
            axes = self._expand(axis)
            shifts = [shift]*len(axis)

        to_roll = {}
        for axis, shift in zip(axes, shifts):
            if axis in to_roll:
                to_roll[axis].append(shift)
            else:
                to_roll[axis] = [shift]

        kwargs = {}
        for axis, shift in to_roll.items():
            count = self.axes.count(axis)
            assert len(shift)==1 or len(shift)==count, """
            If an axis is repeated then a shift must be given for all the repetitions.
            """
            if len(shift)==1:
                to_roll[axis] = [shift[0]]*count
            if count>1:
                kwargs[axis+"_order"] = getattr(self, axis+"_order")
                
        
        raw_fields, out = prepare(self)
        out.field = roll(raw_fields[0], to_roll, self.axes_order, **kwargs)
        return out
        
            
@wraps(FieldMethods.transpose)
def transpose(field, *args, **kwargs):
    from .field import Field
    assert isinstance(field, Field), "field must be a Field type"
    return field.transpose(*args, **kwargs)


@wraps(FieldMethods.dagger)
def dagger(field, *args, **kwargs):
    from .field import Field
    assert isinstance(field, Field), "field must be a Field type"
    return field.dagger(*args, **kwargs)


@wraps(FieldMethods.trace)
def trace(field, *args, **kwargs):
    from .field import Field
    assert isinstance(field, Field), "field must be a Field type"
    return field.trace(*args, **kwargs)


@wraps(FieldMethods.roll)
def roll(field, *args, **kwargs):
    from .field import Field
    assert isinstance(field, Field), "field must be a Field type"
    return field.roll(*args, **kwargs)


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
    ("isclose", True, ),
    ("allclose", True, ),
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


def prepare(*fields, elemwise=True, **kwargs):
    """
    Prepares a set of fields for a calculation and 
    creates the field where to store the output.

    Returns:
    --------
    raw_fields, out_field
    where raw_fields is a tuple of the raw fields (e.g. field.field)
    to be used in the calculation and out_field is a Field type where
    to store the result (e.g. out_field.field = add(*raw_fields))

    Parameters
    ----------
    fields: Field(s)
       List of fields involved in the calculation.
    elemwise: bool
       Wether the calculation is performed element-wise,
       i.e. all the fields must have the common axes in the same order
       and the other axes with shape 1.
    kwargs: dict
       List of parameters to use for the creation of the new field
    """
    from .field import Field
    from builtins import all
    assert all([isinstance(field, Field) for field in fields])
    
    kwargs["zeros_init"] = True
    raw_fields = tuple(field.field for field in fields)
    
    if len(fields)==1:
        return raw_fields, Field(fields[0], **kwargs)
    
    # TODO
    # - should compute the final dtype if not given
    # - should reshape the fields in case of element-wise operation
    # - should take into account coords
    return raw_fields, Field(fields[0], **kwargs)


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
        raw_fields, out_field = prepare(*fields, dtype=dtype)
        
        for i,field in zip(idxs, raw_fields):
            args[i] = field
            
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
            raw_fields, out = prepare(field, field_type=new_axes, dtype=dtype)
        else:
            raw_fields, out = prepare(field, field_type=[], dtype=dtype)
            
        out.field = computable(reduction)(raw_fields[0], **kwargs)
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
