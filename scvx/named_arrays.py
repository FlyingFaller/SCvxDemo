import numpy as np
import cvxpy as cvx

class NamedMixin:
    """
    Mixin for named dimension support.
    """

    def _init_names(self, names, axis=-1):
        self._named_axis = axis
        self._names_list = list(names) if names else []
        
        if self._names_list:
            self._name_map = {n: i for i, n in enumerate(self._names_list)}
        else:
            self._name_map = {}

    def _is_named_query(self, item):
        """Helper to determine if an indexing item is querying by name."""
        if isinstance(item, str):
            return True
        if isinstance(item, slice):
            return isinstance(item.start, str) or isinstance(item.stop, str)
        if isinstance(item, (list, tuple)):
            return any(isinstance(k, str) for k in item)
        return False

    def _translate_slice(self, item: slice, is_named_axis: bool):
        """Helper to translate a slice, making string boundaries inclusive."""
        
        start = self._translate_idx(item.start, is_named_axis)
        stop = self._translate_idx(item.stop, is_named_axis)
        
        # For stop index specifically we break from numpy and do inclusive indexing
        # This makes more sense for natural language based slicing. 
        if isinstance(item.stop, str):
            step = item.step if item.step is not None else 1
            if step > 0:
                stop += 1
            else:
                # If negative step, python slices require None to include index 0
                stop = stop - 1 if stop > 0 else None
                
        return slice(start, stop, item.step)

    def _translate_idx(self, item, is_named_axis: bool):
        """
        Translates a single idx in the key (str, slice, list, tuple).
        """
        if isinstance(item, str):
            if not is_named_axis:
                raise IndexError(f"String key '{item}' invalid on non-named axis.")
            if item not in self._name_map:
                raise KeyError(f"Key '{item}' not found in name mapping.")
            return self._name_map[item]
        
        elif isinstance(item, slice):
            return self._translate_slice(item, is_named_axis)
        
        elif isinstance(item, (list, tuple)):
            # Translate elements but preserve the original container type (list or tuple)
            # Important for fancy indexing 
            return type(item)(self._translate_idx(k, is_named_axis) for k in item)
        
        return item

    def _resolve_key(self, key, shape: tuple):
        """
        Top-level loop that iterates through the indexing dimensions.
        """
        # If the named axis has already been burned none of this logic matters
        if self._named_axis is None:
            return key, None, []
        
        # Support negative axes safely
        ndim = len(shape)
        named_axis = self._named_axis % ndim if ndim > 0 else 0

        # Make the key a tuple
        if not isinstance(key, tuple):
            if self._is_named_query(key):
                # Special Case: Single named query -> Auto-pad to reach named axis
                key_tuple = (slice(None),) * named_axis + (key,)
            else:
                key_tuple = (key,)
        else:
            key_tuple = key

        # Expand ellipses so the key tuple matches data dimension.
        # This way we don't have to worry about them later.
        if Ellipsis in key_tuple:
            # Everything except None and ... consumes an input dimension
            consumed_dims = sum(1 for k in key_tuple if k is not None and k is not Ellipsis)
            missing_dims = max(0, ndim - consumed_dims)
            
            e_idx = key_tuple.index(Ellipsis)
            key_tuple = key_tuple[:e_idx] + (slice(None),) * missing_dims + key_tuple[e_idx + 1:]

        # Loop through keys and handle them induvidually
        new_key = []
        current_axis = 0
        new_named_axis = named_axis
        new_names = self._names_list

        for item in key_tuple:
            # None adds an output dimension but doesn't consume an input dimension
            if item is None:
                new_key.append(None)
                if current_axis <= named_axis:
                    new_named_axis += 1
                continue 

            # Resolve induvidual key
            is_named_axis = (current_axis == named_axis)
            resolved_idx = self._translate_idx(item, is_named_axis)
            new_key.append(resolved_idx)
            is_scalar = isinstance(resolved_idx, (int, np.integer))

            # Handle axis tracking
            if is_named_axis:
                if is_scalar: 
                    # We are selecting within the named axis and completely eliminating it
                    new_named_axis = None
                    new_names = []
                elif self._names_list:
                    # Figure out which names remain because we are slicing or selecting multiple 
                    # We have to make tuples lists so they will fancy index correctly as the first
                    # indices in the key. Quirk of numpy indexing. 
                    names_idx = list(resolved_idx) if isinstance(resolved_idx, tuple) else resolved_idx
                    new_names = np.array(self._names_list)[names_idx].tolist()
            elif is_scalar and current_axis < named_axis:
                # We haven't hit the named axis yet and are reducing the number of axis
                new_named_axis -= 1

            current_axis += 1

        return tuple(new_key), new_named_axis, new_names

class NamedArray(np.ndarray, NamedMixin):
    def __new__(cls, input_array, names=None, axis=-1):
        obj = np.asarray(input_array).view(cls)
        obj._init_names(names, axis)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._name_map = getattr(obj, '_name_map', {})
        self._names_list = getattr(obj, '_names_list', [])
        self._named_axis = getattr(obj, '_named_axis', -1)

    def __getitem__(self, key):
        new_key, next_axis, next_names = self._resolve_key(key, self.shape)
        result = super().__getitem__(new_key)
            
        if isinstance(result, NamedArray):
            result._init_names(next_names, next_axis)
            
        return result
    
    def __setitem__(self, key, value):
        new_key, _, _, _ = self._resolve_key(key, self.shape)
        super().__setitem__(new_key, value)

class NamedVariable(cvx.Variable, NamedMixin):
    def __init__(self, shape, names=None, axis=-1, **kwargs):
        super().__init__(shape, **kwargs)
        self._init_names(names, axis)

    def __getitem__(self, key):
        new_key, *_ = self._resolve_key(key, self.shape)
        return super().__getitem__(new_key)

class NamedParameter(cvx.Parameter, NamedMixin):
    def __init__(self, shape, names=None, axis=-1, **kwargs):
        super().__init__(shape, **kwargs)
        self._init_names(names, axis)

    def __getitem__(self, key):
        new_key, *_ = self._resolve_key(key, self.shape)
        return super().__getitem__(new_key)