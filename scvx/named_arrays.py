import numpy as np
import cvxpy as cvx

class NamedMixin:
    """
    Shared logic for resolving string keys to integer indices.
    """
    def _init_names(self, names, axis=-1):
        self._named_axis = axis
             
        if names:
            self._name_map = {n: i for i, n in enumerate(names)}
        else:
            self._name_map = {}

    def _resolve_key(self, key, shape):
        ndim = len(shape)
        # Ensure axis is positive for comparison
        target_axis = self._named_axis % ndim 
        
        # We want a tuple of length <= ndim to iterate over.
        if isinstance(key, str):
            # SPECIAL CASE: If user provides a single string, they intend to 
            # hit the named axis directly
            # Create a slice(None) for every dimension before the target
            key_tuple = (slice(None),) * target_axis + (key,)
        elif not isinstance(key, tuple):
            key_tuple = (key,)
        else:
            key_tuple = key

        new_key = []
        drop_wrapper = False
        current_axis = 0

        for item in key_tuple:
            # Handle Ellipsis
            if item is Ellipsis:
                # Calculate how many axes the ellipsis covers
                # Total dims - (dims already covered) - (dims remaining in key)
                remaining_items = len(key_tuple) - (len(new_key) + 1)
                num_axes_skipped = ndim - current_axis - remaining_items
                
                # If the named axis falls within the skipped region, we keep the wrapper
                # (Ellipsis implies slicing those dimensions)
                if current_axis <= target_axis < current_axis + num_axes_skipped:
                    drop_wrapper = False 
                
                new_key.append(item)
                current_axis += num_axes_skipped
                continue

            # Standard Logic
            is_string = isinstance(item, str)
            is_int = isinstance(item, (int, np.integer)) and not isinstance(item, bool)
            
            if current_axis == target_axis:
                # WE ARE AT THE NAMED AXIS
                if is_string:
                    if item not in self._name_map:
                        raise KeyError(f"Key '{item}' not found in named axis {target_axis}.")
                    new_key.append(self._name_map[item])
                    drop_wrapper = True # String selects a single index -> Drop
                elif is_int:
                    new_key.append(item)
                    drop_wrapper = True # Int selects a single index -> Drop
                else:
                    # Slices, arrays, lists, etc.
                    new_key.append(item)
                    drop_wrapper = False # Keeping a range -> Keep wrapper
            else:
                # WE ARE AT A NON-NAMED AXIS
                if is_string:
                    raise IndexError(
                        f"String key '{item}' provided for axis {current_axis}, "
                        f"but names are assigned to axis {target_axis}."
                    )
                new_key.append(item)
            
            current_axis += 1

        return tuple(new_key), drop_wrapper

# --- 1. NumPy Wrapper ---

class NamedArray(np.ndarray, NamedMixin):
    def __new__(cls, input_array, names=None, axis=-1):
        obj = np.asarray(input_array).view(cls)
        # We must set _init_names here. Note: ndim is available on obj now.
        obj._init_names(names, axis)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._name_map = getattr(obj, '_name_map', {})
        self._named_axis = getattr(obj, '_named_axis', -1)

    def __getitem__(self, key):
        new_key, drop_wrapper = self._resolve_key(key, self.shape)
        result = super().__getitem__(new_key)
        
        if drop_wrapper:
            # Return raw numpy array/scalar to prevent further string indexing
            return np.asarray(result)
        return result

# --- 2. CVXPY Wrappers ---

class NamedVariable(cvx.Variable, NamedMixin):
    def __init__(self, shape, names=None, axis=-1, **kwargs):
        super().__init__(shape, **kwargs)
        self._init_names(names, axis)

    def __getitem__(self, key):
        new_key, _ = self._resolve_key(key, self.shape)
        return super().__getitem__(new_key)

class NamedParameter(cvx.Parameter, NamedMixin):
    def __init__(self, shape, names=None, axis=-1, **kwargs):
        super().__init__(shape, **kwargs)
        self._init_names(names, axis)

    def __getitem__(self, key):
        new_key, _ = self._resolve_key(key, self.shape)
        return super().__getitem__(new_key)