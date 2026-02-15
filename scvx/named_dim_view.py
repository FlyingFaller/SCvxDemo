import numpy as np
from named_views import NamedArray

class NamedDimArray(NamedArray):
    """
    Combines NamedArray and dimensional scaling properties.
    Always returns a NamedDimArray instance, even for scalar slices, 
    to preserve .dim and .nondim functionality.
    """
    def __new__(cls, data, scalar=1, names=None, axis=-1, broadcast=True):
        # Create underlying NamedArray
        named_axis = axis[0] if hasattr(axis, '__iter__') else axis
        obj = super().__new__(cls, data, names=names, axis=named_axis)
        # Initialize broadcast scalar
        if broadcast:
            obj._scalar = obj._broadcast_scalar(scalar, axis)
        else:
            obj._scalar = scalar
        return obj

    def _broadcast_scalar(self, scalar, axes):
        """
        Broadcasting logic to ensure scalar shape matches data shape.
        """
        scl = np.array(scalar)
        scl_shape = scl.shape
        scl_ndim = scl.ndim
        dat_shape = self.shape
        dat_ndim = self.ndim

        # Simple scalar scalar case
        if scl_ndim == 0: return np.broadcast_to(scl, dat_shape)

        # Make sure axis is iterable
        if not hasattr(axes, '__iter__'): axes = (axes,)

        # Basic error handling
        if len(axes) != scl_ndim:
            raise ValueError(
                f"Dimension Mismatch: Source array is {scl_ndim}-D, but "
                f"{len(axes)} target axes were provided. They must match."
            )
        
        if len(set(axes)) != len(axes):
            raise ValueError(f"Duplicate axes found in mapping: {axes}")
        
        view_shape = [1] * dat_ndim

        for i, axis in enumerate(axes):
            if axis < 0:
                axis += dat_ndim
                
            # Bounds check
            if axis >= dat_ndim or axis < 0:
                raise ValueError(f"Target axis {axis} is out of bounds for data with rank {dat_ndim}")

            # Scalar shape must match data shape on the target axes            
            if scl_shape[i] != dat_shape[axis]:
                raise ValueError(
                    f"Shape Mismatch at data dimension {i} and target axis {axis}. "
                    f"Scalar size {scl_shape[i]} != Data size {dat_shape[axis]}."
                )

            # Fill in the broadcast view with the target axis
            view_shape[axis] = scl_shape[i]
        
        # Reshape and broadcast
        reshaped_scl = scl.reshape(view_shape)
        return np.broadcast_to(reshaped_scl, dat_shape)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj) # Finalize NamedArray

    def __getitem__(self, key):
        new_key, next_axis, next_names = self._resolve_key(key, self.shape)
        # Directly call __getitem__ for underlying ndarray so _resolve_key is not called twice
        data_slice = super(NamedArray, self).__getitem__(new_key)
        scalar_slice = self._scalar[new_key] # slice the scalar with the data slicing

        # Always return NamedDimArray so dim and nondim props are accessible
        # Pass _resolve_key results to __new__ which calls NamedArray.__new__ which calls
        # the _init_names function we need.
        return NamedDimArray(data_slice, 
                             scalar=scalar_slice, 
                             names=next_names, 
                             axis=next_axis, 
                             broadcast=False)

    @property
    def dim(self):
        """Returns the dimensional data."""
        val = self.view(np.ndarray)
        return val.item() if val.ndim == 0 else val
    
    @property
    def nondim(self):
        """Returns the non-dimensional data."""
        val = self.view(np.ndarray) / self._scalar
        return val.item() if val.ndim == 0 else val

    def __float__(self): return float(self.item())
    def __int__(self): return int(self.item())