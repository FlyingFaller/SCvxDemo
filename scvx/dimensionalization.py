import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

class BaseScaledProp(NDArrayOperatorsMixin):
    def __init__(self, value, scalar=1, axis=None, units=None):
        self.value = value
        self.scalar = np.asarray(scalar) if isinstance(scalar, (list, tuple)) else scalar
        self.axis = axis
        self.units = units
        self._validate_shapes()

    # --- Validation Logic ---
    def _validate_shapes(self):
        val_ndim = np.ndim(self.value)
        sca_ndim = np.ndim(self.scalar)
        if sca_ndim > val_ndim:
            raise ValueError(f"Scalar rank ({sca_ndim}) > Value rank ({val_ndim})")
        if self.axis is not None and val_ndim > 0 and sca_ndim > 0:
            val_shape = np.shape(self.value)
            sca_shape = np.shape(self.scalar)
            ax = self.axis % val_ndim
            if sca_shape[0] != val_shape[ax]:
                 raise ValueError(f"Dim mismatch at axis {ax}: {val_shape[ax]} vs {sca_shape[0]}")

    def _get_broadcastable_scalar(self):
        if self.axis is None or np.ndim(self.scalar) == 0:
            return self.scalar
        val_ndim = np.ndim(self.value)
        shape = [1] * val_ndim
        shape[self.axis % val_ndim] = -1
        return self.scalar.reshape(shape)

    # --- THE MAGIC METHOD ---
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercepts ALL numpy operators (+, -, *, np.sin, etc).
        """
        # 1. Unwrap inputs: If an input is a DimValue/NDimValue, extract its .value
        #    If it's a regular number/array, leave it alone.
        args = []
        for x in inputs:
            if isinstance(x, BaseScaledProp):
                args.append(x.value)
            else:
                args.append(x)
        
        # 2. Perform the actual operation on the unwrapped values
        #    getattr(ufunc, method) handles calls like np.add.reduce, etc.
        #    For standard math, method is usually '__call__'
        result = getattr(ufunc, method)(*args, **kwargs)
        
        # 3. Return the raw result (breaking out of the wrapper)
        return result

    # Standard proxies
    def __getattr__(self, name):
        return getattr(self.value, name)
    
    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype)
    
    def __repr__(self):
        return f"<{self.__class__.__name__} value={self.value} scalar={self.scalar} units={self.units}>"

# --- Subclasses ---

class DimProp(BaseScaledProp):
    @property
    def dim(self):
        return self.value
    @property
    def ndim(self):
        return self.value / self._get_broadcastable_scalar()

class NonDimProp(BaseScaledProp):
    @property
    def ndim(self):
        return self.value
    @property
    def dim(self):
        return self.value * self._get_broadcastable_scalar()