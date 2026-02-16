import numpy as np
import cvxpy as cvx

import cvxpy as cvx
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS

class LabeledMixin:
    """
    Mixin for labeled dimension support.
    """

    def _init_labels(self, labels, axis=-1):
        self._labeled_axis = axis
        self._labels_list = list(labels) if labels else []
        
        if self._labels_list:
            self._label_map = {n: i for i, n in enumerate(self._labels_list)}
        else:
            self._label_map = {}

    def _is_labeled_query(self, item):
        """Helper to determine if an indexing item is querying by label."""
        if isinstance(item, str):
            return True
        if isinstance(item, slice):
            return isinstance(item.start, str) or isinstance(item.stop, str)
        if isinstance(item, (list, tuple)):
            return any(isinstance(k, str) for k in item)
        return False

    def _translate_slice(self, item: slice, is_labeled_axis: bool):
        """Helper to translate a slice, making string boundaries inclusive."""
        
        start = self._translate_idx(item.start, is_labeled_axis)
        stop = self._translate_idx(item.stop, is_labeled_axis)
        
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

    def _translate_idx(self, item, is_labeled_axis: bool):
        """
        Translates a single idx in the key (str, slice, list, tuple).
        """
        if isinstance(item, str):
            if not is_labeled_axis:
                raise IndexError(f"String key '{item}' invalid on non-labeled axis.")
            if item not in self._label_map:
                raise KeyError(f"Key '{item}' not found in label mapping.")
            return self._label_map[item]
        
        elif isinstance(item, slice):
            return self._translate_slice(item, is_labeled_axis)
        
        elif isinstance(item, (list, tuple)):
            # Translate elements but preserve the original container type (list or tuple)
            # Important for fancy indexing 
            return type(item)(self._translate_idx(k, is_labeled_axis) for k in item)
        
        return item

    def _resolve_key(self, key, shape: tuple):
        """
        Top-level loop that iterates through the indexing dimensions.
        """
        # If the labeled axis has already been burned none of this logic matters
        if self._labeled_axis is None:
            return key, None, []
        
        # Support negative axes safely
        ndim = len(shape)
        labeled_axis = self._labeled_axis % ndim if ndim > 0 else 0

        # Make the key a tuple
        if not isinstance(key, tuple):
            if self._is_labeled_query(key):
                # Special Case: Single labeled query -> Auto-pad to reach labeled axis
                key_tuple = (slice(None),) * labeled_axis + (key,)
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
        new_labeled_axis = labeled_axis
        new_labels = self._labels_list

        for item in key_tuple:
            # None adds an output dimension but doesn't consume an input dimension
            if item is None:
                new_key.append(None)
                if current_axis <= labeled_axis:
                    new_labeled_axis += 1
                continue 

            # Resolve induvidual key
            is_labeled_axis = (current_axis == labeled_axis)
            resolved_idx = self._translate_idx(item, is_labeled_axis)
            new_key.append(resolved_idx)
            is_scalar = isinstance(resolved_idx, (int, np.integer))

            # Handle axis tracking
            if is_labeled_axis:
                if is_scalar: 
                    # We are selecting within the labeled axis and completely eliminating it
                    new_labeled_axis = None
                    new_labels = []
                elif self._labels_list:
                    # Figure out which labels remain because we are slicing or selecting multiple 
                    # We have to make tuples lists so they will fancy index correctly as the first
                    # indices in the key. Quirk of numpy indexing. 
                    labels_idx = list(resolved_idx) if isinstance(resolved_idx, tuple) else resolved_idx
                    new_labels = np.array(self._labels_list)[labels_idx].tolist()
            elif is_scalar and current_axis < labeled_axis:
                # We haven't hit the labeled axis yet and are reducing the number of axis
                new_labeled_axis -= 1

            current_axis += 1

        return tuple(new_key), new_labeled_axis, new_labels

class LabeledArray(np.ndarray, LabeledMixin):
    def __new__(cls, input_array, labels=None, axis=-1):
        obj = np.asarray(input_array).view(cls)
        obj._init_labels(labels, axis)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._label_map = getattr(obj, '_label_map', {})
        self._labels_list = getattr(obj, '_labels_list', [])
        self._labeled_axis = getattr(obj, '_labeled_axis', -1)

    def __getitem__(self, key):
        new_key, next_axis, next_labels = self._resolve_key(key, self.shape)
        result = super().__getitem__(new_key)
            
        if isinstance(result, LabeledArray):
            result._init_labels(next_labels, next_axis)
            
        return result
    
    def __setitem__(self, key, value):
        new_key, *_ = self._resolve_key(key, self.shape)
        super().__setitem__(new_key, value)

class LabeledVariable(cvx.Variable, LabeledMixin):
    def __init__(self, shape, labels=None, axis=-1, **kwargs):
        super().__init__(shape, **kwargs)
        self._init_labels(labels, axis)

    def __getitem__(self, key):
        new_key, *_ = self._resolve_key(key, self.shape)
        return super().__getitem__(new_key)

class LabeledParameter(cvx.Parameter, LabeledMixin):
    def __init__(self, shape, labels=None, axis=-1, **kwargs):
        super().__init__(shape, **kwargs)
        self._init_labels(labels, axis)

    def __getitem__(self, key):
        new_key, *_ = self._resolve_key(key, self.shape)
        return super().__getitem__(new_key)

class LabeledExpression(cvx.Expression, LabeledMixin):
    """
    A robust wrapper that satisfies the CVXPY Expression contract 
    by delegating all properties to the underlying expression.
    """
    def __init__(self, expr, labels=None, axis=-1):
        self._expr = cvx.Expression.cast_to_const(expr)
        
        # 1. Initialize Labels
        self._init_labels(labels, axis)
        
        # 2. Initialize CVXPY Expression (Must set shape and args)
        super().__init__()
        self.args = [self._expr] # Registers the child for the solver graph

    def __getattr__(self, attr):
        # This is only called if 'attr' is NOT found on self.
        # It forwards the request to the underlying expression.
        return getattr(self._expr, attr)

    # --- The Boilerplate (Delegation) ---
    def name(self): return self._expr.name()
    def is_convex(self): return self._expr.is_convex()
    def is_concave(self): return self._expr.is_concave()
    def is_log_log_convex(self): return self._expr.is_log_log_convex()
    def is_log_log_concave(self): return self._expr.is_log_log_concave()
    def is_nonneg(self): return self._expr.is_nonneg()
    def is_nonpos(self): return self._expr.is_nonpos()
    def is_complex(self): return self._expr.is_complex()
    def is_dpp(self, context='dcp'): return self._expr.is_dpp(context)
    
    # Properties must be forwarded explicitly
    @property
    def domain(self): return self._expr.domain
    @property
    def grad(self): return self._expr.grad
    @property
    def value(self): return self._expr.value
    @property
    def shape(self): return self._expr.shape
    @property
    def is_imag(self): return self._expr.is_imag

    # --- Labeled Slicing Logic ---
    def __getitem__(self, key):
        # 1. Resolve key via Mixin
        new_key, next_axis, next_labels = self._resolve_key(key, self.shape)
        
        # 2. Slice the child expression
        sliced_expr = self._expr[new_key]
        
        # 3. Re-wrap if it's still an Expression and we have labels
        if isinstance(sliced_expr, cvx.Expression) and next_axis is not None:
             return LabeledExpression(sliced_expr, next_labels, next_axis)
        
        return sliced_expr

# --- Critical: Register Canonicalization ---
# This tells CVXPY: "When you solve, replace LabeledExpression with its child."
def labeled_canon(expr, args):
    return args[0], []

CANON_METHODS[LabeledExpression] = labeled_canon