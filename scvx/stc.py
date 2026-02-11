import cvxpy as cvx
from typing import Callable
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality
import numpy as np
from trajectory import Trajectory

class StateTriggeredConstraint:
    def __init__(self, constraint: cvx.Constraint, trigger_func: Callable):
        """
        :param constraint: A raw CVXPY constraint (e.g., sigma >= T_min)
        :param trigger_func: A callable(prev_traj) -> np.array[bool]
                             Returns a mask matching the constraint's shape.
        """
        self.trigger_func = trigger_func
        self.original_constraint = constraint
        self.expression = constraint.args[0] - constraint.args[1]
        self.shape = constraint.shape if constraint.shape else (1,)
        self.mask_param = cvx.Parameter(self.shape, nonneg=True)
        
        if isinstance(constraint, Inequality):
            self.wrapped_constraint = (
                cvx.multiply(self.mask_param, self.expression) <= 0
            )
        elif isinstance(constraint, Equality):
            self.wrapped_constraint = (
                cvx.multiply(self.mask_param, self.expression) == 0
            )
        else:
            raise ValueError("STC only supports <=, >=, or == constraints")
        
    @property
    def constraint(self):
        """Return list for Problem"""
        return [self.wrapped_constraint]
    
    def update(self, prev_traj: Trajectory):
        """
        Calculates the mask based on the previous trajectory 
        and updates the internal parameter.
        """

        mask_values = np.asarray(self.trigger_func(prev_traj))
        
        if mask_values.shape != self.shape:
             raise ValueError(f"Trigger shape {mask_values.shape} mismatch with constraint {self.shape}")
             
        self.mask_param.value = mask_values.astype(float)