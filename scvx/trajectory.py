import numpy as np
from dataclasses import dataclass
from labeled_views import LabeledArray, LabeledVariable, LabeledParameter, LabeledExpression

class AuxVariable:
    """
    Container for auxilliary variable data registered by the problem.
    """
    def __init__(self, 
                 name: str, 
                 shape: int|tuple, 
                 local: bool = True,
                 w_tr: float = 0.0,
                 labels: list[str]|None = None,
                 **cvx_kwargs):
        
        self.name = name # Name of the variable, keep consistent!
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.local = local # Whether there is 1 instance or K instances
        self.w_tr = w_tr # Zero to no be included in trust region
        self.labels = labels # Labels if multidimensional
        self.cvx_kwargs = cvx_kwargs # Args to be passed toward CVXPY variable def

class Trajectory:
    """
    Container for numerical trajectory data. 
    """
    def __init__(self,
                 x: LabeledArray,
                 u: LabeledArray,
                 **aux_vars):
        self.x = x
        self.u = u
        # kwargs of aux_var_name = LabeledArray 
        for key, val in aux_vars.items():
            setattr(self, key, val)

    @classmethod
    def zeros(cls, K, nx, nu, xlabels=None, ulabels=None, 
              aux_vars: list[AuxVariable] = None):
        
        aux_vars = aux_vars or []
        aux_kwargs = {}
        
        # Build numerical arrays for auxiliary variables
        for var in aux_vars:
            shape = (K, *var.shape) if var.local else var.shape
            aux_kwargs[var.name] = LabeledArray(np.zeros(shape), labels=var.labels)

        return cls(
            x=LabeledArray(np.zeros((K, nx)), labels=xlabels),
            u=LabeledArray(np.zeros((K, nu)), labels=ulabels),
            **aux_kwargs
        )

# Sigma of size zero implise a fixed-final-time problem
# Responsibility is on the user to correctly construct the functionals for
# A, B+, B-, S, and F, all temporally scaled regardless of fixed/free final
# time

class CvxTrajectory:
    """
    A container for the CVXPY symbolics and paramters that represent the trajetory.
    """
    def __init__(self, 
                 K: int, 
                 nx: int, 
                 nu: int,
                 xlabels = None, 
                 ulabels = None,
                 aux_vars: list[AuxVariable] = None):
        
        self.K, self.nx, self.nu= K, nx, nu
        self.xlabels, self.ulabels= xlabels, ulabels
        self.aux_vars_meta = aux_vars or []

        self.x = LabeledVariable((K, nx), labels=xlabels, name='x')
        self.u = LabeledVariable((K, nu), labels=ulabels, name='u')

        self.x_last = LabeledParameter((K, nx), labels=xlabels, name='x_last')
        self.u_last = LabeledParameter((K, nu), labels=ulabels, name='u_last')

        self.dx = LabeledExpression(self.x - self.x_last, labels=xlabels)
        self.du = LabeledExpression(self.u - self.u_last, labels=ulabels)
        
        for var in self.aux_vars_meta:
            actual_shape = (K, *var.shape) if var.local else var.shape
            cvx_var = LabeledVariable(actual_shape, labels=var.labels,
                                      name=var.name, **var.cvx_kwargs)
            setattr(self, var.name, cvx_var)
            cvx_param = LabeledParameter(actual_shape, labels=var.labels,
                                         name=f"{var.name}_last")
            setattr(self, f"{var.name}_last", cvx_param)
            cvx_diff = LabeledExpression(cvx_var - cvx_param, labels=var.labels)
            setattr(self, f"d{var.name}", cvx_diff)


    def set_parameters(self, traj: Trajectory):
        """Sets the CVXPY parameters using a Trajectory object."""
        self.x_last.value = traj.x
        self.u_last.value = traj.u
        
        for var in self.aux_vars_meta:
            param: LabeledParameter = getattr(self, f"{var.name}_last")
            param.value = getattr(traj, var.name)

    def get_result(self, update_params: bool = True) -> Trajectory:
        """
        Returns the current solution as a Trajectory object.
        """
        if self.x.value is None:
            raise RuntimeError("Variables are None. Has the problem been solved?")

        aux_kwargs = {}
        for var in self.aux_vars_meta:
            cvx_var: LabeledVariable = getattr(self, var.name)
            aux_kwargs[var.name] = LabeledArray(cvx_var.value, labels=var.labels)

        new_traj = Trajectory(
            x=LabeledArray(self.x.value, labels=self.xlabels),
            u=LabeledArray(self.u.value, labels=self.ulabels),
            **aux_kwargs
        )

        if update_params:
            self.set_parameters(new_traj)

        return new_traj

# placeholder for solution package
@dataclass
class Solution:
    trajectory: Trajectory
    cost: float
    stc: np.ndarray
    status: bool