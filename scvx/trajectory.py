import cvxpy as cvx
import numpy as np
from dataclasses import dataclass
from labeled_views import LabeledArray, LabeledVariable, LabeledParameter, LabeledExpression

@dataclass(frozen=True)
class Trajectory:
    """
    Container for numerical trajectory data. 
    """
    x: LabeledArray    
    u: LabeledArray    
    sigma: LabeledArray

    @classmethod
    def zeros(cls, K, nx, nu, ns=0, xlabels=None, ulabels=None, slabels=None):
        return cls(
            x=LabeledArray(np.zeros((K, nx)), xlabels),
            u=LabeledArray(np.zeros((K, nu)), ulabels),
            sigma=LabeledArray(np.zeros(ns), slabels)
        )

# Sigma of size zero implise a fixed-final-time problem
# Responsibility is on the user to correctly construct the functionals for
# A, B+, B-, S, and F, all temporally scaled regardless of fixed/free final
# time

class SymbolicTrajectory:
    def __init__(self, K: int, nx: int, nu: int, ns: int = 0,
                 xlabels = None, ulabels = None, slabels = None):
        self.K, self.nx, self.nu, self.ns = K, nx, nu, ns
        self.xlabels, self.ulabels, self.slabels = xlabels, ulabels, slabels

        self.x = LabeledVariable((K, nx), labels=xlabels, name='x')
        self.u = LabeledVariable((K, nu), labels=ulabels, name='u')
        
        if ns > 0:
            self.sigma = LabeledVariable(ns, labels=slabels, name='sigma')
            self.sigma_last = LabeledParameter(ns, labels=slabels, name='sigma_last')
        else:
            self.sigma = np.zeros(0)
            self.sigma_last = np.zeros(0)

        self.x_last = LabeledParameter((K, nx), labels=xlabels, name='x_last')
        self.u_last = LabeledParameter((K, nu), labels=ulabels, name='u_last')

        self.dx = LabeledExpression(self.x - self.x_last, labels=xlabels)
        self.du = LabeledExpression(self.u - self.u_last, labels=ulabels)
        self.dsigma = LabeledExpression(self.sigma - self.sigma_last, labels=slabels)


    def set_parameters(self, traj: Trajectory):
        """Sets the CVXPY parameters using a Trajectory object."""
        self.x_last.value = traj.x
        self.u_last.value = traj.u
        
        if self.ns > 0:
            self.sigma_last.value = traj.sigma

    def get_result(self, update_params: bool = True) -> Trajectory:
        """
        Returns the current solution as a Trajectory object.
        """
        if self.x.value is None:
            raise RuntimeError("Variables are None. Has the problem been solved?")

        new_traj = Trajectory(
            x=LabeledArray(self.x.value, labels=self.xlabels),
            u=LabeledArray(self.u.value, labels=self.ulabels),
            sigma=LabeledArray(self.sigma.value, labels=self.slabels) if self.ns > 0 else np.zeros(0)
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