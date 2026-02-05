import cvxpy as cp
import numpy as np
from dataclasses import dataclass

# These will be agnostic of trajectory scaling/units
# Scaling must be handled by the solver when it calls the init func and returns solutions
# The exact handshake between solver and init will have to be determined

@dataclass(frozen=True)
class Trajectory:
    """
    Container for numerical trajectory data. 
    """
    x: np.ndarray    
    u: np.ndarray    
    sigma: np.ndarray

    @classmethod
    def zeros(cls, K, nx, nu, ns=0):
        return cls(
            x=np.zeros((K, nx)),
            u=np.zeros((K, nu)),
            sigma=np.zeros(ns)
        )

# Sigma of size zero implise a fixed-final-time problem
# Responsibility is on the user to correctly construct the functionals for
# A, B+, B-, S, and F, all temporally scaled regardless of fixed/free final
# time

class SymbolicTrajectory:
    def __init__(self, K: int, nx: int, nu: int, ns: int = 0):
        self.K, self.nx, self.nu, self.ns = K, nx, nu, ns

        self.x = cp.Variable((K, nx), name='x')
        self.u = cp.Variable((K, nu), name='u')
        
        if ns > 0:
            self.sigma = cp.Variable(ns, name='sigma')
            self.sigma_last = cp.Parameter(ns, name='sigma_last')
        else:
            self.sigma = np.zeros(0)
            self.sigma_last = np.zeros(0)

        self.x_last = cp.Parameter((K, nx), name='x_last')
        self.u_last = cp.Parameter((K, nu), name='u_last')

        self.dx = self.x - self.x_last
        self.du = self.u - self.u_last
        self.dsigma = self.sigma - self.sigma_last


    def set_parameters(self, traj: Trajectory):
        """Sets the CVXPY parameters using a Trajectory object."""
        self.x_last.value = traj.x
        self.u_last.value = traj.u
        
        if self.ns > 0:
            self.sigma_last.value = traj.sigma

    def get_solution(self, update_params: bool = True) -> Trajectory:
        """
        Returns the current solution as a Trajectory object.
        """
        if self.x.value is None:
            raise RuntimeError("Variables are None. Has the problem been solved?")

        new_traj = Trajectory(
            x=self.x.value,
            u=self.u.value,
            sigma=self.sigma.value if self.ns > 0 else np.zeros(0)
        )

        if update_params:
            self.set_parameters(new_traj)

        return new_traj