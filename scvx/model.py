from abc import ABC, abstractmethod
from trajectory import Trajectory, CvxTrajectory, AuxVariable
from stc import StateTriggeredConstraint
from typing import Callable, Any
import cvxpy as cvx
import numpy as np

class Model(ABC):
    
    @property
    def aux_vars(self) -> list[AuxVariable]:
        """
        [Optional] List of auxiliary variables registered in the problem.
        Can be overridden as a property or instantiated in the subclass __init__.
        """
        return []

    @abstractmethod
    def get_dynamics(self) -> tuple[Callable, Callable, Callable, dict[str, Callable]]:
        """
        [Called Once] Returns the linearized dynamics as a tuple of callables and an optional dictionary.
        
        The implicit positional argument order for ALL callables returned by this function is:
        (*args) = (x, u, *aux_jacs.keys())
        
        :return: (F_func, A_func, B_func, aux_jacs)
            - F_func: Evaluates the continuous dynamics.
            - A_func: Evaluates the Jacobian with respect to the state (x).
            - B_func: Evaluates the Jacobian with respect to the control (u).
            - aux_jacs: A dictionary mapping auxiliary variable names to their Jacobian callables.
                        (e.g., {'tf': tf_jac_func, 'mass': mass_jac_func})
        """
        raise NotImplementedError

    @abstractmethod
    def init_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        [Called Once] Initializes the solver with a Trajectory object containing state array, 
        controls array, and any registered auxiliary variables. Note: the initial trajectory 
        must be appropriately normalized and nondimensionalized before passing to the solver. 
        
        :param traj: Empty, initialized, trajectory object.
        :return: Trajectory object containing initial guesses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self, traj: CvxTrajectory) -> list[Any]:
        """
        [Called Once] Returns a list of Constraint objects composed of the parameterized new 
        and old trajectories. Note: the constraints must be constructed with normalized and 
        nondimensionalized coefficients appropriately. 
        
        :param traj: CvxTrajectory containing parameterized state, control, and aux arrays.
        :return: List of CVXPY Constraint objects (and potentially future NonlinearConstraint objects).
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_objective(self, traj: CvxTrajectory) -> cvx.Minimize:
        """
        [Called Once] Returns the problem Objective Object---the thing to be minimized. Note: 
        must be constructed with normalized and nondimensionalized coefficients appropriately. 
        
        :param traj: CvxTrajectory containing parameterized state, control, and aux arrays.
        :return: CVXPY Minimize.
        """
        raise NotImplementedError
    
    # Optional for problems w/o STCs
    def get_STCs(self, traj: CvxTrajectory) -> list[StateTriggeredConstraint]:
        """
        [Optional][Called Once] Returns a list of StateTriggeredConstraint objects if STCs 
        are used in the problem. Each contains a valid CVXPY Constraint object and a callable 
        trigger function that returns an appropriately sized array of bools.
        
        :param traj: CvxTrajectory containing parameterized state, control, and aux arrays.
        :return: List of StateTriggeredConstraint objects.
        """
        return []

    @property
    @abstractmethod
    def K(self) -> int: 
        """Number of discretization nodes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nx(self) -> int: 
        """Number of state variables in the problem."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nu(self) -> int: 
        """Number of control variables in the problem."""
        raise NotImplementedError
    
    @property
    def xlabels(self) -> list[str]: 
        """[Optional] String labels for each state variable."""
        return None
    
    @property
    def ulabels(self) -> list[str]: 
        """[Optional] String labels for each control variable."""
        return None

    @property
    @abstractmethod
    def max_iters(self) -> int: 
        """Maximum number of iterations to allowed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tol_vc(self) -> float: 
        """Convergence tolerance on cost of virtual control usage."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tol_tr(self) -> float: 
        """Convergence tolerance on cost of trust region violations."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def w_vc(self) -> float: 
        """Penalty weight on virtual control usage."""
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def w_tr(self) -> np.ndarray:
    #     """
    #     Penalty weights on trust region violations for the core state and control variables.
    #     Auxiliary variables define their own trust region penalties in the AuxVariable registry.
        
    #     Must be a Symmetric Positive Definite matrix of shape (nx+nu, nx+nu). 
    #     """
    #     raise NotImplementedError

    @property
    @abstractmethod
    def w_tr_x(self) -> np.ndarray:
        """
        Penalty weights on trust region violations for the state variables.
        
        Must be an array of length nx.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def w_tr_u(self) -> np.ndarray:
        """
        Penalty weights on trust region violations for the control variables.
        
        Must be an array of length nu.
        """
        raise NotImplementedError