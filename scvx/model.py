from abc import ABC, abstractmethod
from trajectory import Trajectory, SymbolicTrajectory
from stc import StateTriggeredConstraint
from typing import Callable
import cvxpy as cvx
from labeled_views import LabeledArray
import numpy as np

class Model(ABC):
    @abstractmethod
    def get_dynamics(self) -> tuple[Callable[[LabeledArray, LabeledArray, LabeledArray], np.ndarray[float]], 
                                    Callable[[LabeledArray, LabeledArray, LabeledArray], np.ndarray[float]],
                                    Callable[[LabeledArray, LabeledArray, LabeledArray], np.ndarray[float]],
                                    Callable[[LabeledArray, LabeledArray, LabeledArray], np.ndarray[float]]]:
        """
        [Called Once] Returns the linearized dynamics as a tuple of callables: A(s, x, u), B(s, x, u), S(s, x, u), and f(s, x, u). 
        The arguments s, x, and u passed to each callable are 1D auxilliary, state, and control vectors. Note: the discretizations 
        must be temporally normalized for all problems times fixed- and free-final-time. The discretizations must also be constructed
        with normalized and nondimensionalized coefficients appropriately. 
        
        :return: (A(s, x, u), B(s, x, u), S(s, x, u), f(s, x, u))
        """
        raise NotImplementedError

    @abstractmethod
    def init_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        [Called Once] Initializes the solver with a Trajectory object containing auxilliary decision varaiables vector,
        state array, and controls array. Note: the initial trajectory must be appropriately normalized and nondimensionalized
        by the user before passing to the solver. 
        
        :param traj: Empty, initialized, trajectory objects.
        :return: Trajectory object containing intial trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self, traj: SymbolicTrajectory) -> list[cvx.Constraint]:
        """
        [Called Once] Returns a list of Constraint objects composed of the parameterized new and old trajectories. Note: the constraints
        must be constructed with normalized and nondimensionalized coefficients appropriately. 
        
        :param traj: SymbolicTrajectory containing parameterized auxilliary, state, and control arrays and their differences.
        :return: List of CVXPY Constraint objects.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_objective(self, traj: SymbolicTrajectory) -> cvx.Expression:
        """
        [Called Once] Returns the problem objective function Expression---the thing to be minimized. Note: must be constructed with 
        normalized and nondimensionalized coefficients appropriately. 
        
        :param traj: SymbolicTrajectory containing parameterized auxilliary, state, and control arrays and their differences.
        :return: CVXPY Expression.
        """
        raise NotImplementedError
    
    # Optional for problems w/o STCs
    def get_STCs(self, traj: SymbolicTrajectory) -> list[StateTriggeredConstraint]:
        """
        [Optional][Called Once] Returns a list of StateTriggeredConstraint objects if STCs are used in the problem. Each contains
        a valid CVXPY Constraint object and a callable trigger function that returns an appropriately sized array of bools. Note: 
        must be constructed with normalized and nondimensionalized coefficients appropriately. 
        
        :param traj: SymbolicTrajectory containing parameterized auxilliary, state, and control arrays and their differences.
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
    def ns(self) -> int: 
        """Number of auxilliary decision variables in the problem (e.g. ns >= 1 for all free-final-time type problems)."""
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

    # Technically optional
    @property
    def slabels(self) -> list[str]: 
        """[Optional] String labels for each auxilliary variables if any exist."""
        return None
    
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

    # Is it enough to supply just a positive diagonal? What about scalar values? 
    # Do the axilliary variables get added onto each sum or can they be added to the X and U sum?

    @property
    @abstractmethod
    def w_tr(self) -> np.ndarray[float]:
        """
        Penalty weights on trust region violations for each decision variable. 
        Must be a Symmetric Positive Definite matrix of shape (ns+nx+nu, ns+nx+nu). 
        """
        raise NotImplementedError


    
