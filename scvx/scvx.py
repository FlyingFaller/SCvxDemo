from model import Model
from integrator import Integrator
from stc import StateTriggeredConstraint
from trajectory import Trajectory, SymbolicTrajectory
from labeled_views import LabeledArray
import cvxpy as cvx
import numpy as np

class SCvx:

    def __init__(self, model: Model):
        self.m = model
        self.integrator = Integrator(model)

        # parameter and variable creation

    def solve(self):
        # Need to construct the problem first:
        #   constraints + objective
        # constraints:
        #   * using SymTraj and Ak, Bmk, Bpk, Sk, wk parameters and vk vars
        #     construct dynamics constraints
        #   * get STC obj list, loop through and construct list of wrapped STC
        #     Constraint objects
        #   * loop through and update STCs using old Trajectory obj
        #   * get constraints list from model
        # objective:
        #   * get model obj
        #   * sum vk variables
        #   * need to somehow compute the composite z = [s, x, u] and then the
        #     difference between the prev trajectory z, dz, for all k in K. 
        #   * compute tr cost, maybe look for different formula for that quadratic sum?
        #     Does s need to be duplicated? Can s, x, u be summed seperately?
        pass