from model import Model
from integrator import Integrator
from stc import StateTriggeredConstraint
from trajectory import Trajectory, SymbolicTrajectory
from labeled_views import LabeledArray
import cvxpy as cvx
import numpy as np

class SCvxProblem:

    def __init__(self, model: Model):
        self.m = model

        # Allows for calculation of Ak Bmk, Bpk, Sk, Wk from Trajectory obj
        self.integrator = Integrator(model)

        # Stores current trajectory, past trajectory parameters, and difference
        self.symtraj = SymbolicTrajectory(model.K, model.nx, model.nu, model.ns, 
                                          model.xlabels, model.ulabels, model.slabels)
        
        # Discretization parameters
        self.Ak_param = cvx.Parameter((model.nx, model.nx, model.K-1))
        self.Bkm_param = cvx.Parameter((model.nx, model.nu, model.K-1))
        self.Bkp_param = cvx.Parameter((model.nx, model.nu, model.K-1))
        self.Sk_param = cvx.Parameter((model.nx, model.ns, model.K-1))
        self.wk_param = cvx.Parameter((model.nx, model.K-1))

        # Virtual control term
        self.vk = cvx.Variable((model.K-1, model.nx))
 
        # Construct constraints, should make sure these are actually lists
        self.constraints = model.get_constraints(self.symtraj)
        self.stcs = model.get_STCs(self.symtraj) 
        for stc in self.stcs: self.constraints += stc.constraint
        
        # Dynamics constraints, is there a vectorized way to do this maybe?
        self.constraints += [self.symtraj.x[k+1] == self.Ak_param[..., k] @ self.symtraj.x[k] 
                             + self.Bkm_param[..., k] @ self.symtraj.u[k]
                             + self.Bkp_param[..., k] @ self.symtraj.u[k+1]
                             + self.Sk_param[..., k] @ self.symtraj.s
                             + self.wk_param[..., k] + self.vk[k]
                             for k in range(model.K-1)]

        # Construct objective
        self.J_vc = model.w_vc * cvx.norm1(self.vk)
        self.J_tr = 0.0
        for k in range(model.K):
            dz_k = cvx.hstack([self.symtraj.ds, self.symtraj.dx[k], self.symtraj.du[k]])
            self.J_tr += cvx.quad_form(dz_k, model.w_tr) #dz_k^T * W_tr * dz_k

        self.objective = cvx.Minimize(self.J_vc + self.J_tr) + model.get_objective(self.symtraj)
        
        # Construct problem
        self.problem = cvx.Problem(self.objective, self.constraints)

        # Get intial trajectory
        self.history = [model.init_trajectory(Trajectory.zeros(model.K, model.nx, model.nu, model.ns, 
                                                               model.xlabels, model.ulabels, model.slabels))]

    def solve(self, **kwargs):
        solved = False
        for _ in range(self.m.max_iters):
            prev_traj = self.history[-1]

            self.symtraj.set_parameters(prev_traj)

            for stc in self.stcs: stc.update(prev_traj)

            Ak, Bkm, Bkp, Sk, wk = self.integrator.discretize(prev_traj)
            self.Ak_param.value = Ak 
            self.Bkm_param.value = Bkm
            self.Bkp_param.value = Bkp
            self.Sk_param.value = Sk
            self.wk_param.value = wk

            self.problem.solve(solver='CLARABEL', **kwargs)

            self.history.append(self.symtraj.get_result(update_params=False))

            if self.J_vc.value < self.m.tol_vc and self.J_tr.value < self.m.tol_tr:
                solved = True
                break

        return solved