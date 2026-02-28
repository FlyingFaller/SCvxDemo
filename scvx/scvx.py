from model import Model
from integrator import Integrator
# from stc import StateTriggeredConstraint
from trajectory import Trajectory, CvxTrajectory
# from labeled_views import LabeledArray
import cvxpy as cvx
# import numpy as np

class SCvxProblem:

    def __init__(self, model: Model):
        self.m = model

        # Allows for calculation of Ak Bmk, Bpk, Sk, Wk from Trajectory obj
        self.integrator = Integrator(model)

        # Stores current trajectory, past trajectory parameters, and difference
        self.cvxtraj = CvxTrajectory(model.K, model.nx, model.nu, 
                                     model.xlabels, model.ulabels, 
                                     model.aux_vars)
        
        # Discretization parameters
        self.Ak_param = cvx.Parameter((model.nx, model.nx, model.K-1))
        self.Bkm_param = cvx.Parameter((model.nx, model.nu, model.K-1))
        self.Bkp_param = cvx.Parameter((model.nx, model.nu, model.K-1))
        self.wk_param = cvx.Parameter((model.nx, model.K-1))

        self.Sk_params: dict[str, cvx.Parameter] = {}
        for i, name in enumerate(self.integrator.aux_names):
            p = self.integrator.aux_p[i]
            self.Sk_params[name] = cvx.Parameter((self.m.nx, p, self.m.K - 1))

        # Virtual control term
        self.vk = cvx.Variable((model.K-1, model.nx))
        self.J_vc = model.w_vc * cvx.norm1(self.vk)

        # Trust region, we assume diagonal W_tr matrices, may update later but this is simple
        self.J_tr = 0.0
        for k in range(self.m.K):
            self.J_tr += cvx.vdot(self.m.w_tr_x, cvx.square(self.cvxtraj.dx[k]))
            self.J_tr += cvx.vdot(self.m.w_tr_u, cvx.square(self.cvxtraj.du[k]))

        for var_meta in self.m.aux_vars:
            if var_meta.w_tr is not None:
                diff = getattr(self.cvxtraj, f"d{var_meta.name}")
                
                if var_meta.local:
                    self.J_tr += cvx.vdot(var_meta.w_tr, cvx.square(diff))
                else:
                    self.J_tr += self.m.K * cvx.vdot(var_meta.w_tr, cvx.square(diff))
 
        # Construct constraints, should make sure these are actually lists
        self.constraints = model.get_constraints(self.cvxtraj)
        self.stcs = model.get_STCs(self.cvxtraj) 
        for stc in self.stcs: self.constraints += stc.constraint
        
        # Dynamics linearization constraint
        for k in range(self.m.K - 1):
            dyn_expr = (self.Ak_param[:, :, k] @ self.cvxtraj.x[k] +
                        self.Bkm_param[:, :, k] @ self.cvxtraj.u[k] +
                        self.Bkp_param[:, :, k] @ self.cvxtraj.u[k+1] +
                        self.wk_param[:, k] + self.vk[k])

            # Inject the auxiliary variables that dictate the dynamics
            for name in self.integrator.aux_names:
                cvx_var: cvx.Variable = getattr(self.cvxtraj, name)
                # We know these are global due to the Integrator's enforcement
                # Flatten the variable to match the (nx, p) shape of the Sk parameter
                dyn_expr += self.Sk_params[name][:, :, k] @ cvx_var.flatten(order='C')

            self.constraints += [self.cvxtraj.x[k+1] == dyn_expr]

        # Construct objective
        self.objective = cvx.Minimize(self.J_vc + self.J_tr) + model.get_objective(self.cvxtraj)
        
        # Construct problem
        self.problem = cvx.Problem(self.objective, self.constraints)

        # Setup history, empty until solved? Should we init traj here? idk
        self.history = []

    def solve(self, **kwargs):
        self.history = [self.m.init_trajectory(Trajectory.zeros(self.m.K, self.m.nx, self.m.nu, 
                                                                self.m.xlabels, self.m.ulabels, 
                                                                self.m.aux_vars))]
        for iter_idx in range(self.m.max_iters):
            prev_traj = self.history[-1]

            # Update parameters with previous traj
            self.cvxtraj.set_parameters(prev_traj)

            # Compute discretization
            Ak, Bkm, Bkp, wk, Sk_dict = self.integrator.discretize(prev_traj)

            # Update discretization parameters
            self.Ak_param.value = Ak
            self.Bkm_param.value = Bkm
            self.Bkp_param.value = Bkp
            self.wk_param.value = wk

            for name, Sk in Sk_dict.items():
                self.Sk_params[name].value = Sk

            # Update STCs and evaluate the trigger funcs
            for stc in self.stcs:
                stc.update(prev_traj) 

            # Solve!!!
            self.problem.solve(**kwargs)
            
            # Check if solved
            if self.problem.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                print(f"Solver failed at iteration {iter_idx+1} with status: {self.problem.status}")
                break

            # Extract resultant trajectory
            new_traj = self.cvxtraj.get_result(update_params=False)
            self.history.append(new_traj)

            # Check convergence 
            if self.J_vc.value < self.m.tol_vc and self.J_tr.value < self.m.tol_tr:
                print(f"Converged in {iter_idx+1} iterations. (VC: {self.J_vc.value:.2e}, TR: {self.J_tr.value:.2e})")
                break
        else:
            print("Failed to converge within maximum iterations.")