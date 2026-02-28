from model import Model
from scvx import SCvxProblem
from dimmed_views import DimProp
from labeled_dim_view import LabeledDimArray
from labeled_views import LabeledArray, LabeledVariable, LabeledParameter
import sympy as sp
import cvxpy as cvx
import numpy as np

class QuadcopterFixedFT(Model):
    nx = 6 # pos, vel
    nu = 3 # trust vec
    ns = 0 # Nothing involved in physics

    xlabels = ['px', 'py', 'pz', 'vx', 'vy', 'vz']
    ulabels = ['Tx', 'Ty', 'Tz']

    max_iters = 10
    tol_vc = 1e-7
    tol_tr = 5e-3
    w_vc = 1e4
    w_tr = np.diag((nx + nu + ns)*[0.5])

    K = 20

    def __init__(self):
        # REQUIRED STUFF

        # MODEL STUFF
        # Scales
        self.m_scale = 0.3
        self.r_scale = 10

        # Convex Relaxation for Thrust Magnitude (K)
        self.sigma = cvx.Variable(self.K, nonneg=True)
        
        # Constants
        self.kD = 0.5 # condensed drag coeff
        self.theta_max = np.deg2rad(45) # max thrust tilt
        self.tf = 3.0 # final time
        self.m = DimProp(0.3, self.m_scale) # mass
        self.g_val = DimProp(-9.81, self.r_scale) # gravity
        self.T_min = DimProp(1.0, self.m_scale*self.r_scale) # thrust min
        self.T_max = DimProp(4.0, self.m_scale*self.r_scale) # thrust max

        # Boundary conditions
        self.x_ic = DimProp(np.array([0, 0,  0, 0, 0.5, 0]),      self.r_scale) # p_ic, v_ic
        self.x_fc = DimProp(np.array([0, 10, 0, 0, 0.5, 0]),      self.r_scale) # p_fc, v_fc
        self.u_ic = DimProp(np.array([-self.m*self.g_val, 0, 0]), self.r_scale*self.m_scale)
        self.u_fc = DimProp(np.array([-self.m*self.g_val, 0, 0]), self.r_scale*self.m_scale)

        # Obstacles
        self.obstacles = [
            {'p_obs': DimProp(np.array([0.0, 3.0,  0.45]), self.r_scale), 
             'R_obs': DimProp(1.0, self.r_scale)},
            {'p_obs': DimProp(np.array([0.0, 7.0, -0.45]), self.r_scale), 
             'R_obs': DimProp(1.0, self.r_scale)},
        ]
        self.n_obs = len(self.obstacles)

        self.dt = self.tf / (self.K - 1)

    def init_trajectory(self, traj):
        for k in range(self.K):
            alpha = k / (self.K-1)
            traj.x[k] = (1 - alpha) * self.x_ic.nondim + alpha * self.x_fc.nondim
            traj.u[k] = (1 - alpha) * self.u_ic.nondim + alpha * self.u_fc.nondim
        return traj
    
    def get_dynamics(self):
        s = sp.Matrix(sp.symbols([], real=True))
        x = sp.Matrix(sp.symbols(self.xlabels, real=True))
        u = sp.Matrix(sp.symbols(self.ulabels, real=True))
        v = sp.Matrix(x[3:6])

        g_vec = sp.Matrix([self.g_val.nondim, 0, 0])
        a = 1/self.m.nondim * u - self.kD * v.norm(2)*v + g_vec # use nonsdim values
        F = sp.simplify(self.tf*sp.Matrix([v, a])) # scale by time
        A = sp.simplify(F.jacobian(x))
        B = sp.simplify(F.jacobian(u))
        # S = sp.simplify(F.jacobian(s))
        S = sp.Matrix([])

        F_func = sp.lambdify((s, x, u), F, 'numpy')
        A_func = sp.lambdify((s, x, u), A, 'numpy')
        B_func = sp.lambdify((s, x, u), B, 'numpy')
        S_func = sp.lambdify((s, x, u), S, 'numpy')

        return F_func, A_func, B_func, S_func

    def get_constraints(self, traj):

        constraints = [
            # Initial and Final State
            traj.x[0] == self.x_ic.nondim,
            traj.x[-1] == self.x_fc.nondim,
            traj.u[0] == self.u_ic.nondim,
            traj.u[-1] == self.u_fc.nondim,

            # Planar motion
            traj.x['px'] == 0,

            # Thrust magnitude
            cvx.norm(traj.u, p=2, axis=1) <= self.sigma,

            # Thrust limits
            self.sigma >= self.T_min.nondim,
            self.sigma <= self.T_max.nondim,

            # Tilt angle
            np.cos(self.theta_max)*self.sigma <= traj.u['Tx'],
        ]

        # Obstacles (non-convex constraints)
        #  h \approx h(zbar) + dh/dz|zbar*dz where h <= 0
        # h = s_obs = R_obs - ||p_old - p_obs||_2 <= 0
        # dhdz|zbar = [-(p_old-p_obs) / ||p_old-p_obs||_2, 0]
        # Obstacles h(zbar) + dhdz|zbar @ dzbar <= 0
        p_last = traj.x_last['px':'pz'] # (K, 3)
        p = traj.x['px':'pz']           # (K, 3)
        
        for j, obs in enumerate(self.obstacles):
            p_obs = obs['p_obs'].nondim # (3)
            R_obs = obs['R_obs'].nondim

            diff = p_last - p_obs # (K, 3)
            dist = cvx.norm(diff, p=2, axis=1) # (K)
            h = R_obs - dist # (K)

            dhdz = -diff/dist # (K, 3)
            linear_term = cvx.sum(
                cvx.multiply(dhdz, p - p_last), 
                axis=1
            )

            constraints += [h + linear_term <= 0]
        return constraints

    def get_objective(self, traj):
        return cvx.Minimize(cvx.sum(self.sigma) * self.dt)