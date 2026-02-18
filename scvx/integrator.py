import numpy as np
from scipy.integrate import solve_ivp
from model import Model
from trajectory import Trajectory

class Integrator:
    def __init__(self, m: Model):
        self.m = m
        self.K = m.K
        self.ns = m.ns
        self.nx = m.nx
        self.nu = m.nu

        self.F, self.A, self.B, self.S = m.get_dynamics()
        self.dt = 1.0/(self.K-1)
        # y0 contains: 
        #   x: (nx)
        #   Phi/Ak: (nx, nx)
        #   Psi/Phi^-1: (nx, nx)
        #   Bk-: (nx, nu)
        #   Bk+: (nx, nu)
        #   Sk: (nx, ns)
        #   wk: (nx)
        # All this K-1 times over. y0 is a vector of length (K-1)(2 + 2*nx + 2*nu + ns)*nx
        
        # We have to fill in x_bar for each K-1 slot and init Phi/Psi to idents
        # Need a scalable way to slice y 

        self.sizes = [
            self.nx,             # x is (nx,)
            self.nx**2,          # Phi is (nx, nx) -> flattened to (nx^2,)
            self.nx**2,          # Psi
            self.nx*self.nu,     # Bkm
            self.nx*self.nu,     # Bkp
            self.nx*self.ns,     # Sk
            self.nx              # wk
        ]
        self.split_idxs = np.cumsum(self.sizes)[:-1]

        self.N = np.sum(self.sizes)

    def piecewise(self):
        """Integrate the nonlinear dynamics piecewise across each subinterval."""
        pass

    def continuous(self):
        """Integrate the nonlinear dynamics continously across the time horizon."""
        pass

    def discretize(self, traj: Trajectory):
        """Integrate to compute the discretization of the dynamics."""
        s = np.tile(traj.sigma, (self.K, 1)) # (K, ns)
        x = traj.x     # (K, nx)
        u = traj.u     # (K, nu)

        # init y, note: lambda funcs expect column states
        y0 = np.zeros((self.K-1, self.N))

        # split outputs into shapes of (K-1, shape)
        parts = np.split(y0, self.split_idxs, axis=-1)
        x0 = parts[0]
        Phi0 = parts[1]
        Psi0 = parts[2]

        x0[:] = x[:-1] # Set all x0 

        # init Phi and Psi to identity 
        I_flat = np.eye(self.nx).flatten()
        Phi0[:] = I_flat
        Psi0[:] = I_flat

        # call integrator, Vf will be shape (K-1, N)
        sol = solve_ivp(fun=self._dydt,
                       t_span=(0, self.dt),
                       y0=y0.flatten(),
                       args=(u, s))
        
        # Extract final state
        Vf = sol.y[:, -1].reshape((self.K-1, self.N))

        parts = np.split(Vf, self.split_idxs, axis=-1)
        
        # x_prop = parts[0] # Propagated state x(t_{k+1}), useful for debug
        Phif = parts[1].reshape((self.K-1, self.nx, self.nx))
        # Psi_f = parts[2] # Not needed for discrete matrices
        Bm_int = parts[3].reshape((self.K-1, self.nx, self.nu))
        Bp_int = parts[4].reshape((self.K-1, self.nx, self.nu))
        S_int = parts[5].reshape((self.K-1, self.nx, self.ns))
        w_int = parts[6].reshape((self.K-1, self.nx))
        
        Ak = Phif 
        Bkm = Ak @ Bm_int
        Bkp = Ak @ Bp_int
        Sk = Ak @ S_int
        wk = Ak @ w_int
        return Ak, Bkm, Bkp, Sk, wk

    def _dydt(self, t, yi, u, s):
        # Due to my more general formulation we actually have:
        # x' = F(old) + A(x - x_old) + B(u - u_old) + S(s-s_old)
        # x' = Ax + Bu + Ss + F(old) -Ax_old - Bu_old - Ss_old

        # FOH on controls
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        ui = alpha*u[:-1] + beta*u[1:] # (K-1, nu) output

        si = s[:-1] # Need to ensure this matches xi in dimension (K-1, ns)

        # Split outputs into shapes of (K-1, shape)
        # (xi, Phii, Psii, Bmi, Bpi, Si, wi)
        parts = np.split(yi.reshape((self.K-1, self.N)), self.split_idxs, axis=-1)
        xi = parts[0]
        Phii = parts[1].reshape((self.K-1, self.nx, self.nx))
        Psii = parts[2].reshape((self.K-1, self.nx, self.nx))
        # Bmi = parts[3].reshape((self.K-1, self.nx, self.nu))
        # Bpi = parts[4].reshape((self.K-1, self.nx, self.nu))
        # Si = parts[5].reshape((self.K-1, self.nx, self.ns))
        # wi = parts[6]

        # Get dynamics & jacobians, expect shapes of (shape, K-1) to vectorize
        # outputs are of shapes (nx, 1, K-1), (nx, nx, K-1), (nx, nu, K-1), (nx, ns, K-1)
        # Lambdified sympy functions
        Fi = np.moveaxis(self.F(si.T, xi.T, ui.T), -1, 0).reshape((self.K-1, self.nx)) # make shape (K-1, nx)
        Ai = np.moveaxis(self.A(si.T, xi.T, ui.T), -1, 0) # make shape (K-1, nx, nx)
        Bi = np.moveaxis(self.B(si.T, xi.T, ui.T), -1, 0) # make shape (K-1, nx, nu)
        Si = np.moveaxis(self.S(si.T, xi.T, ui.T), -1, 0) # make shape (K-1, nx, ns)

        dxi = Fi
        dPhi = Ai @ Phii
        dPsi = -Psii @ Ai
        dBm = Psii @ Bi * alpha
        dBp = Psii @ Bi * beta
        dS = Psii @ Si

        residual = Fi - (Ai @ xi + Bi @ ui + Si @ si)
        dw = Psii @ residual

        dPhi_flat = dPhi.reshape((self.K-1, -1))
        dPsi_flat = dPsi.reshape((self.K-1, -1))
        dBm_flat  = dBm.reshape((self.K-1, -1))
        dBp_flat  = dBp.reshape((self.K-1, -1))
        dS_flat   = dS.reshape((self.K-1, -1))
               
        # Stack column-wise
        dydt = np.hstack([dxi, dPhi_flat, dPsi_flat, dBm_flat, dBp_flat, dS_flat, dw])
        
        return dydt.flatten()

