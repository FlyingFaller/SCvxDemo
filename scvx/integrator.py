import numpy as np
from scipy.integrate import solve_ivp
from model import Model
from trajectory import Trajectory
from labeled_views import LabeledArray

class Integrator:
    def __init__(self, m: Model):
        # self.m = m
        self.K = m.K
        self.ns = m.ns
        self.nx = m.nx
        self.nu = m.nu

        self.F, self.A, self.B, self.S = m.get_dynamics()
        self.dt = 1.0/(self.K-1)

        self.xlabels = m.xlabels

        # y0 contains: 
        #   x: (nx)
        #   Phi/Ak: (nx, nx)
        #   Psi/Phi^-1: (nx, nx)
        #   Bk-: (nx, nu)
        #   Bk+: (nx, nu)
        #   Sk: (nx, ns)
        #   wk: (nx)
        # All this K-1 times over. y0 is a vector of length N = (K-1)(2 + 2*nx + 2*nu + ns)*nx
        
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

    def integrate_multiple_shooting(self, traj: Trajectory) -> LabeledArray:
        """Integrate the nonlinear dynamics piecewise across each subinterval (Multiple Shooting)."""
        s = np.tile(traj.s, (self.K, 1)).T     # (ns, K)
        x = traj.x.T                           # (nx, K)
        u = traj.u.T                           # (nu, K)
 
        x0 = x[:, :-1] # (nx, K-1)
        xf = np.zeros_like(x) # (nx, K)
        xf[:, 0] = x[:, 0] # Set first element
        
        # Start and end controls for the interval
        u0 = u[:, :-1] # (nu, K-1)
        u1 = u[:, 1:]  # (nu, K-1)
        
        sol = solve_ivp(fun = self._dxdt,
                        t_span = (0, self.dt),
                        y0=x0.ravel(),
                        args=(u0, u1, s[:, :-1]))
        xf[:, 1:] = sol.y[:, -1].reshape((self.nx, self.K-1)) # Set last elements
        return LabeledArray(xf.T, labels=self.xlabels)

    def integrate_single_shooting(self, traj: Trajectory) -> LabeledArray:
        """Integrate the nonlinear dynamics sequentially across the time horizon (Single Shooting)."""
        s = traj.s[:, None] # Use [:, None] to columnate
        x = traj.x
        u = traj.u
        xf = np.zeros_like(x) # (K, nx)
        xf[0] = x[0]
        
        for k in range(self.K-1):
            # Pass controls as column vectors using [:, None]
            u0_col = u[k][:, None]
            u1_col = u[k+1][:, None]
            
            sol = solve_ivp(fun = self._dxdt,
                            t_span = (0, self.dt),
                            y0=xf[k], 
                            args=(u0_col, u1_col, s))
            xf[k+1] = sol.y[:, -1]

        return LabeledArray(xf, labels=self.xlabels)

    def _dxdt(self, 
              t: float, 
              xi_flat: np.ndarray, 
              u0: np.ndarray, 
              u1: np.ndarray, 
              s: np.ndarray) -> np.ndarray:
        """Function for state derivative."""
        # FOH on controls
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        ui = alpha * u0 + beta * u1
        
        # Reshape to either (nx, 1) or (nx, K-1)
        xi = xi_flat.reshape((self.nx, -1))

        # Evaluate dynamics and return flattened
        Fi = self.F(s, xi, ui)
        return Fi.ravel()

    def discretize(self, traj: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrate to compute the discretization of the dynamics."""
        # State-major order to avoid transposing in _dydt
        s = np.tile(traj.s, (self.K, 1)).T     # (ns, K)
        x = traj.x.T                           # (nx, K)
        u = traj.u.T                           # (nu, K)

        # Init integration vector, (N, K-1)
        y0 = np.zeros((self.N, self.K-1))

        # Split rows apart, split should create a view importantly
        parts = np.split(y0, self.split_idxs, axis=0)
        x0, Phi0, Psi0 = parts[0], parts[1], parts[2] # (shape, K-1)

        # Init STM and states in y0
        x0[:] = x[:, :-1]
        I_flat = np.eye(self.nx).flatten()[:, None] # Creates a dim at idx 1 (column vector)
        Phi0[:] = I_flat
        Psi0[:] = I_flat

        # Integrate. flatten creates a new array but ravel doesn't
        # slicing s so we don't have to do it inside _dydt
        sol = solve_ivp(fun = self._dydt,
                        t_span=(0, self.dt),
                        y0=y0.ravel(),
                        args=(u, s[:, :-1]))
        
        # Collect output and split
        yf = sol.y[:, -1].reshape((self.N, self.K-1))
        parts = np.split(yf, self.split_idxs, axis=0)
        Phif   = parts[1].reshape((self.nx, self.nx, self.K-1))
        Bm_int = parts[3].reshape((self.nx, self.nu, self.K-1))
        Bp_int = parts[4].reshape((self.nx, self.nu, self.K-1))
        S_int  = parts[5].reshape((self.nx, self.ns, self.K-1))
        w_int  = parts[6].reshape((self.nx, self.K-1))

        # Post left multiply by Ak to generate discretization
        Ak = Phif # (nx, nx, K-1)
        Bkm = np.einsum('ijk,jlk->ilk', Ak, Bm_int) # (nx, nu, K-1)
        Bkp = np.einsum('ijk,jlk->ilk', Ak, Bp_int) # (nx, nu, K-1)
        Sk  = np.einsum('ijk,jlk->ilk', Ak, S_int)  # (nx, ns, K-1)
        wk  = np.einsum('ijk,jk->ik', Ak, w_int)    # (nx, K-1)
        return Ak, Bkm, Bkp, Sk, wk

    def _dydt(self, 
              t: float, 
              yi_flat: np.ndarray, 
              u: np.ndarray, 
              s: np.ndarray) -> np.ndarray:
        """Calculates the massive derivative of the discretization problem thing."""
        # FOH on controls
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        ui = alpha*u[:, :-1] + beta*u[:, 1:] # (nu, K-1)

        # Reshape and split y into useful views
        yi = yi_flat.reshape((self.N, self.K-1))
        parts_in = np.split(yi, self.split_idxs, axis=0)
        xi = parts_in[0]
        Phii = parts_in[1].reshape((self.nx, self.nx, self.K-1))
        Psii = parts_in[2].reshape((self.nx, self.nx, self.K-1))

        # Allocate output and split
        dydt = np.empty((self.N, self.K-1))
        parts_out = np.split(dydt, self.split_idxs, axis=0)
        dxi  = parts_out[0] # (nx, K-1)
        dPhi = parts_out[1].reshape((self.nx, self.nx, self.K-1))
        dPsi = parts_out[2].reshape((self.nx, self.nx, self.K-1))
        dBm  = parts_out[3].reshape((self.nx, self.nu, self.K-1))
        dBp  = parts_out[4].reshape((self.nx, self.nu, self.K-1))
        dS   = parts_out[5].reshape((self.nx, self.ns, self.K-1))
        dw   = parts_out[6] # (nx, K-1)

        # Get dynamics, likely performance bottleneck here
        # Ouput is expected to be (nx, 1, K-1) so we reshape
        Fi = self.F(s, xi, ui).reshape((self.nx, self.K-1))
        Ai = self.A(s, xi, ui) # (nx, nx, K-1)
        Bi = self.B(s, xi, ui) # (nx, nu, K-1)
        Si = self.S(s, xi, ui) # (nx, ns, K-1)

        # COMPUTE DERIVATIVES!!!!
        dxi[:] = Fi
        # dPhi = Ai @ Phii
        np.einsum('ijk,jlk->ilk', Ai, Phii, out=dPhi)
        # dPsi = -dPsi @ Ai
        np.einsum('ijk,jlk->ilk', Psii, Ai, out=dPsi)
        np.negative(dPsi, out=dPsi) # Is this any faster?
        # Bm and Bp terms efficiently = Psi @ Bi * (alpha|beta)
        np.einsum('ijk,jlk->ilk', Psii, Bi, out=dBp) # store to dBp temporarily
        np.multiply(dBp, alpha, out=dBm) 
        np.multiply(dBp, beta, out=dBp)
        # dS = Psii @ Si
        np.einsum('ijk,jlk->ilk', Psii, Si, out=dS)
        # dw = Psii @ (F - (Ax + Bu + Ss)), potential bottleneck here too
        Ax = np.einsum('ijk,jk->ik', Ai, xi)
        Bu = np.einsum('ijk,jk->ik', Bi, ui)
        Ss = np.einsum('ijk,jk->ik', Si, s)
        residual = Fi - (Ax + Bu + Ss) # (nx, K-1)
        np.einsum('ijk,jk->ik', Psii, residual, out=dw)
        
        return dydt.ravel()