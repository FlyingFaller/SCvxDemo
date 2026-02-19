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
        # State-major order to avoid transposing in _dydt
        s = np.tile(traj.sigma, (self.K, 1)).T # (ns, K)
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
              s: np.ndarray):
        """Calculates the massive derivative of the discretization problem thing."""
        # FOH on controls
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        ui = alpha*u[:, :-1] + beta[:, 1:] # (nu, K-1)

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

    # def discretize(self, traj: Trajectory):
    #     """Integrate to compute the discretization of the dynamics."""
    #     s = np.tile(traj.sigma, (self.K, 1)) # (K, ns)
    #     x = traj.x     # (K, nx)
    #     u = traj.u     # (K, nu)

    #     # init y, note: lambda funcs expect column states
    #     y0 = np.zeros((self.K-1, self.N))

    #     # split outputs into shapes of (K-1, shape)
    #     parts = np.split(y0, self.split_idxs, axis=-1)
    #     x0 = parts[0]
    #     Phi0 = parts[1]
    #     Psi0 = parts[2]

    #     x0[:] = x[:-1] # Set all x0 

    #     # init Phi and Psi to identity 
    #     I_flat = np.eye(self.nx).flatten()
    #     Phi0[:] = I_flat
    #     Psi0[:] = I_flat

    #     # call integrator, Vf will be shape (K-1, N)
    #     sol = solve_ivp(fun=self._dydt,
    #                    t_span=(0, self.dt),
    #                    y0=y0.flatten(),
    #                    args=(u, s))
        
    #     # Extract final state
    #     Vf = sol.y[:, -1].reshape((self.K-1, self.N))

    #     parts = np.split(Vf, self.split_idxs, axis=-1)
        
    #     # x_prop = parts[0] # Propagated state x(t_{k+1}), useful for debug
    #     Phif = parts[1].reshape((self.K-1, self.nx, self.nx))
    #     # Psi_f = parts[2] # Not needed for discrete matrices
    #     Bm_int = parts[3].reshape((self.K-1, self.nx, self.nu))
    #     Bp_int = parts[4].reshape((self.K-1, self.nx, self.nu))
    #     S_int = parts[5].reshape((self.K-1, self.nx, self.ns))
    #     w_int = parts[6].reshape((self.K-1, self.nx))
        
    #     Ak = Phif 
    #     Bkm = Ak @ Bm_int
    #     Bkp = Ak @ Bp_int
    #     Sk = Ak @ S_int
    #     wk = Ak @ w_int
    #     return Ak, Bkm, Bkp, Sk, wk

    # def _dydt(self, t, yi, u, s):
    #     # Due to my more general formulation we actually have:
    #     # x' = F(old) + A(x - x_old) + B(u - u_old) + S(s-s_old)
    #     # x' = Ax + Bu + Ss + F(old) -Ax_old - Bu_old - Ss_old

    #     # FOH on controls
    #     alpha = (self.dt - t) / self.dt
    #     beta = t / self.dt
    #     ui = alpha*u[:-1] + beta*u[1:] # (K-1, nu) output

    #     si = s[:-1] # Need to ensure this matches xi in dimension (K-1, ns)

    #     # Split outputs into shapes of (K-1, shape)
    #     # (xi, Phii, Psii, Bmi, Bpi, Si, wi)
    #     parts = np.split(yi.reshape((self.K-1, self.N)), self.split_idxs, axis=-1)
    #     xi = parts[0]
    #     Phii = parts[1].reshape((self.K-1, self.nx, self.nx))
    #     Psii = parts[2].reshape((self.K-1, self.nx, self.nx))
    #     # Bmi = parts[3].reshape((self.K-1, self.nx, self.nu))
    #     # Bpi = parts[4].reshape((self.K-1, self.nx, self.nu))
    #     # Si = parts[5].reshape((self.K-1, self.nx, self.ns))
    #     # wi = parts[6]

    #     # Get dynamics & jacobians, expect shapes of (shape, K-1) to vectorize
    #     # outputs are of shapes (nx, 1, K-1), (nx, nx, K-1), (nx, nu, K-1), (nx, ns, K-1)
    #     # Lambdified sympy functions
    #     Fi = np.moveaxis(self.F(si.T, xi.T, ui.T), -1, 0).reshape((self.K-1, self.nx)) # make shape (K-1, nx)
    #     Ai = np.moveaxis(self.A(si.T, xi.T, ui.T), -1, 0) # make shape (K-1, nx, nx)
    #     Bi = np.moveaxis(self.B(si.T, xi.T, ui.T), -1, 0) # make shape (K-1, nx, nu)
    #     Si = np.moveaxis(self.S(si.T, xi.T, ui.T), -1, 0) # make shape (K-1, nx, ns)

    #     dxi = Fi
    #     dPhi = Ai @ Phii
    #     dPsi = -Psii @ Ai
    #     dBm = Psii @ Bi * alpha
    #     dBp = Psii @ Bi * beta
    #     dS = Psii @ Si

    #     residual = Fi - (Ai @ xi + Bi @ ui + Si @ si)
    #     dw = Psii @ residual

    #     dPhi_flat = dPhi.reshape((self.K-1, -1))
    #     dPsi_flat = dPsi.reshape((self.K-1, -1))
    #     dBm_flat  = dBm.reshape((self.K-1, -1))
    #     dBp_flat  = dBp.reshape((self.K-1, -1))
    #     dS_flat   = dS.reshape((self.K-1, -1))
               
    #     # Stack column-wise
    #     dydt = np.hstack([dxi, dPhi_flat, dPsi_flat, dBm_flat, dBp_flat, dS_flat, dw])
        
    #     return dydt.flatten()

