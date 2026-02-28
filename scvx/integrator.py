import numpy as np
from scipy.integrate import solve_ivp
from model import Model
from trajectory import Trajectory
from labeled_views import LabeledArray

class Integrator:
    def __init__(self, m: Model):
        self.m = m
        self.K = m.K
        self.nx = m.nx
        self.nu = m.nu

        # Unpack the dynamics and default aux_jacs to an empty dict if not provided
        dyn_returns = m.get_dynamics()
        self.F, self.A, self.B = dyn_returns[:3]
        self.aux_jacs = dyn_returns[3] if len(dyn_returns) > 3 else {}
        
        # Aux variables in dynamics, expected order for args
        # type: list[str], list[Callable]
        self.aux_names = list(self.aux_jacs.keys())
        self.aux_jac_funcs = list(self.aux_jacs.values())
        
        # Store metadata for aux variables that are in dynamics, unordered!
        # type: dict[str, AuxVariable]
        self.aux_meta = {var.name: var for var in m.aux_vars if var.name in self.aux_names}

        # Precompute loop variables for maximum ODE hot-loop performance
        self.aux_p = [int(np.prod(self.aux_meta[name].shape)) for name in self.aux_names]
        self.aux_idx = list(range(6, 6 + len(self.aux_names)))
        self.aux_out_shapes = [(self.nx, p, self.K-1) for p in self.aux_p]

        # Aux variables must be "global" in dynamics. If they are local they are better 
        # classified as state or control variables.
        for name, var in self.aux_meta.items():
            if var.local:
                raise ValueError(f"AuxVariable '{name}' is marked as local but affects the continuous dynamics. "
                                 "Local time-varying variables that dictate the dynamics must be formulated as "
                                 "control variables.")

        self.dt = 1.0/(self.K-1)
        self.xlabels = m.xlabels

        # y0 contains K-1 copies of: 
        #   x: (nx)
        #   Phi/Ak: (nx, nx)
        #   Psi/Phi^-1: (nx, nx)
        #   Bk-: (nx, nu)
        #   Bk+: (nx, nu)
        #   wk: (nx)
        #   ... plus any dynamically registered auxiliary variable sensitivities ...
        
        self.sizes = [
            self.nx,             # x
            self.nx**2,          # Phi
            self.nx**2,          # Psi
            self.nx*self.nu,     # Bkm
            self.nx*self.nu,     # Bkp
            self.nx              # wk
        ]
        
        # Add allocation for aux vars 
        for p in self.aux_p: self.sizes.append(self.nx * p)

        self.split_idxs = np.cumsum(self.sizes)[:-1]
        self.N = np.sum(self.sizes) # Total allocation size for each K integration problem

    def integrate_multiple_shooting(self, traj: Trajectory) -> LabeledArray:
        """Integrate the nonlinear dynamics piecewise across each subinterval (Multiple Shooting)."""
        x = traj.x.T                           # (nx, K)
        u = traj.u.T                           # (nu, K)
 
        x0 = x[:, :-1] # (nx, K-1)
        xf = np.zeros_like(x) # (nx, K)
        xf[:, 0] = x[:, 0] # Set first element
        
        # Pre-slice controls for the entire horizon
        u0 = u[:, :-1] # (nu, K-1)
        u1 = u[:, 1:]  # (nu, K-1)
        
        aux_args = [getattr(traj, name) for name in self.aux_names]
        
        sol = solve_ivp(fun = self._dxdt,
                        t_span = (0, self.dt),
                        y0=x0.ravel(),
                        args=(u0, u1, aux_args))
                        
        xf[:, 1:] = sol.y[:, -1].reshape((self.nx, self.K-1)) # Set last elements
        return LabeledArray(xf.T, labels=self.xlabels)

    def integrate_single_shooting(self, traj: Trajectory) -> LabeledArray:
        """Integrate the nonlinear dynamics sequentially across the time horizon (Single Shooting)."""
        x = traj.x
        u = traj.u
        xf = np.zeros_like(x) # (K, nx)
        xf[0] = x[0]
        
        aux_args = [getattr(traj, name) for name in self.aux_names]
        
        for k in range(self.K-1):
            # Pass controls as column vectors using [:, None]
            u0_col = u[k][:, None]
            u1_col = u[k+1][:, None]
            
            sol = solve_ivp(fun = self._dxdt,
                            t_span = (0, self.dt),
                            y0=xf[k], 
                            args=(u0_col, u1_col, aux_args))
                            
            xf[k+1] = sol.y[:, -1]

        return LabeledArray(xf, labels=self.xlabels)

    def _dxdt(self, 
              t: float, 
              xi_flat: np.ndarray, 
              u0: np.ndarray, 
              u1: np.ndarray, 
              aux_args: list) -> np.ndarray:
        """Function for state derivative."""
        # FOH on controls
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        ui = alpha * u0 + beta * u1
        
        # Reshape to either (nx, 1) or (nx, K-1)
        xi = xi_flat.reshape((self.nx, -1))

        # Evaluate dynamics and return flattened
        Fi = self.F(xi, ui, *aux_args)
        return Fi.ravel()

    def discretize(self, traj: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Integrate to compute the discretization of the dynamics."""
        x = traj.x.T # (nx, K)
        u = traj.u.T # (nu, K)

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

        # Pre-slice parameters and controls to eliminate slices in hot loops
        u0 = u[:, :-1]
        u1 = u[:, 1:]
        aux_args = [getattr(traj, name) for name in self.aux_names]

        # Integrate
        sol = solve_ivp(fun = self._dydt,
                        t_span=(0, self.dt),
                        y0=y0.ravel(),
                        args=(u0, u1, aux_args))
        
        # Collect output and split
        yf = sol.y[:, -1].reshape((self.N, self.K-1))
        parts = np.split(yf, self.split_idxs, axis=0)
        
        Phif   = parts[1].reshape((self.nx, self.nx, self.K-1))
        Bm_int = parts[3].reshape((self.nx, self.nu, self.K-1))
        Bp_int = parts[4].reshape((self.nx, self.nu, self.K-1))
        w_int  = parts[5].reshape((self.nx, self.K-1))

        # Post left multiply by Ak to generate discretization
        Ak = Phif # (nx, nx, K-1)
        Bkm = np.einsum('ijk,jlk->ilk', Ak, Bm_int) # (nx, nu, K-1)
        Bkp = np.einsum('ijk,jlk->ilk', Ak, Bp_int) # (nx, nu, K-1)
        wk  = np.einsum('ijk,jk->ik', Ak, w_int)    # (nx, K-1)

        # Dynamically extract and multiply registered Aux sensitivities
        Sk_dict = {}
        for i, name in enumerate(self.aux_names):
            S_int = parts[self.aux_idx[i]].reshape(self.aux_out_shapes[i])
            Sk_dict[name] = np.einsum('ijk,jlk->ilk', Ak, S_int)

        return Ak, Bkm, Bkp, wk, Sk_dict

    def _dydt(self, 
              t: float, 
              yi_flat: np.ndarray, 
              u0: np.ndarray, 
              u1: np.ndarray, 
              aux_args: list) -> np.ndarray:
        """Calculates the massive derivative of the discretization problem."""
        # FOH on controls
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        ui = alpha * u0 + beta * u1 # (nu, K-1)

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
        dw   = parts_out[5] # (nx, K-1)

        # Avoid dynamic loops or property fetches within hot-loop
        dS = [parts_out[idx].reshape(shape) for idx, shape in zip(self.aux_idx, self.aux_out_shapes)]

        # Get dynamics, reshaping F to (nx, K-1) in case User passes flat (nx,) for scalar shooting
        Fi = self.F(xi, ui, *aux_args).reshape((self.nx, self.K-1))
        Ai = self.A(xi, ui, *aux_args) # (nx, nx, K-1)
        Bi = self.B(xi, ui, *aux_args) # (nx, nu, K-1)

        # COMPUTE DERIVATIVES
        dxi[:] = Fi
        
        # dPhi = Ai @ Phii
        np.einsum('ijk,jlk->ilk', Ai, Phii, out=dPhi)
        
        # dPsi = -dPsi @ Ai
        np.einsum('ijk,jlk->ilk', Psii, Ai, out=dPsi)
        np.negative(dPsi, out=dPsi) 
        
        # Bm and Bp terms
        np.einsum('ijk,jlk->ilk', Psii, Bi, out=dBp) # store to dBp temporarily
        np.multiply(dBp, alpha, out=dBm) 
        np.multiply(dBp, beta, out=dBp)
        
        # Core Affine terms
        Ax = np.einsum('ijk,jk->ik', Ai, xi)
        Bu = np.einsum('ijk,jk->ik', Bi, ui)
        
        # Dynamically evaluate auxiliary dependencies only if they exist
        if self.aux_names:
            Ss = np.zeros((self.nx, self.K-1))
            
            for i, jac_func in enumerate(self.aux_jac_funcs):
                Si = jac_func(xi, ui, *aux_args)
                
                # dSi = Psii @ Si
                np.einsum('ijk,jlk->ilk', Psii, Si, out=dS[i])
                
                # S @ s term for the wk residual calculation
                # we ravel aux_args just incase the user does something weird 
                # like define a multi-dimensional global var
                Ss += np.einsum('ijk,j->ik', Si, np.ravel(aux_args[i]))
                
            residual = Fi - (Ax + Bu + Ss)
        else:
            residual = Fi - (Ax + Bu)

        # dw = Psii @ residual
        np.einsum('ijk,jk->ik', Psii, residual, out=dw)
        
        return dydt.ravel()