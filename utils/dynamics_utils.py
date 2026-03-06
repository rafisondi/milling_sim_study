
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# -------------------------
# Functions
# -------------------------

def load_macro_from_npz(npz_path, axes=(0, 1)):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Linearized model not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    Lambda = data["Lambda_xyz"]  # (3,3)
    Dx     = data["Dx_xyz"]      # (3,3)
    Kx     = data["Kx_xyz"]      # (3,3)

    a0, a1 = axes
    M2 = Lambda[np.ix_([a0, a1], [a0, a1])]
    C2 = Dx[np.ix_([a0, a1], [a0, a1])]
    K2 = Kx[np.ix_([a0, a1], [a0, a1])]
    extras = {k: data[k] for k in data.files if k not in ["Lambda_xyz", "Dx_xyz", "Kx_xyz"]}
    return M2, C2, K2, extras


# -------------------------
# Dataclasses
# -------------------------
@dataclass
class MacroOscillatorParams:
    M: np.ndarray  # (2,2) total mass matrix (macro + micro)
    C: np.ndarray  # (2,2) damping matrix
    K: np.ndarray  # (2,2) stiffness matrix

    def __post_init__(self):
        for name, arr in [("M", self.M), ("K", self.K), ("C", self.C)]:
            if arr.shape != (2, 2):
                raise ValueError(f"{name} must have shape (2,2), got {arr.shape}")
            
            
# -------------------------
# Single (macro+micro) oscillator dynamics
# State: [x(2); xdot(2)]  (4x1)
# Input: u (2x1) direct force
# Disturbance: f_milling (2x1)
# -------------------------
def dynamics(t, state, p_osc: MacroOscillatorParams, u, f_ext):
    x    = state[0:2]
    xdot = state[2:4]
    # M xddot = -C xdot - K x + u + f_ext
    rhs  = -p_osc.C @ xdot - p_osc.K @ x + u + f_ext
    xddot = np.linalg.solve(p_osc.M, rhs)
    return np.vstack((xdot, xddot))

def RungeKutta4(func, t, x, dt, *args):
    k1 = func(t, x, *args)
    k2 = func(t + dt/2, x + dt/2*k1, *args)
    k3 = func(t + dt/2, x + dt/2*k2, *args)
    k4 = func(t + dt,   x + dt*k3,   *args)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)