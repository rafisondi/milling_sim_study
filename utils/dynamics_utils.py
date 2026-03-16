
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# -------------------------
# Functions
# -------------------------

def load_macro_from_npz(npz_path, axes=None):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Linearized model not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    Lambda = data["Lambda_xyz"]  # (3,3)
    Dx     = data["Dx_xyz"]      # (3,3)
    Kx     = data["Kx_xyz"]      # (3,3)

    if axes is None:
        axis_idx = np.arange(Lambda.shape[0], dtype=int)
    else:
        axis_idx = np.asarray(tuple(axes), dtype=int)

    M2 = Lambda[np.ix_(axis_idx, axis_idx)]
    C2 = Dx[np.ix_(axis_idx, axis_idx)]
    K2 = Kx[np.ix_(axis_idx, axis_idx)]
    extras = {k: data[k] for k in data.files if k not in ["Lambda_xyz", "Dx_xyz", "Kx_xyz"]}
    return M2, C2, K2, extras


# -------------------------
# Dataclasses
# -------------------------
@dataclass
class MacroOscillatorParams:
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray

    def __post_init__(self):
        for name, arr in [("M", self.M), ("K", self.K), ("C", self.C)]:
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"{name} must be square, got shape {arr.shape}")
        if not (self.M.shape == self.C.shape == self.K.shape):
            raise ValueError(f"M, C, and K must have matching shapes, got {self.M.shape}, {self.C.shape}, {self.K.shape}")
            
            
# -------------------------
# Single (macro+micro) oscillator dynamics
# State: [x(n); xdot(n)]
# Input: u (n x 1) direct force
# Disturbance: f_ext (n x 1)
# -------------------------
def dynamics(t, state, p_osc: MacroOscillatorParams, u, f_ext):
    n = p_osc.M.shape[0]
    x = state[0:n]
    xdot = state[n : 2 * n]
    rhs = -p_osc.C @ xdot - p_osc.K @ x + u + f_ext
    xddot = np.linalg.solve(p_osc.M, rhs)
    return np.vstack((xdot, xddot))

def RungeKutta4(func, t, x, dt, *args):
    k1 = func(t, x, *args)
    k2 = func(t + dt/2, x + dt/2*k1, *args)
    k3 = func(t + dt/2, x + dt/2*k2, *args)
    k4 = func(t + dt,   x + dt*k3,   *args)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
