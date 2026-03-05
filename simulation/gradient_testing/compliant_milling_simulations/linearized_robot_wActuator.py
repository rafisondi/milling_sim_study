import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

def sat(x, lo, hi):
    return np.clip(x, lo, hi)

# -------------------------
# Params
# -------------------------
@dataclass
class MotorParams:
    R: float
    L: float
    Ke: float
    Kt: float
    Vmax: float

@dataclass
class MacroActuatorParams:
    M: np.ndarray
    K: np.ndarray
    C: np.ndarray
    def __post_init__(self):
        for name, arr in [("M", self.M), ("K", self.K), ("C", self.C)]:
            if arr.shape != (2,2):
                raise ValueError(f"{name} must have shape (2,2), got {arr.shape}")

@dataclass
class MicroActuatorParams:
    M: np.ndarray
    B: np.ndarray
    def __post_init__(self):
        for name, arr in [("M", self.M), ("B", self.B)]:
            if arr.shape != (2,2):
                raise ValueError(f"{name} must have shape (2,2), got {arr.shape}")

# -------------------------
# Plant integration
# -------------------------
def RungeKutta4(func, t, x, dt, *args):
    k1 = func(t, x, *args)
    k2 = func(t + dt/2, x + dt/2*k1, *args)
    k3 = func(t + dt/2, x + dt/2*k2, *args)
    k4 = func(t + dt,   x + dt*k3,   *args)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def dynamics(t, state, p_m: MotorParams, p_macro: MacroActuatorParams, p_micro: MicroActuatorParams, v_cmd):
    """
    state = [x(2); p(2); i(2); x_dot(2); p_dot(2)]  shape (10,1)
    v_cmd shape (2,1)
    """
    x     = state[0:2]
    p     = state[2:4]
    i     = state[4:6]
    x_dot = state[6:8]
    p_dot = state[8:10]

    # Internal force on micro mass C (and equal-opposite on macro mass B)
    f_motor = p_m.Kt * i           # (2,1)
    f_damp  = -p_micro.B @ p_dot   # (2,1) relative damper B<->C
    f_int   = f_motor + f_damp

    # Ground compliance acting on macro mass B
    f_ground = -p_macro.K @ x - p_macro.C @ x_dot

    # Macro accel: M x_ddot = f_ground - f_int
    x_ddot = np.linalg.solve(p_macro.M, f_ground - f_int)

    # Relative accel: p_ddot = a_C - a_B, with a_C = M_micro^{-1} f_int
    p_ddot = np.linalg.solve(p_micro.M, f_int) - x_ddot

    # Electrical: L i_dot = v - R i - Ke * p_dot
    i_dot = (v_cmd - (p_m.R * i) - (p_m.Ke * p_dot)) / p_m.L

    return np.vstack((x_dot, p_dot, i_dot, x_ddot, p_ddot))

# -------------------------
# Current PI controller
# -------------------------
@dataclass
class CurrentPI:
    Kp: float
    Ki: float
    Vmax: float
    kaw: float = None
    integ: np.ndarray = None

    def __post_init__(self):
        if self.integ is None:
            self.integ = np.zeros((2,1))
        if self.kaw is None:
            # simple, reasonable default
            self.kaw = 1.0 / max(self.Kp, 1e-9)

    def step(self, i_des, i_meas, p_dot, R, Ke, dt):
        """
        Vectorized 2-axis PI current controller with back-EMF decoupling and anti-windup.
        """
        e = i_des - i_meas  # (2,1)

        # V_cmd_unsaturated = R*i(t) +  V_EMF (Feedforward term) + Kp*E (P-Feedback) + Inegrations part
        V_cmd_unsaturated = (R * i_meas) + (Ke * p_dot) + (self.Kp * e) + (self.Ki * self.integ)

        # saturate to drive limit
        v_sat = np.clip(V_cmd_unsaturated, -self.Vmax, self.Vmax)

        # anti-windup back-calculation
        self.integ += (e + self.kaw * (v_sat - V_cmd_unsaturated)) * dt

        return v_sat

# -------------------------
# Simulation
# -------------------------
# Macro parameters (your original idea)
m = 50.0
f0 = 10.0
omega = 2*np.pi*f0

k_ground = m*omega**2
zeta = 0.10
c_ground = 2*zeta*m*omega

p_macro = MacroActuatorParams(M=np.eye(2)*m, K=np.eye(2)*k_ground, C=np.eye(2)*c_ground)
p_m     = MotorParams(R=18.2, L=100e-3, Ke=89.0, Kt=157.0, Vmax=540.0/np.sqrt(2))
p_micro = MicroActuatorParams(M=np.eye(2)*1.0, B=np.eye(2)*1.0)

# Simulation running rates
dt_i = 1e-4   # inner loop (10 kHz)
dt_o = 1e-3   # outer loop (1 kHz)

# Total Time 
T    = 5.0
N_i = int(T / dt_i)
outer_every = max(1, int(round(dt_o / dt_i)))

# Current-loop tuning
# Note: f_c=500 Hz, assuming a 10 kHz update motor rate.
f_c = 500.0
wc  = 2*np.pi*f_c
Kp  = p_m.L * wc
Ki  = p_m.R * wc
i_ctrl = CurrentPI(Kp=Kp, Ki=Ki, Vmax=p_m.Vmax)

# State
state = np.zeros((10,1))
t = 0.0

# Logs
t_hist = np.zeros(N_i)
x_hist = np.zeros((N_i,2))
p_hist = np.zeros((N_i,2))
i_hist = np.zeros((N_i,2))
v_hist = np.zeros((N_i,2))
i_des_hist = np.zeros((N_i,2))
F_cmd_hist = np.zeros((N_i,2))

# Outer command variables
i_des = np.zeros((2,1))

# Trajectory definition (both axes oscillate)
Fx_amp = 10.0
Fy_amp = 10.0
fx = 10.0   # Hz
fy = 10.0   # Hz (change to 1.2 or similar if you want Lissajous)

for n in range(N_i):

    # Outer loop: Update Inputs based on Controller action
    if (n % outer_every) == 0:
        Fx = Fx_amp * np.cos(2*np.pi*fx*t)
        Fy = Fy_amp * np.sin(2*np.pi*fy*t)  
        F_cmd = np.array([[Fx],[Fy]])        # (2,1)
        i_des = F_cmd / p_m.Kt

    # Inner loop: Current controller -> voltage
    i_meas = state[4:6]
    p_dot  = state[8:10]
    v_cmd  = i_ctrl.step(i_des=i_des, i_meas=i_meas, p_dot=p_dot,
                         R=p_m.R, Ke=p_m.Ke, dt=dt_i)

    # Integrate plant at inner rate
    state = RungeKutta4(dynamics, t, state, dt_i, p_m, p_macro, p_micro, v_cmd)

    # Log
    t_hist[n] = t
    x_hist[n,:] = state[0:2,0]
    p_hist[n,:] = state[2:4,0]
    i_hist[n,:] = state[4:6,0]
    v_hist[n,:] = v_cmd[:,0]
    i_des_hist[n,:] = i_des[:,0]
    F_cmd_hist[n,:] = F_cmd[:,0]

    t += dt_i

# -------------------------
# Plots
# -------------------------
B = x_hist                 # macro point B in world
B = x_hist                 # macro point B in world
C = x_hist + p_hist        # micro point C in world

# Downsample for cleaner plots
step = 20
Bd = B[::step]
Cd = C[::step]
td = t_hist[::step]
vd = v_hist[::step]
fc = F_cmd_hist[::step]
i_c = i_hist[::step]

plt.figure()
plt.plot(Bd[:,0], Bd[:,1], label="B trajectory (macro)")
plt.plot(Cd[:,0], Cd[:,1], label="C trajectory (micro)")
plt.scatter([0],[0], marker="s", label="Ground (A)")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("2D Trajectory")
plt.legend()

plt.figure()
plt.plot(td, vd[:,0], label="Vx (motor x)")
plt.plot(td, vd[:,1], label="Vy (motor y)")
plt.grid(True)
plt.xlabel("t [s]")
plt.ylabel("Voltage [V]")
plt.title("Motor Voltages")
plt.legend()

plt.figure()
plt.plot(td, i_c[:,0] * p_m.Kt, label="Fx (motor x)")
plt.plot(td, i_c[:,1] * p_m.Kt, label="Fy (motor y)")
plt.grid(True)
plt.xlabel("t [s]")
plt.ylabel("Force [N]")
plt.title("Motor Forces")
plt.legend()


plt.show()
