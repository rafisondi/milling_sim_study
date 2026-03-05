import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import control as ct

# Milling simulation for process forces
from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *


# -------------------------
# Dataclasses
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
            
@dataclass
class WorkpieceGeometry:
    p1: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    p2: np.ndarray = field(default_factory=lambda: np.array([1000, 0]))
    p3: np.ndarray = field(default_factory=lambda: np.array([1000, 76]))
    p4: np.ndarray = field(default_factory=lambda: np.array([0, 76]))

    @property
    def xy(self):
        return np.array([self.p1, self.p2, self.p3, self.p4]).T
            
@dataclass
class ToolParameters:
    diameter: float = 20.0
    radial_engagement: float = 8.0
    axial_depth: float = 1.0
    number_of_teeth: int = 8

    @property
    def radius(self):
        return self.diameter / 2.0

    def tool_offset(self):
        offset_y = self.radius - self.radial_engagement

        if offset_y > 0:
            offset_x = -np.sqrt(self.radius**2 - offset_y**2)
        else:
            offset_x = -self.radius

        return offset_x, offset_y
    
@dataclass
class ProcessParameters:
    rpm: float = 8000
    feed_per_tooth: float = 0.15
    delta_deg: float = 1.0
    T_end: float = 0.5

    def omega(self):
        return -self.rpm * 2 * m.pi / 60

    def rot_angle_per_step(self):
        return self.delta_deg * (2 * m.pi / 360)

@dataclass
class MillingSimulationParameters:
    workpiece:  WorkpieceGeometry   = field(default_factory=WorkpieceGeometry)
    tool:       ToolParameters      = field(default_factory=ToolParameters)
    process:    ProcessParameters   = field(default_factory=ProcessParameters)

    # -------- Derived geometry --------
    def initial_tool_center(self):
        ox, oy = self.tool.tool_offset()

        return np.array([
            self.workpiece.p4[0] + ox,
            self.workpiece.p4[1] + oy
        ])

    # -------- Derived simulation values --------
    def feed_speed(self):
        omega = self.process.omega()
        return -omega / (2 * m.pi) * self.process.feed_per_tooth * self.tool.number_of_teeth

    def sample_frequency(self):
        time_step_p_rev = (2 * m.pi) / self.process.rot_angle_per_step()
        return time_step_p_rev * self.process.rpm / 60

    def dt(self):
        return 1.0 / self.sample_frequency()

    def time_vector(self):
        return np.arange(0.0, self.process.T_end, self.dt())

    def samples_per_revolution(self):
        rev_period = 1 / (self.process.rpm / 60.0)
        return int(self.sample_frequency() * rev_period)



def load_macro_from_npz(npz_path, axes=(0, 1)):
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


class MeasurementModel:
    def __init__(self, noise_std = None):
        self.C = np.zeros((10,10))
        # x
        self.C[0:2, 0:2] = np.eye(2)
        # xdot
        self.C[2:4, 2:4] = np.eye(2)
        # p
        self.C[4:6, 4:6] = np.eye(2)
        # pdot
        self.C[6:8, 6:8] = np.eye(2)
        # F_Act 
        self.C[8:10, 8:10] = np.eye(2)
        
        # For now we assume full state feedback without noise
        self.noise_std = noise_std
        
        # track tcp global B = x + p
        Cref = np.zeros((2,10))
        Cref[:, 0:2] = np.eye(2)   # x
        Cref[:, 4:6] = np.eye(2)   # p  
        self.C_tcp = Cref

    def output_full_state(self, state):
        y =  self.C @ state
        if self.noise_std is not None:
            y += np.random.randn(*y.shape) * self.noise_std
        return y

    def output_TCP(self, state):
        y =  self.C_tcp @ state
        if self.noise_std is not None:
            y += np.random.randn(*y.shape) * self.noise_std
        return y 
    
    
# -------------------------
# Controller functions
# -------------------------

def compute_Lr(A, B, K, C, rcond_warn=1e10):

    """
    We assume the following as restriction to the use of this feedforward term
    
    1. no poles at zero --> Guarantees invertability of G0
    2. equal number of control input to tracking variables
    
    """
    
    Acl = A - B @ K

    # X = (Acl)^{-1} B  via solve (better than inv)
    X = np.linalg.solve(Acl, B)      # (n,m)
    G0 = -C @ X                      # (ny,m)

    ny, m = G0.shape
    if ny != m:
        raise ValueError(f"compute_Lr requires square tracking (ny==m). Got ny={ny}, m={m}")

    # Solve G0 Lr = I
    Lr = np.linalg.solve(G0, np.eye(m))
    return Lr, G0

class DynamicsModel:
    """
    Builds the 10-state linear model:
        s = [x(2), xdot(2), p(2), pdot(2), F(2)]
        u = F_cmd (2)

    Also computes an LQR gain K at init.
    """

    def __init__(self, *, M, C, K, M_micro, B_micro, tau, Q=None, R=None):
        self.M = np.asarray(M)
        self.C = np.asarray(C)
        self.K = np.asarray(K)
        self.M_micro = np.asarray(M_micro)
        self.B_micro = np.asarray(B_micro)
        self.tau = float(tau)

        self.A, self.B = self._build_state_space()


        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

        # Compute LQR
        self.K_lqr, self.S_lqr, self.E_lqr = ct.lqr(self.A, self.B, self.Q, self.R)


    def _build_state_space(self):
        # checks
        for name, mat, shp in [
            ("M", self.M, (2,2)),
            ("C", self.C, (2,2)),
            ("K", self.K, (2,2)),
            ("M_micro", self.M_micro, (2,2)),
            ("B_micro", self.B_micro, (2,2)),
        ]:
            if mat.shape != shp:
                raise ValueError(f"{name} must be shape {shp}, got {mat.shape}")

        I = np.eye(2)

        S = np.linalg.inv(self.M)
        T = np.linalg.inv(self.M_micro)

        A = np.zeros((10,10))
        B = np.zeros((10, 2))

        # State ordering: [x(2), xdot(2), p(2), pdot(2), F(2)]
        # xdot = xdot
        A[0:2, 2:4] = I

        # xddot = -S K x - S C xdot + S B pdot - S F
        A[2:4, 0:2]  = -S @ self.K
        A[2:4, 2:4]  = -S @ self.C
        A[2:4, 6:8]  =  S @ self.B_micro
        A[2:4, 8:10] = -S

        # pdot = pdot
        A[4:6, 6:8] = I

        # pddot = +S K x + S C xdot - (S+T)B pdot + (S+T)F
        A[6:8, 0:2]  =  S @ self.K
        A[6:8, 2:4]  =  S @ self.C
        A[6:8, 6:8]  = -(S + T) @ self.B_micro
        A[6:8, 8:10] =  (S + T)

        # Fdot = -(1/tau)F + (1/tau)u
        A[8:10, 8:10] = -(1.0/self.tau) * I
        B[8:10, 0:2]  =  (1.0/self.tau) * I

        return A, B


# -------------------------
# Plant integration
# -------------------------
def RungeKutta4(func, t, x, dt, *args):
    k1 = func(t, x, *args)
    k2 = func(t + dt/2, x + dt/2*k1, *args)
    k3 = func(t + dt/2, x + dt/2*k2, *args)
    k4 = func(t + dt,   x + dt*k3,   *args)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def dynamics(t, state, p_m: MotorParams, p_macro: MacroActuatorParams, p_micro: MicroActuatorParams, v_cmd, f_ext = 0):
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
    f_int   = f_motor + f_damp + f_ext

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



def sat(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

class LineTrajectory2D:
    def __init__(self, p0, p1, T):
        """
        p0, p1: array-like shape (2,) or (2,1)
        T: duration in seconds
        """
        self.p0 = np.array(p0, dtype=float).reshape(2,1)
        self.p1 = np.array(p1, dtype=float).reshape(2,1)
        self.T  = float(T)
        if self.T <= 0:
            raise ValueError("T must be > 0")

        self.dp = self.p1 - self.p0

    @staticmethod
    def _sigma(tau):
        # quintic smoothstep
        return 10*tau**3 - 15*tau**4 + 6*tau**5

    @staticmethod
    def _dsigma_dtau(tau):
        return 30*tau**2 - 60*tau**3 + 30*tau**4

    @staticmethod
    def _d2sigma_dtau2(tau):
        return 60*tau - 180*tau**2 + 120*tau**3

    def pos(self, t):
        tau = sat(t / self.T)
        s = self._sigma(tau)
        return self.p0 + s * self.dp

    def vel(self, t):
        tau = sat(t / self.T)
        ds = self._dsigma_dtau(tau) / self.T
        return ds * self.dp

    def acc(self, t):
        tau = sat(t / self.T)
        d2s = self._d2sigma_dtau2(tau) / (self.T**2)
        return d2s * self.dp

if __name__ == "__main__":
    



    
    # -------------------------
    # Simulation
    # -------------------------
    # Macro parameters from Pinocchio operational-space projection
    M2, C2, K2, extras = load_macro_from_npz("./data/linearized_operational_space_xyz.npz", axes=(0, 1))  # (x,y)
    print(f"Inertia matrix: {M2}")
    print(f"Damping matrix: {C2}")
    print(f"Stiffnesss matrix: {K2}")
    wp_p1 = np.array([-50.0, -50.0])
    wp_p2 = np.array([1000 , -50.0])
    wp_p3 = np.array([1000.0, 0.0])
    wp_p4 = np.array([0.0,    0.0])

    # -------------------------
    # Initiate dataclasses 
    # -------------------------

    p_macro = MacroActuatorParams(M=M2, K=K2, C=C2)
    p_m     = MotorParams(R=18.2, L=100e-3, Ke=89.0, Kt=157.0, Vmax=540.0) # /np.sqrt(2)
    p_micro = MicroActuatorParams(M=np.eye(2)*80.0, B=np.eye(2)*1.0)

    workpiece = WorkpieceGeometry(wp_p1, wp_p2 ,wp_p3 ,wp_p4)
    params_milling = MillingSimulationParameters( workpiece= workpiece)

    xy_workpiece = params_milling.workpiece.xy
    xy_tool_center_0 = params_milling.initial_tool_center()
    samples_per_rev = params_milling.samples_per_revolution()
    feed_speed = params_milling.feed_speed()

    
    # Controller Parameters
    f_c = 500.0
    wc = 2*np.pi*f_c
    tau = 1.0 / wc

    xmax   = 1e-3     # m
    xdmax  = 1e0      # m/s
    pmax   = 1e-4    # m
    pdmax  = 1e-2     # m/s
    umax   = 1000.0     # N (force command)
    
    # LQR weights
    Q = np.zeros((10,10))
    Q[0:2,0:2] = (1/xmax**2)*np.eye(2)   #x
    Q[2:4,2:4] = (1/xdmax**2)*np.eye(2)  #xdot
    Q[4:6,4:6] = (1/pmax**2)*np.eye(2)   #p
    Q[6:8,6:8] = (1/pdmax**2)*np.eye(2)  #pdot
    Q[8:10,8:10] = (1/umax**2)*np.eye(2) #if wanted F

    R = (1/umax**2)*np.eye(2)
    
    dynamics_model = DynamicsModel(
        M=M2, C=C2, K=K2,
        M_micro=p_micro.M,
        B_micro=p_micro.B,
        tau=tau,
        Q=Q, R=R
    )
        
    A = dynamics_model.A
    B = dynamics_model.B
    K = dynamics_model.K_lqr
    
    measurement_model = MeasurementModel()
    Cref = measurement_model.C_tcp
    Acl = A - B @ K
    Lr, G0 = compute_Lr(A, B, K, Cref)
    
    print("G0=\n", G0)
    print("cond(G0)=", np.linalg.cond(G0))
    print("||Lr||=", np.linalg.norm(Lr))
    # print("u_ff at final ref =", (Lr @ traj.pos(Tmove)).ravel())
    
    
    # -------------------------
    # Init Process Class
    # -------------------------
    milling_process                     = milling_workpiece(params_milling.workpiece.xy)
    milling_process.diameter_end_mill   = params_milling.tool.diameter
    milling_process.number_of_teeth     = params_milling.tool.number_of_teeth
    milling_process.radius_tool         = params_milling.tool.radius
    milling_process.slice_height        = params_milling.tool.axial_depth

    # Simulation running rates
    dt_i = 1e-4   # inner loop (10 kHz)
    dt_o = 1e-3   # outer loop (1 kHz)

    # Total Time 
    T    = 2.0
    N_i = int(T / dt_i)
    outer_every = max(1, int(round(dt_o / dt_i)))
    
    feed_speed = params_milling.feed_speed() * 1e-3 #[m]
    time_to_goal = 0.1 / feed_speed
    p0 = (0.0, 0.0)
    p1 = (0.1, 0.0)   # meters
    Tmove = time_to_goal       # seconds (pick >= 0.5–2s depending on stiffness)

    traj = LineTrajectory2D(p0, p1, Tmove)
    # print("u_ff at final ref =", (Lr @ traj.pos(Tmove)).ravel())

    # Current-loop tuning
    # Note: f_c=500 Hz, assuming a 10 kHz update motor rate.
    f_c = 500.0
    wc  = 2*np.pi*f_c
    Kp  = p_m.L * wc
    Ki  = p_m.R * wc
    i_ctrl = CurrentPI(Kp=Kp, Ki=Ki, Vmax=p_m.Vmax)

    # State
    state = np.zeros((10,1))
    x = np.zeros_like(state)
    t = 0.0

    # Logs
    t_hist = np.zeros(N_i)
    x_hist = np.zeros((N_i,2))
    p_hist = np.zeros((N_i,2))
    i_hist = np.zeros((N_i,2))
    v_hist = np.zeros((N_i,2))
    i_des_hist = np.zeros((N_i,2))
    F_cmd_hist = np.zeros((N_i,2))
    # ----- BEFORE SIMULATION LOOP -----
    r_hist = np.zeros((N_i,2))

    # Outer command variables
    i_des = np.zeros((2,1))

    # milling_process.show_plot()
        
    # ts = np.linspace(0, 1.2*Tmove, 400)
    # P = np.hstack([traj.pos(tt) for tt in ts])  # 2xN
    # plt.figure()
    # plt.plot(ts, P[0,:], label="x_ref")
    # plt.plot(ts, P[1,:], label="y_ref")
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel("t [s]")
    # plt.ylabel("position [m]")
    # plt.title("TCP reference")
    # plt.show()

    # Fx_amp = 10.0
    # Fy_amp = 10.0
    # fx = 5.0   # Hz
    # fy = 5.0   # Hz
    milling_forces = np.zeros((2,1))
    for n in range(N_i):
        # Outer loop: Update Inputs based on Controller action
        if (n % outer_every) == 0:
            y_state = measurement_model.output_full_state(x) # Full state feedback
            y_tcp  = measurement_model.output_TCP(x) # TCP Position Global

            r_tcp = traj.pos(t)       # (2,1)
            u_cmd = -K @ x + Lr @ r_tcp
            i_des = u_cmd / p_m.Kt
        
        # if (n % outer_every) == 0:
        #     Fx = Fx_amp * np.sin(2*np.pi*fx*t)
        #     Fy = Fy_amp * np.cos(2*np.pi*fy*t)  
        #     F_cmd = np.array([[Fx],[Fy]])        # (2,1)
        #     i_des = F_cmd / p_m.Kt
            
        # Inner loop: Current controller -> voltage
        i_meas = state[4:6]
        p_dot  = state[8:10]
        v_cmd  = i_ctrl.step(i_des=i_des, i_meas=i_meas, p_dot=p_dot,
                            R=p_m.R, Ke=p_m.Ke, dt=dt_i)

        # Current external force due to simulation
        orientation = 0.0 + t * params_milling.process.omega() # current tool orientation [rad]
        milling_process.erase_step(state[0:2] + state[2:4], orientation, -1, t)
        if n >1 :
            milling_forces =  milling_process.total_milling_force[0:2,np.newaxis]
            
        # Integrate plant at inner  # canoot really see this
        state = RungeKutta4(dynamics, t, state, dt_i, p_m, p_macro, p_micro, v_cmd , milling_forces[0:2])

        # Original state var: state = [x(2); p(2); i(2); x_dot(2); p_dot(2)]  shape (10,1)
        # I have access to 
            # pos x
            # pos p
            # xdot
            # pdot
            # Force acting
        x_pos  = state[0:2]
        p_pos  = state[2:4]
        i_cur  = state[4:6]
        x_dot  = state[6:8]
        p_dot  = state[8:10]
        F_act  = p_m.Kt * i_cur

        # LQR state ordering: [x; xdot; p; pdot; F]
        x = np.vstack([x_pos, x_dot, p_pos, p_dot, F_act])
        
        # History
        t_hist[n] = t
        x_hist[n,:] = state[0:2,0]
        p_hist[n,:] = state[2:4,0]
        i_hist[n,:] = state[4:6,0]
        v_hist[n,:] = v_cmd[:,0]
        i_des_hist[n,:] = i_des[:,0]
        F_cmd_hist[n,:] = u_cmd[:,0]
        r_hist[n,:] = r_tcp[:,0]


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
    rd = r_hist[::step]

    plt.figure()
    plt.plot(Bd[:,0], Bd[:,1], label="B trajectory (macro)")
    plt.plot(Cd[:,0], Cd[:,1], label="C trajectory (micro)")
    plt.scatter([0],[0], marker="s", label="Ground (A)")
    # plt.axis("equal")
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

    

    # =========================
    # Stacked Base Position Plot
    # =========================
    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(td, Bd[:,0], label="Base X")
    ax[0].set_ylabel("x [m]")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(td, Bd[:,1], label="Base Y")
    ax[1].set_ylabel("y [m]")
    ax[1].set_xlabel("t [s]")
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle("Base (Macro) Position vs Time")


    # =========================
    # Stacked TCP Position Plot (with reference)
    # =========================
    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(td, Cd[:,0], label="TCP X")
    ax[0].plot(td, rd[:,0], '--', label="Reference X")
    ax[0].set_ylabel("x [m]")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(td, Cd[:,1], label="TCP Y")
    ax[1].plot(td, rd[:,1], '--', label="Reference Y")
    ax[1].set_ylabel("y [m]")
    ax[1].set_xlabel("t [s]")
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle("TCP Position Tracking")


    # =========================
    # OPTIONAL: TCP Tracking Error
    # =========================
    err = Cd - rd

    plt.figure()
    plt.plot(td, err[:,0], label="TCP error X")
    plt.plot(td, err[:,1], label="TCP error Y")
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("Tracking error [m]")
    plt.title("TCP Tracking Error")
    plt.legend()


    plt.show()
