import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import math as m
import os
import sys
from pathlib import Path
from datetime import datetime


# Milling simulation for process forces
from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *


# ============================================================
# 1) Global Configuration
# ============================================================

# ---- Time / sampling
DT = 1e-4                      # [s]
SIMULATION_TIME_S = 1.0         # [s]
PRINT_EVERY_N = 1000

ENABLE_SPRING_DAMPER_DYNAMICS = False
ENABLE_LIVE_PLOTTING = False
SAVE_EXPERIMENT_DATA = True

# ---- Paths / output
LINEARIZED_MODEL_NPZ =  "data/linearized_operational_space_xyz.npz"
OUTPUT_DIR =   "data/experiments"

# ---- Workpiece geometry [mm]
WORKPIECE_P1_MM = np.array([0.0, 0.0])
WORKPIECE_P2_MM = np.array([50.0, 0.0])
WORKPIECE_P3_MM = np.array([50.0, 50.0])
WORKPIECE_P4_MM = np.array([0.0, 50.0])

# ---- Tool / cut parameters
TOOL_DIAMETER_MM = 20.0
RADIUS_MM = TOOL_DIAMETER_MM / 2.0
RADIAL_ENGAGEMENT_MM = 8.0
AXIAL_CUTTING_DEPTH_MM = 1.0
Z_TEETH = 4

# ---- Motion / spindle
SPINDLE_RPM = 4000.0
FEED_PER_TOOTH_MM = 0.15
SPINDLE_SPIN_DIR = -1           # +/- sign convention
DELTA_DEG = 1.0
APPROACH_DISTANCE_MM = TOOL_DIAMETER_MM // 2 + 0.1
OVERTRAVEL_DISTANCE_MM = TOOL_DIAMETER_MM
FEED_DIRECTION_MM = np.array([0.0, 1.0])  # from below bottom-left upward along left edge

# ---- Process duration
PROCESS_END_TIME_S = 1.5

APPROACH_MM = RADIUS_MM + 0.1   # how far before entry
OVERTRAVEL_MM = TOOL_DIAMETER_MM
FEED_DIR = np.array([0.0, 1.0]) 



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
    diameter: float = TOOL_DIAMETER_MM
    radial_engagement: float = RADIAL_ENGAGEMENT_MM
    axial_depth: float = AXIAL_CUTTING_DEPTH_MM
    number_of_teeth: int = Z_TEETH

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
    rpm: float = SPINDLE_RPM
    feed_per_tooth: float = FEED_PER_TOOTH_MM
    delta_deg: float = DELTA_DEG
    T_end: float = PROCESS_END_TIME_S
    spindle_spin_dir: int = SPINDLE_SPIN_DIR
    approach_distance_mm: float = APPROACH_DISTANCE_MM
    overtravel_distance_mm: float = OVERTRAVEL_DISTANCE_MM
    feed_direction_mm: np.ndarray = field(default_factory=lambda: FEED_DIRECTION_MM.copy())

    def omega(self):
        # Angular speed in rad/s. rpm -> rev/s -> rad/s, with configurable sign convention.
        return self.spindle_spin_dir * self.rpm * 2.0 * m.pi / 60.0

    def rot_angle_per_step(self):
        return self.delta_deg * (2 * m.pi / 360)

@dataclass
class MillingSimulationParameters:
    workpiece:  WorkpieceGeometry = field(default_factory=WorkpieceGeometry)
    tool:       ToolParameters = field(default_factory=ToolParameters)
    process:    ProcessParameters = field(default_factory=ProcessParameters)

    def engagement_entry_tool_center(self):
        ox, oy = self.tool.tool_offset()
        return np.array([self.workpiece.p1[0] + ox, self.workpiece.p1[1] + oy])

    def workpiece_length_x(self):
        return float(self.workpiece.p2[0] - self.workpiece.p1[0])

    def workpiece_height_y(self):
        return float(self.workpiece.p4[1] - self.workpiece.p1[1])

    def feed_direction_unit(self):
        d = np.asarray(self.process.feed_direction_mm, dtype=float).reshape(2)
        n = np.linalg.norm(d)
        if n <= 1e-12:
            raise ValueError("feed_direction_mm must be non-zero")
        return d / n

    def initial_tool_center(self):
        tool_at_entry = self.engagement_entry_tool_center()
        return tool_at_entry - self.process.approach_distance_mm * self.feed_direction_unit()

    def exit_tool_center(self):
        tool_at_entry = self.engagement_entry_tool_center()
        travel_length_mm = self.workpiece_height_y() + self.process.overtravel_distance_mm
        return tool_at_entry + travel_length_mm * self.feed_direction_unit()

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
# Trajectory
# -------------------------
def sat(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

class LineTrajectory2D:
    def __init__(self, p0, p1, T):
        self.p0 = np.array(p0, dtype=float).reshape(2,1)
        self.p1 = np.array(p1, dtype=float).reshape(2,1)
        self.T  = float(T)
        if self.T <= 0:
            raise ValueError("T must be > 0")
        self.dp = self.p1 - self.p0

    @staticmethod
    def _sigma(tau):
        return 10*tau**3 - 15*tau**4 + 6*tau**5

    def pos(self, t):
        tau = sat(t / self.T)
        s = self._sigma(tau)
        return self.p0 + s * self.dp


# -------------------------
# Single (macro+micro) oscillator dynamics
# State: [x(2); xdot(2)]  (4x1)
# Input: u (2x1) direct force
# Disturbance: f_milling (2x1)
# -------------------------
def oscillator_dynamics(t, state, p_osc: MacroOscillatorParams, u, f_ext):
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


def analytical_downmilling_force(tool_center_mm, orientation, params_milling, milling_process):
    phi_st, phi_ex = compute_entry_exit_angles_downmilling(
        r_xy_current=np.asarray(tool_center_mm, dtype=float).reshape(2),
        xy_workpiece=params_milling.workpiece.xy,
        radius_tool=params_milling.tool.radius,
        diameter_end_mill=params_milling.tool.diameter,
    )

    if (phi_ex - phi_st) <= 0.0:
        return np.zeros(2), np.zeros(2), (phi_st, phi_ex)

    Kt = milling_process.cutting_force_coefficient_Ktc
    Kr = milling_process.cutting_force_coefficient_Krc
    fz = params_milling.process.feed_per_tooth
    a_axial = params_milling.tool.axial_depth
    z_teeth = params_milling.tool.number_of_teeth

    F_inst = np.zeros(2, dtype=float)
    for j in range(z_teeth):
        phi_j = (orientation + 2.0 * m.pi * j / z_teeth) % (2.0 * m.pi)
        F_inst += analytical_force(
            phi=phi_j,
            phi_st=phi_st,
            phi_ex=phi_ex,
            b=a_axial,
            fz=fz,
            Kt=Kt,
            Kr=Kr,
        )

    F_zero = zero_order_force_analytical(
        c=fz,
        a=a_axial,
        N=z_teeth,
        phi_st=phi_st,
        phi_ex=phi_ex,
        Ktc=Kt,
        Krc=Kr,
    )

    return F_inst, F_zero, (phi_st, phi_ex)


def print_process_summary(params_milling, feed_speed_mm_s):
    feed_abs_mm_s = max(abs(feed_speed_mm_s), 1e-12)
    tool_start = params_milling.initial_tool_center()
    tool_entry = params_milling.engagement_entry_tool_center()
    tool_exit = params_milling.exit_tool_center()
    feed_dir = params_milling.feed_direction_unit()

    approach_distance_mm = max(0.0, float(np.dot(tool_entry - tool_start, feed_dir)))
    cut_and_exit_distance_mm = max(0.0, float(np.dot(tool_exit - tool_entry, feed_dir)))
    t_entry_s = approach_distance_mm / feed_abs_mm_s
    t_cut_s = cut_and_exit_distance_mm / feed_abs_mm_s
    t_total_s = t_entry_s + t_cut_s

    print("=== Milling Setup Summary ===")
    print(
        f"Workpiece (mm): width={params_milling.workpiece_length_x():.2f}, "
        f"height={(params_milling.workpiece.p4[1] - params_milling.workpiece.p1[1]):.2f}"
    )
    print(
        f"Tool: D={params_milling.tool.diameter:.2f} mm, R={params_milling.tool.radius:.2f} mm, "
        f"radial_eng={params_milling.tool.radial_engagement:.2f} mm, "
        f"axial_depth={params_milling.tool.axial_depth:.2f} mm, teeth={params_milling.tool.number_of_teeth}"
    )
    print(
        f"Process: rpm={params_milling.process.rpm:.1f}, "
        f"feed_per_tooth={params_milling.process.feed_per_tooth:.4f} mm/tooth, "
        f"feed_speed={feed_speed_mm_s:.3f} mm/s"
    )
    print(
        f"Path: approach={params_milling.process.approach_distance_mm:.2f} mm, "
        f"overtravel={params_milling.process.overtravel_distance_mm:.2f} mm, "
        f"feed_dir={feed_dir}"
    )
    print(
        f"Tool center x positions (mm): start={tool_start[0]:.3f}, "
        f"entry={tool_entry[0]:.3f}, exit={tool_exit[0]:.3f}"
    )
    print(
        f"Timing from feed: entry={t_entry_s:.3f} s, "
        f"cut+exit={t_cut_s:.3f} s, total={t_total_s:.3f} s"
    )

    return t_total_s


def tool_center_offset_for_radial_engagement(radius_mm, radial_eng_mm, feed_dir):
    
    n_surface = np.array([-feed_dir[1], feed_dir[0]])  # normal vector pointing outward from the cut surface
    
    offset_from_surface = radius_mm - radial_eng_mm
    offset_y = offset_from_surface * np.dot(n_surface, np.array([0, 1]))
    offset_x = offset_from_surface * np.dot(n_surface, np.array([1, 0]))
    return np.array([offset_x, offset_y])


def _float_token(value, decimals=3):
    token = f"{float(value):.{decimals}f}"
    return token.replace("-", "m").replace(".", "p")


def build_experiment_file_stem(params_milling, dt, total_time_s, spring_damper_enabled):
    feed_dir = params_milling.feed_direction_unit()
    pieces = [
        "linearized_milling",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        f"rpm{int(round(params_milling.process.rpm))}",
        f"fz{_float_token(params_milling.process.feed_per_tooth, decimals=3)}",
        f"z{params_milling.tool.number_of_teeth}",
        f"D{_float_token(params_milling.tool.diameter, decimals=1)}",
        f"ae{_float_token(params_milling.tool.radial_engagement, decimals=1)}",
        f"ap{_float_token(params_milling.tool.axial_depth, decimals=1)}",
        f"dt{_float_token(dt, decimals=5)}",
        f"T{_float_token(total_time_s, decimals=3)}",
        f"fdx{_float_token(feed_dir[0], decimals=2)}",
        f"fdy{_float_token(feed_dir[1], decimals=2)}",
        f"sd{int(bool(spring_damper_enabled))}",
    ]
    return "_".join(pieces)


def save_experiment_data(
    output_dir,
    params_milling,
    dt,
    total_time_s,
    spring_damper_enabled,
    t_hist,
    x_hist,
    u_hist,
    fmill_hist,
    fanalyt_hist,
    fzero_hist,
    tool_center_nominal_hist_mm,
    tool_center_actual_hist_mm,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = build_experiment_file_stem(
        params_milling=params_milling,
        dt=dt,
        total_time_s=total_time_s,
        spring_damper_enabled=spring_damper_enabled,
    )
    save_path = output_dir / f"{file_stem}.npz"

    np.savez_compressed(
        save_path,
        t_s=t_hist,
        x_m=x_hist,
        u_N=u_hist,
        f_mill_N=fmill_hist,
        f_analytical_inst_N=fanalyt_hist,
        f_analytical_zero_order_N=fzero_hist,
        tool_center_nominal_mm=tool_center_nominal_hist_mm,
        tool_center_actual_mm=tool_center_actual_hist_mm,
        dt_s=np.array([dt]),
        simulation_time_s=np.array([total_time_s]),
        spring_damper_enabled=np.array([int(bool(spring_damper_enabled))]),
        spindle_rpm=np.array([params_milling.process.rpm]),
        feed_per_tooth_mm=np.array([params_milling.process.feed_per_tooth]),
        spindle_spin_dir=np.array([params_milling.process.spindle_spin_dir]),
        tool_diameter_mm=np.array([params_milling.tool.diameter]),
        radial_engagement_mm=np.array([params_milling.tool.radial_engagement]),
        axial_depth_mm=np.array([params_milling.tool.axial_depth]),
        number_of_teeth=np.array([params_milling.tool.number_of_teeth]),
        feed_direction_mm=np.asarray(params_milling.process.feed_direction_mm, dtype=float).reshape(2),
        workpiece_xy_mm=params_milling.workpiece.xy,
    )
    return save_path


if __name__ == "__main__":

    # -------------------------
    # Load macro matrices (2D slice)
    # -------------------------
    M2, C2, K2, extras = load_macro_from_npz(LINEARIZED_MODEL_NPZ, axes=(0, 1))

    # -------------------------
    # Add micro mass into macro mass matrix (simple lumped mass)
    # -------------------------
    M_micro = np.eye(2) * 80.0
    Mtot = M2 + M_micro

    p_osc = MacroOscillatorParams(M=Mtot, C=C2, K=K2)

    # -------------------------
    # Milling setup
    # Workpiece in mm, start cutting from upper-left and feed in +x direction.
    # -------------------------
    wp_p1 = WORKPIECE_P1_MM
    wp_p2 = WORKPIECE_P2_MM
    wp_p3 = WORKPIECE_P3_MM
    wp_p4 = WORKPIECE_P4_MM
    workpiece = WorkpieceGeometry(wp_p1, wp_p2, wp_p3, wp_p4)
    params_milling = MillingSimulationParameters(workpiece=workpiece)

    milling_process = milling_workpiece(params_milling.workpiece.xy)
    milling_process.diameter_end_mill   = params_milling.tool.diameter
    milling_process.number_of_teeth     = params_milling.tool.number_of_teeth
    milling_process.radius_tool         = params_milling.tool.radius
    milling_process.slice_height        = params_milling.tool.axial_depth
    
    
    height_mm = float(wp_p4[1] - wp_p1[1])

    entry_center = wp_p1 + tool_center_offset_for_radial_engagement(RADIUS_MM, RADIAL_ENGAGEMENT_MM, FEED_DIR)

    start_center = entry_center - APPROACH_MM * FEED_DIR
    end_center   = entry_center + (height_mm + OVERTRAVEL_MM) * FEED_DIR    
    tool_center0_mm = start_center

    # tool_center0_mm = params_milling.initial_tool_center()
    feed_speed_mm_s = params_milling.feed_speed()
    required_process_time_s = print_process_summary(params_milling, feed_speed_mm_s)

    # Open-loop configuration: no controller/actuator force injection.
    u = np.zeros((2,1))

    # -------------------------
    # Simulation
    # -------------------------
    dt = DT       # single-rate simulation now (no inner loop)
    T  = max(SIMULATION_TIME_S, required_process_time_s + 0.1)
    N  = int(T / dt)

    state = np.zeros((4,1))  # [x; xdot] in m
    t = 0.0

    t_hist = np.zeros(N)
    x_hist = np.zeros((N,2))
    u_hist = np.zeros((N,2))
    fmill_hist = np.zeros((N,2))
    fanalyt_hist = np.zeros((N,2))
    fzero_hist = np.zeros((N,2))
    tool_center_nominal_hist_mm = np.zeros((N, 2))
    tool_center_actual_hist_mm = np.zeros((N, 2))

    milling_forces = np.zeros((2,1))
    
    print("Feed speed (mm/s):", feed_speed_mm_s)
    print(f"Simulation time (s): {T:.3f}")
    print("Spring-damper dynamics enabled:", ENABLE_SPRING_DAMPER_DYNAMICS)
    print("Live plotting enabled:", ENABLE_LIVE_PLOTTING)

    if ENABLE_LIVE_PLOTTING:
        plt.ion()
    live_plot_stride = max(1, PRINT_EVERY_N)
    for n in range(N):

        # Milling force disturbance
        # Physical spindle angle used by the analytical model.
        spindle_angle = t * params_milling.process.omega()
        tool_center_mm = (
            tool_center0_mm
            + FEED_DIR * feed_speed_mm_s * t
            + state[0:2, 0] * 1e3
        )
        tool_center_nominal_mm = tool_center0_mm + FEED_DIR * feed_speed_mm_s * t
        # Geometry engine in `erase_step` uses opposite rotation sign convention.
        milling_process.erase_step(tool_center_mm, -spindle_angle, params_milling.process.spindle_spin_dir, t)
        F_analytical, F_zero, _ = analytical_downmilling_force(
            tool_center_mm=tool_center_mm,
            orientation=spindle_angle,
            params_milling=params_milling,
            milling_process=milling_process,
        )
        if n > 1:
            milling_forces = milling_process.total_milling_force[0:2, np.newaxis]
        else:
            milling_forces = np.zeros((2,1))

        # Integrate oscillator only when spring-damper dynamics are enabled.
        if ENABLE_SPRING_DAMPER_DYNAMICS:
            state = RungeKutta4(oscillator_dynamics, t, state, dt, p_osc, u, milling_forces)
        else:
            state[:, 0] = 0.0

        # logs
        t_hist[n] = t
        x_hist[n,:] = state[0:2,0]
        u_hist[n,:] = u[:,0]
        fmill_hist[n,:] = milling_forces[:,0]
        fanalyt_hist[n,:] = F_analytical
        fzero_hist[n,:] = F_zero
        tool_center_nominal_hist_mm[n, :] = tool_center_nominal_mm
        tool_center_actual_hist_mm[n, :] = tool_center_mm

        t += dt
        
        if n % live_plot_stride == 0:
            progress = 100.0 * n / max(1, N - 1)
            print(f"Sim state: {progress:5.1f}% t={t:.3f} s, x={state[0:2,0]}, milling_force={milling_forces[:,0]}")
            if ENABLE_LIVE_PLOTTING and n > 1:
                fig_live, _ = milling_process.show_plot(window_name="Milling process (live)", clear=True)
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()
                plt.pause(0.001)
                
    # -------------------------
    # Plots
    # -------------------------
    step = 20
    td = t_hist[::step]
    xd = x_hist[::step]
    ud = u_hist[::step]
    fd = fmill_hist[::step]
    fad = fanalyt_hist[::step]
    f0d = fzero_hist[::step]

    # Block-average simulated forces over one spindle revolution.
    omega_abs = abs(params_milling.process.omega())
    rev_period = (2.0 * m.pi) / max(omega_abs, 1e-12)
    samples_per_rev = max(1, int(round(rev_period / dt)))
    fmill_revavg = np.full_like(fmill_hist, np.nan)
    for i0 in range(0, N, samples_per_rev):
        i1 = min(N, i0 + samples_per_rev)
        fmean = np.mean(fmill_hist[i0:i1, :], axis=0)
        fmill_revavg[i0:i1, :] = fmean
    frad = fmill_revavg[::step]

    if SAVE_EXPERIMENT_DATA:
        save_path = save_experiment_data(
            output_dir=OUTPUT_DIR,
            params_milling=params_milling,
            dt=dt,
            total_time_s=T,
            spring_damper_enabled=ENABLE_SPRING_DAMPER_DYNAMICS,
            t_hist=t_hist,
            x_hist=x_hist,
            u_hist=u_hist,
            fmill_hist=fmill_hist,
            fanalyt_hist=fanalyt_hist,
            fzero_hist=fzero_hist,
            tool_center_nominal_hist_mm=tool_center_nominal_hist_mm,
            tool_center_actual_hist_mm=tool_center_actual_hist_mm,
        )
        print(f"Saved experiment trajectories to: {save_path}")
        print(f"File Name: {save_path.name}")


    # Position response
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(td, xd[:,0], label="x")
    ax[0].grid(True); ax[0].legend(); ax[0].set_ylabel("x [m]")

    ax[1].plot(td, xd[:,1], label="y")
    ax[1].grid(True); ax[1].legend(); ax[1].set_ylabel("y [m]"); ax[1].set_xlabel("t [s]")
    fig.suptitle("Open-loop TCP response (single oscillator)")

    # Control force
    plt.figure()
    plt.plot(td, ud[:,0], label="u_x")
    plt.plot(td, ud[:,1], label="u_y")
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("Force command [N]")
    plt.title("Direct force input u")
    plt.legend()

    # Milling disturbance
    plt.figure()
    plt.plot(td, fd[:,0], label="F_mill_x (simulation)")
    plt.plot(td, fd[:,1], label="F_mill_y (simulation)")
    plt.plot(td, frad[:,0], "-.", linewidth=2.0, label="F_mill_x (sim avg/rev)")
    plt.plot(td, frad[:,1], "-.", linewidth=2.0, label="F_mill_y (sim avg/rev)")
    plt.plot(td, fad[:,0], "--", label="F_x (analytical, inst.)")
    plt.plot(td, fad[:,1], "--", label="F_y (analytical, inst.)")
    plt.plot(td, f0d[:,0], ":", label="F_x (analytical, zero-order)")
    plt.plot(td, f0d[:,1], ":", label="F_y (analytical, zero-order)")
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("Milling force [N]")
    plt.title("Milling force: simulation vs analytical downmilling")
    plt.legend()
    
    

    plt.ioff()
    plt.show()
