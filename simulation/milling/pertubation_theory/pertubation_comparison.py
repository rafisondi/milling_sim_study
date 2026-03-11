import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import math as m

# Milling simulation for process forces
from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *
from utils.save_utils import *
from utils.dynamics_utils import * 


# -------------------------
# Directional Cutting Matrix
# -------------------------

def angle_in_sector(phi, phi_st, phi_ex):
    phi = phi % (2.0*np.pi)
    phi_st = phi_st % (2.0*np.pi)
    phi_ex = phi_ex % (2.0*np.pi)

    if phi_st <= phi_ex:
        return phi_st <= phi <= phi_ex
    else:
        return (phi >= phi_st) or (phi <= phi_ex)


def directional_cutting_matrix(
        orientation,
        phi_st,
        phi_ex,
        z_teeth,
        milling_process,
        axial_depth_mm,
        tooth_period_s,
        feed_dir,
):
    """
    B(t) matrix such that

        F_pert = B(t) @ u_dot
    """

    Kt = float(milling_process.cutting_force_coefficient_Ktc)
    Kr = float(milling_process.cutting_force_coefficient_Krc)

    b = float(axial_depth_mm)
    N = int(z_teeth)

    B = np.zeros((2,2))
    feed_dir = np.asarray(feed_dir, dtype=float).reshape(2)
    feed_norm = np.linalg.norm(feed_dir)
    if feed_norm <= 0.0:
        raise ValueError("feed_dir must be non-zero")
    feed_dir = feed_dir / feed_norm

    for j in range(N):

        phi_j = (orientation + 2*np.pi*j/N) % (2*np.pi)

        if not angle_in_sector(phi_j, phi_st, phi_ex):
            continue

        dphi = np.array([
            -Kt*np.cos(phi_j) - Kr*np.sin(phi_j),
             Kt*np.sin(phi_j) - Kr*np.cos(phi_j)
        ]).reshape(2,1)

        # Dynamic chip thickness is the feed-direction velocity projected
        # over one tooth period, scaled by the usual sin(phi) term.
        pphi = (np.sin(phi_j) * feed_dir).reshape(1,2)

        B += b * tooth_period_s * (dphi @ pphi)

    return B


def perturbation_force_estimate(F_zero, xdot_state_m_s, B):

    xdot_state_mm_s = 1e3 * np.asarray(xdot_state_m_s).reshape(2,1)

    F_pert = B @ xdot_state_mm_s
    F_est = np.asarray(F_zero).reshape(2,1) + F_pert

    return F_est


# -------------------------
# Global config 
# -------------------------

SIMULATION_TIME_S = 3.5
PRINT_EVERY_N = 2000
SAVE_EXPERIMENT_DATA = True
ENABLE_LIVE_PLOTTING = False 
ENABLE_SPRING_DAMPER_DYNAMICS = True
ENABLE_FINAL_FORCE_PLOT = True

RADIAL_ENGAGEMENT_MM = 8.0
AXIAL_CUTTING_DEPTH_MM = 1.0

FEED_DIR = np.array([0.0, 1.0], dtype=float) 
DT =  1e-4
Samples_per_Period = 360 
TOOL_DIAMETER_MM = 20.0

FEED_SPEED = 20.0
Z_TEETH = 4
SPINDLE_SPIN = -1


# ----- derived parameters -----

Rotation_Period = DT * Samples_per_Period
frequency = 1 / Rotation_Period
rpm = 60 * frequency
TOOL_RADIUS_MM = TOOL_DIAMETER_MM / 2.0
RPM_TARGET = rpm
rev_per_s = rpm / 60.0
omega = SPINDLE_SPIN * 2.0 * np.pi * rev_per_s
fz_mm = FEED_SPEED / (rev_per_s * Z_TEETH)

tooth_period_s = 1.0 / (rev_per_s * Z_TEETH)


# --- Workpiece parameters --- 

OUTPUT_DIR =   "data/experiments"
PARAMS_DIR =   "data/experiments/settings"
LINEARIZED_MODEL_NPZ =  "data/linearized_operational_space_xyz.npz"


WORKPIECE_P1_MM = np.array([0.0, 0.0])
WORKPIECE_P2_MM = np.array([50.0, 0.0])
WORKPIECE_P3_MM = np.array([50.0, 50.0])
WORKPIECE_P4_MM = np.array([0.0, 50.0])


# -------------------------
# Build Macro dynamics
# -------------------------

if ENABLE_SPRING_DAMPER_DYNAMICS:

    M2, C2, K2, extras = load_macro_from_npz(LINEARIZED_MODEL_NPZ, axes=(0, 1))

    M_micro = np.eye(2) * 80.0
    Mtot = M2 + M_micro

    p_osc = MacroOscillatorParams(M=Mtot, C=C2, K=K2)


# ----- experiment dictionary -----

params = {
    "dt": DT,
    "samples_per_period": Samples_per_Period,
    "period": Rotation_Period,
    "frequency_hz": frequency,
    "rpm": rpm,
    "rev_per_s": rev_per_s,
    "omega_rad_s": omega,
    "spindle_spin": SPINDLE_SPIN,
    "z_teeth": Z_TEETH,
    "tool_diameter_mm": TOOL_DIAMETER_MM,
    "tool_radius_mm": TOOL_RADIUS_MM,
    "axial_cutting_depth_mm": AXIAL_CUTTING_DEPTH_MM,
    "feed_speed_mm_s": FEED_SPEED,
    "feed_per_tooth_mm": fz_mm,
    "dynamics_enabled" : ENABLE_SPRING_DAMPER_DYNAMICS,
    "save_experiment_data": SAVE_EXPERIMENT_DATA,
    "enable_final_force_plot": ENABLE_FINAL_FORCE_PLOT,
}


@dataclass
class WorkpieceGeometry:

    p1: np.ndarray = field(default_factory=lambda: np.array([0,0]))
    p2: np.ndarray = field(default_factory=lambda: np.array([0,0]))
    p3: np.ndarray = field(default_factory=lambda: np.array([0,0]))
    p4: np.ndarray = field(default_factory=lambda: np.array([0,0]))

    @property
    def xy(self):
        return np.array([self.p1,self.p2,self.p3,self.p4]).T


def tool_center_offset_for_radial_engagement(radius_mm, radial_eng_mm, feed_dir):

    n_surface = np.array([-feed_dir[1], feed_dir[0]])
    offset_from_surface = radius_mm - radial_eng_mm

    offset = (n_surface * offset_from_surface - feed_dir* radius_mm)

    return np.array([offset[0],offset[1]])


# -------------------------
# Analytical milling forces
# -------------------------

def analytical_downmilling_force(
        tool_center_mm,
        orientation,
        workpiece_xy,
        tool_radius_mm,
        tool_diameter_mm,
        axial_depth_mm,
        z_teeth,
        feed_per_tooth_mm,
        milling_process,
    ):

    phi_st, phi_ex = compute_entry_exit_angles_downmilling(
        r_xy_current=np.asarray(tool_center_mm).reshape(2),
        xy_workpiece=workpiece_xy,
        radius_tool=float(tool_radius_mm),
        diameter_end_mill=float(tool_diameter_mm),
    )

    if (phi_ex - phi_st) <= 0.0:
        return np.zeros(2), np.zeros(2), (phi_st,phi_ex)

    Kt = milling_process.cutting_force_coefficient_Ktc
    Kr = milling_process.cutting_force_coefficient_Krc

    fz = float(feed_per_tooth_mm)
    b = float(axial_depth_mm)
    N = int(z_teeth)

    F_inst = np.zeros(2)

    for j in range(N):

        phi_j = (orientation + 2*np.pi*j/N) % (2*np.pi)

        F_inst += analytical_force(
            phi=phi_j,
            phi_st=phi_st,
            phi_ex=phi_ex,
            b=b,
            fz=fz,
            Kt=Kt,
            Kr=Kr,
        )

    F_zero = zero_order_force_analytical(
        c=fz,
        a=b,
        N=N,
        phi_st=phi_st,
        phi_ex=phi_ex,
        Ktc=Kt,
        Krc=Kr,
    )

    return F_inst, F_zero, (phi_st,phi_ex)


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    workpiece = WorkpieceGeometry(
        WORKPIECE_P1_MM,
        WORKPIECE_P2_MM,
        WORKPIECE_P3_MM,
        WORKPIECE_P4_MM
    )

    milling_process = milling_workpiece(workpiece.xy)

    milling_process.diameter_end_mill = params["tool_diameter_mm"]
    milling_process.number_of_teeth = params["z_teeth"]
    milling_process.radius_tool = params["tool_radius_mm"]
    milling_process.slice_height = params["axial_cutting_depth_mm"]


    FEED_DIR = np.array([1.0,0.0])

    start_corner = WORKPIECE_P4_MM
    travel_distance = float(WORKPIECE_P3_MM[0] - WORKPIECE_P4_MM[0])

    overtravel_dist = params["tool_radius_mm"] + 0.1

    start_center = start_corner + tool_center_offset_for_radial_engagement(
        params["tool_radius_mm"],
        RADIAL_ENGAGEMENT_MM,
        FEED_DIR
    )

    end_center = start_center + (travel_distance + overtravel_dist)*FEED_DIR

    tool_center0_mm = start_center


    u = np.zeros((2,1))

    dt = DT
    T = SIMULATION_TIME_S
    N = int(T/dt)

    state = np.zeros((4,1))
    t = 0.0


    t_hist = np.zeros(N)
    x_hist = np.zeros((N,2))
    u_hist = np.zeros((N,2))
    fmill_hist = np.zeros((N,2))
    fanalyt_hist = np.zeros((N,2))
    fzero_hist = np.zeros((N,2))
    fest_hist = np.zeros((N,2))
    tool_center_nominal_hist_mm = np.zeros((N, 2))
    tool_center_actual_hist_mm = np.zeros((N, 2))

    phi_tool_hist = np.zeros(N)

    live_plot_stride = max(1, PRINT_EVERY_N)
    for n in range(N):

        spindle_angle = t * params["omega_rad_s"]

        tool_center_mm = (
            tool_center0_mm
            + FEED_DIR * params["feed_speed_mm_s"] * t
            + state[0:2,0]*1e3
        )
        tool_center_nominal_mm = tool_center0_mm + FEED_DIR * params["feed_speed_mm_s"] * t

        milling_process.erase_step(
            tool_center_mm,
            -spindle_angle,
            params["spindle_spin"],
            t
        )


        F_analytical, F_zero, (phi_st,phi_ex) = analytical_downmilling_force(
            tool_center_mm=tool_center_mm,
            orientation=spindle_angle + np.pi/2,
            workpiece_xy=workpiece.xy,
            tool_radius_mm=params["tool_radius_mm"],
            tool_diameter_mm=params["tool_diameter_mm"],
            axial_depth_mm=params["axial_cutting_depth_mm"],
            z_teeth=params["z_teeth"],
            feed_per_tooth_mm=params["feed_per_tooth_mm"],
            milling_process=milling_process,
        )


        B = directional_cutting_matrix(
            orientation=spindle_angle + np.pi/2,
            phi_st=phi_st,
            phi_ex=phi_ex,
            z_teeth=params["z_teeth"],
            milling_process=milling_process,
            axial_depth_mm=params["axial_cutting_depth_mm"],
            tooth_period_s=tooth_period_s,
            feed_dir=FEED_DIR,
        )


        F_est = perturbation_force_estimate(
            F_zero,
            state[2:4,0],
            B
        )


        if n > 1:
            milling_forces = milling_process.total_milling_force[0:2,np.newaxis]
        else:
            milling_forces = np.zeros((2,1))


        if ENABLE_SPRING_DAMPER_DYNAMICS:
            state = RungeKutta4(
                dynamics,
                t,
                state,
                dt,
                p_osc,
                u,
                milling_forces
            )


        if n % live_plot_stride == 0:
            progress = 100.0 * n / max(1, N - 1)
            print(f"Sim state: {progress:5.1f}% t={t:.3f} s, x={state[0:2,0]}, milling_force={milling_forces[:,0]}")
            if ENABLE_LIVE_PLOTTING and n > 1:
                fig_live, _ = milling_process.show_plot(window_name="Milling process (live)", clear=True)
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()
                plt.pause(0.001)
        
        t_hist[n] = t
        x_hist[n,:] = state[0:2,0]
        u_hist[n,:] = u[:,0]

        fmill_hist[n,:] = milling_forces[:,0]
        fanalyt_hist[n,:] = F_analytical
        fzero_hist[n,:] = F_zero
        fest_hist[n,:] = F_est[:,0]
        tool_center_nominal_hist_mm[n, :] = tool_center_nominal_mm
        tool_center_actual_hist_mm[n, :] = tool_center_mm

        phi_tool_hist[n] = spindle_angle

        t += dt

    if SAVE_EXPERIMENT_DATA:
        settings_hash, params_dir = save_params(params=params, base_dir=PARAMS_DIR)

        exp_dir, csv_path, run_id = save_experiment_csv(
            params=params,
            settings_hash=settings_hash,
            output_dir=OUTPUT_DIR,
            t_hist=t_hist,
            x_hist=x_hist,
            u_hist=u_hist,
            fmill_hist=fmill_hist,
            fanalyt_hist=fanalyt_hist,
            fzero_hist=fzero_hist,
            tool_center_nominal_hist_mm=tool_center_nominal_hist_mm,
            tool_center_actual_hist_mm=tool_center_actual_hist_mm,
            tool_orientation_hist=phi_tool_hist
        )

        write_to_experimental_log(
            params=params,
            settings_hash=settings_hash,
            run_id=run_id,
            exp_dir=exp_dir,
            csv_path=csv_path,
            params_dir=params_dir,
            log_path="data/experimental_log.txt",
        )

        print("Saved CSV to:", csv_path)

    if ENABLE_FINAL_FORCE_PLOT:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        labels = ("Fx", "Fy")

        for i, ax in enumerate(axs):
            ax.plot(t_hist, fmill_hist[:, i], label=f"{labels[i]} milling")
            ax.plot(t_hist, fest_hist[:, i], "--", label=f"{labels[i]} perturbation estimate")
            ax.set_ylabel(f"{labels[i]} [N]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axs[-1].set_xlabel("time [s]")
        fig.suptitle("Milling force vs perturbation force estimate")
        fig.tight_layout()
        plt.show()
