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
# Global config 
# -------------------------
SIMULATION_TIME_S = 10
PRINT_EVERY_N = 2000
SAVE_EXPERIMENT_DATA = True
ENABLE_LIVE_PLOTTING = False 
ENABLE_SPRING_DAMPER_DYNAMICS = True

# Motion / engagement config 
RADIAL_ENGAGEMENT_MM = 8.0
AXIAL_CUTTING_DEPTH_MM = 1.0


FEED_DIR = np.array([0.0, 1.0], dtype=float) 
DT =  1e-4
Samples_per_Period = 360
TOOL_DIAMETER_MM = 20.0

FEED_SPEED = 20.0     # [mm/s]
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


# --- Workpiece parameters --- 
OUTPUT_DIR =   "data/experiments"
PARAMS_DIR =   "data/experiments/settings"
LINEARIZED_MODEL_NPZ =  "data/linearized_operational_space_xyz.npz"

# ---- Workpiece geometry [mm]
WORKPIECE_P1_MM = np.array([0.0, 0.0])
WORKPIECE_P2_MM = np.array([100.0, 0.0])
WORKPIECE_P3_MM = np.array([100.0, 100.0])
WORKPIECE_P4_MM = np.array([0.0, 100.0])

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
    # simulation
    "dt": DT,
    "samples_per_period": Samples_per_Period,
    "period": Rotation_Period,
    "frequency_hz": frequency,

    # spindle
    "rpm": rpm,
    "rev_per_s": rev_per_s,
    "omega_rad_s": omega,
    "spindle_spin": SPINDLE_SPIN,
    "z_teeth": Z_TEETH,

    # tool
    "tool_diameter_mm": TOOL_DIAMETER_MM,
    "tool_radius_mm": TOOL_RADIUS_MM,
    "axial_cutting_depth_mm": AXIAL_CUTTING_DEPTH_MM,
    "radial_engagement_mm": RADIAL_ENGAGEMENT_MM,

    # process
    "feed_speed_mm_s": FEED_SPEED,
    "feed_per_tooth_mm": fz_mm,
    
    # settings
    "dynamics_enabled" : ENABLE_SPRING_DAMPER_DYNAMICS,
}



@dataclass
class WorkpieceGeometry:
    p1: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    p2: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    p3: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    p4: np.ndarray = field(default_factory=lambda: np.array([0, 0]))

    @property
    def xy(self):
        return np.array([self.p1, self.p2, self.p3, self.p4]).T


    
def tool_center_offset_for_radial_engagement(radius_mm, radial_eng_mm, feed_dir):
    """Helper function which allows easy starting position for tool tip placement"""
    n_surface = np.array([-feed_dir[1], feed_dir[0]]) 
    offset_from_surface = radius_mm - radial_eng_mm
    
    offset = (n_surface * offset_from_surface - feed_dir* radius_mm)
    offset_x = offset[0]
    offset_y = offset[1]
    
    return np.array([offset_x, offset_y])

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
        r_xy_current=np.asarray(tool_center_mm, dtype=float).reshape(2),
        xy_workpiece=workpiece_xy,
        radius_tool=float(tool_radius_mm),
        diameter_end_mill=float(tool_diameter_mm),
    )

    if (phi_ex - phi_st) <= 0.0:
        return np.zeros(2), np.zeros(2), (phi_st, phi_ex)

    # cutting coefficients from your milling_process object
    Kt = milling_process.cutting_force_coefficient_Ktc
    Kr = milling_process.cutting_force_coefficient_Krc

    fz = float(feed_per_tooth_mm)
    b = float(axial_depth_mm)
    N = int(z_teeth)

    F_inst = np.zeros(2, dtype=float)
    for j in range(N):
        phi_j = (orientation + 2.0 * m.pi * j / N) % (2.0 * m.pi)
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

    return F_inst, F_zero, (phi_st, phi_ex)

if __name__ == "__main__":
    wp_p1 = WORKPIECE_P1_MM
    wp_p2 = WORKPIECE_P2_MM
    wp_p3 = WORKPIECE_P3_MM
    wp_p4 = WORKPIECE_P4_MM
    
    workpiece = WorkpieceGeometry(wp_p1, wp_p2, wp_p3, wp_p4)
    milling_process = milling_workpiece(workpiece.xy)
    milling_process.diameter_end_mill   = params["tool_diameter_mm"]
    milling_process.number_of_teeth     = params["z_teeth"]
    milling_process.radius_tool         = params["tool_radius_mm"]
    milling_process.slice_height        = params["axial_cutting_depth_mm"]
    
    # # ---- Trajectory configuration ----
    # FEED_DIR = np.array([0.0, 1.0])
    # start_corner = wp_p1
    # travel_distance = float(wp_p4[1] - wp_p1[1])
    # overtravel_dist = params["tool_radius_mm"] + 0.1
    # start_center = start_corner  + tool_center_offset_for_radial_engagement(params["tool_radius_mm"], RADIAL_ENGAGEMENT_MM, FEED_DIR)
    # end_center   = start_center + (travel_distance + overtravel_dist) * FEED_DIR    
    # tool_center0_mm = start_center
    # # -----------------------------------
    
    # ---- Trajectory configuration ----
    FEED_DIR = np.array([1.0, 0.0])
    start_corner = wp_p4
    travel_distance = float(wp_p3[0] - wp_p4[0])
    overtravel_dist = params["tool_radius_mm"] + 0.1
    start_center = start_corner  + tool_center_offset_for_radial_engagement(params["tool_radius_mm"], RADIAL_ENGAGEMENT_MM, FEED_DIR)
    end_center   = start_center + (travel_distance + overtravel_dist) * FEED_DIR    
    tool_center0_mm = start_center
    # -----------------------------------
    
    # Open-loop configuration:
    u = np.zeros((2,1))
    dt = DT       
    T  = SIMULATION_TIME_S
    N  = int(T / dt)

    state = np.zeros((4,1))  # [x; xdot] in m
    t = 0.0

    t_hist = np.zeros(N)
    x_hist = np.zeros((N,2))
    u_hist = np.zeros((N,2))
    phi_tool_hist = np.zeros(N)
    fmill_hist = np.zeros((N,2))
    fanalyt_hist = np.zeros((N,2))
    fzero_hist = np.zeros((N,2))
    tool_center_nominal_hist_mm = np.zeros((N, 2))
    tool_center_actual_hist_mm = np.zeros((N, 2))
    milling_forces = np.zeros((2,1))
    
    feed_speed_mm_s = params["feed_speed_mm_s"]

    workpiece_width = WORKPIECE_P2_MM[0] - WORKPIECE_P1_MM[0]
    workpiece_height = WORKPIECE_P4_MM[1] - WORKPIECE_P1_MM[1]

    tool_start = tool_center0_mm
    tool_entry = start_center
    tool_exit = end_center

    approach_distance_mm = np.linalg.norm(tool_entry - tool_start)
    cut_distance_mm = np.linalg.norm(tool_exit - tool_entry)

    t_entry_s = approach_distance_mm / max(feed_speed_mm_s, 1e-12)
    t_cut_s = cut_distance_mm / max(feed_speed_mm_s, 1e-12)
    t_total_s = t_entry_s + t_cut_s


    print("\n=== Simulation ===")
    print("Feed speed (mm/s):", params["feed_speed_mm_s"])
    print(f"Simulation time (s): {T:.3f}")
    print("Spring-damper dynamics enabled:", ENABLE_SPRING_DAMPER_DYNAMICS)
    print("Live plotting enabled:", ENABLE_LIVE_PLOTTING)

    print("\n=== Milling Setup Summary ===")

    print(
        f"Workpiece (mm): width={workpiece_width:.2f}, "
        f"height={workpiece_height:.2f}"
    )

    print(
        f"Tool: D={params['tool_diameter_mm']:.2f} mm, "
        f"R={params['tool_radius_mm']:.2f} mm, "
        f"radial_eng={RADIAL_ENGAGEMENT_MM:.2f} mm, "
        f"axial_depth={params['axial_cutting_depth_mm']:.2f} mm, "
        f"teeth={params['z_teeth']}"
    )

    print(
        f"Process: rpm={params['rpm']:.1f}, "
        f"feed_per_tooth={params['feed_per_tooth_mm']:.4f} mm/tooth, "
        f"feed_speed={feed_speed_mm_s:.3f} mm/s"
    )

    print(
        f"Path: overtravel={params['tool_radius_mm'] + 0.1:.2f} mm, "
        f"feed_dir={FEED_DIR}"
    )

    print(
        f"Tool center positions (mm): "
        f"start=({tool_start[0]:.3f}, {tool_start[1]:.3f}), "
        f"entry=({tool_entry[0]:.3f}, {tool_entry[1]:.3f}), "
        f"exit=({tool_exit[0]:.3f}, {tool_exit[1]:.3f})"
    )

    print(
        f"Timing from feed: "
        f"entry={t_entry_s:.3f} s, "
        f"cut+exit={t_cut_s:.3f} s, "
        f"total={t_total_s:.3f} s"
    )
    
    if ENABLE_LIVE_PLOTTING:
        plt.ion()
    live_plot_stride = max(1, PRINT_EVERY_N)
    for n in range(N):

        spindle_angle = t * params["omega_rad_s"]
        tool_center_mm = (
            tool_center0_mm
            + FEED_DIR * params["feed_speed_mm_s"] * t
            + state[0:2, 0] * 1e3
        )
        tool_center_nominal_mm = tool_center0_mm + FEED_DIR * params["feed_speed_mm_s"]  * t
        milling_process.erase_step(tool_center_mm, -spindle_angle, params["spindle_spin"], t)
        
        F_analytical, F_zero, _ = analytical_downmilling_force(
            tool_center_mm=tool_center_mm,
            orientation=spindle_angle  + np.pi / 2,
            workpiece_xy=workpiece.xy,
            tool_radius_mm=params["tool_radius_mm"],
            tool_diameter_mm=params["tool_diameter_mm"],
            axial_depth_mm=params["axial_cutting_depth_mm"],
            z_teeth=params["z_teeth"],
            feed_per_tooth_mm=params["feed_per_tooth_mm"],
            milling_process=milling_process,
        )
        
        if n > 1:
            milling_forces = milling_process.total_milling_force[0:2, np.newaxis]
        else:
            milling_forces = np.zeros((2,1))
            
        # Integrate oscillator only when spring-damper dynamics are enabled.
        if ENABLE_SPRING_DAMPER_DYNAMICS:
            state = RungeKutta4(dynamics, t, state, dt, p_osc, u, milling_forces)
        else:
            state[:, 0] = 0.0

        t_hist[n] = t
        x_hist[n,:] = state[0:2,0]
        u_hist[n,:] = u[:,0]
        fmill_hist[n,:] = milling_forces[:,0]
        fanalyt_hist[n,:] = F_analytical
        fzero_hist[n,:] = F_zero
        tool_center_nominal_hist_mm[n, :] = tool_center_nominal_mm
        tool_center_actual_hist_mm[n, :] = tool_center_mm
        phi_tool_hist[n] = spindle_angle
        t += dt
        
        if n % live_plot_stride == 0:
            progress = 100.0 * n / max(1, N - 1)
            print(f"Sim state: {progress:5.1f}% t={t:.3f} s, x={state[0:2,0]}, milling_force={milling_forces[:,0]}")
            if ENABLE_LIVE_PLOTTING and n > 1:
                fig_live, _ = milling_process.show_plot(window_name="Milling process (live)", clear=True)
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()
                plt.pause(0.001)
                
    
    
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
        
        