import numpy as np
import pandas as pd
from pathlib import Path

from eraser_of_matter import milling_workpiece
from milling_data import generate_chafli_trajectory, load_workpiece_vertices_from_pickle
from utils.dynamics_utils import load_macro_from_npz, MacroOscillatorParams, dynamics, RungeKutta4
from utils.save_utils import save_experiment_csv, save_params, write_to_experimental_log

# -------------------------
# Global config
# -------------------------
DT = 1e-4
SAMPLES_PER_PERIOD = 180 #  360 
PRINT_EVERY_N = 2000
SAVE_EXPERIMENT_DATA = True

# Chafli geometry/path assets
MILLING_PATH_CSV = "data/paths/Workpiece_long_with_start_milling_path.csv"
WORKPIECE_PICKLE = "data/paths/Workpiece_long_with_start.pickle"

# Process settings
AXIAL_CUTTING_DEPTH_MM = 1.0
FEED_SPEED_MM_S = 40.0
MIN_FEED_SPEED_MM_S = 1.0
SPINDLE_SPIN = -1
Z_TEETH = 8

PATH_OFFSET_MM = None
PATH_OFFSET_SIDE = "left"

USE_PRECOMPUTED_FEED_PROFILE = True
FEED_PROFILE_CSV = "optimized_feed/optimization_offline_cc/Optimal_feed_curve.csv"
TRAJECTORY_FINAL_S_MM = 200
USE_ROBOT_DYNAMICS = True
LINEARIZED_MODEL_NPZ = "linearized_operational_space_xyz.npz"
MICRO_MASS_DIAG_KG = 80.0

# Save locations
OUTPUT_DIR = "data/experiments"
PARAMS_DIR = "data/experiments/settings"


# ----- derived parameters -----
rotation_period_s = DT * SAMPLES_PER_PERIOD
frequency_hz = 1.0 / max(rotation_period_s, 1e-12)
rpm = 60.0 * frequency_hz
rev_per_s = rpm / 60.0
omega = SPINDLE_SPIN * 2.0 * np.pi * rev_per_s
feed_per_tooth_mm = FEED_SPEED_MM_S / max(rev_per_s * Z_TEETH, 1e-12)


def load_precomputed_feed_curve(csv_path: str | Path) -> np.ndarray:
    feed_curve_df = pd.read_csv(csv_path)
    required_cols = {"x_support", "feed_curve"}
    missing_cols = required_cols.difference(feed_curve_df.columns)

    feed_curve = feed_curve_df[["x_support", "feed_curve"]].to_numpy(dtype=float)
    order = np.argsort(feed_curve[:, 0], kind="stable")
    feed_curve = feed_curve[order]

    unique_support_mask = np.ones(len(feed_curve), dtype=bool)
    unique_support_mask[1:] = np.diff(feed_curve[:, 0]) > 0.0
    feed_curve = feed_curve[unique_support_mask]

    return feed_curve


def resolve_final_path_distance_mm(
    feed_curve: np.ndarray | None,
    requested_final_s_mm: float | None,
) -> float | None:
    if requested_final_s_mm is not None:
        return float(requested_final_s_mm)
    if feed_curve is not None:
        return float(feed_curve[-1, 0])
    return None


# ----- experiment dictionary -----
params = {
    # simulation
    "dt": DT,
    "samples_per_period": SAMPLES_PER_PERIOD,
    "period": rotation_period_s,
    "frequency_hz": frequency_hz,

    # spindle / process
    "rpm": rpm,
    "rev_per_s": rev_per_s,
    "omega_rad_s": omega,
    "spindle_spin": SPINDLE_SPIN,
    "z_teeth": Z_TEETH,
    "feed_per_tooth_mm": feed_per_tooth_mm,

    # trajectory feed
    "feed_speed_mm_s": FEED_SPEED_MM_S,
    "feed_speed_path_mm_s": FEED_SPEED_MM_S,
    "min_feed_speed_mm_s": MIN_FEED_SPEED_MM_S,
    "use_precomputed_feed_profile": USE_PRECOMPUTED_FEED_PROFILE,
    "feed_profile_csv": FEED_PROFILE_CSV if USE_PRECOMPUTED_FEED_PROFILE else None,
    "trajectory_final_s_mm": TRAJECTORY_FINAL_S_MM,

    # tool / cut
    "axial_cutting_depth_mm": AXIAL_CUTTING_DEPTH_MM,

    # geometry/path
    "milling_path_csv": MILLING_PATH_CSV,
    "workpiece_pickle": WORKPIECE_PICKLE,
    "path_offset_mm": PATH_OFFSET_MM,
    "path_offset_side": PATH_OFFSET_SIDE,

    # compliant-base model
    "use_robot_dynamics": USE_ROBOT_DYNAMICS,
    "linearized_model_npz": LINEARIZED_MODEL_NPZ if USE_ROBOT_DYNAMICS else None,
    "micro_mass_diag_kg": MICRO_MASS_DIAG_KG if USE_ROBOT_DYNAMICS else None,
}
if __name__ == "__main__":
    precomputed_feed_curve = None
    p_osc = None
    Mtot = None
    C2 = None
    K2 = None
    if USE_PRECOMPUTED_FEED_PROFILE:
        precomputed_feed_curve = load_precomputed_feed_curve(FEED_PROFILE_CSV)
    final_s_mm = resolve_final_path_distance_mm(precomputed_feed_curve, TRAJECTORY_FINAL_S_MM)
    params["trajectory_final_s_mm"] = final_s_mm
    if USE_ROBOT_DYNAMICS:
        M2, C2, K2, _ = load_macro_from_npz(LINEARIZED_MODEL_NPZ, axes=(0, 1))
        M_micro = np.eye(2) * MICRO_MASS_DIAG_KG
        Mtot = M2 + M_micro
        p_osc = MacroOscillatorParams(M=Mtot, C=C2, K=K2)

    workpiece_vertices = load_workpiece_vertices_from_pickle(WORKPIECE_PICKLE)
    trajectory_local_mm, pos_on_path_mm = generate_chafli_trajectory(
        MILLING_PATH_CSV,
        axial_cutting_depth_mm=AXIAL_CUTTING_DEPTH_MM,
        feed_speed_mm_s=FEED_SPEED_MM_S,
        dt=DT,
        feed_curve=precomputed_feed_curve,
        min_feed_speed_mm_s=MIN_FEED_SPEED_MM_S,
        final_pos_on_path_mm=final_s_mm,
        path_offset_mm=PATH_OFFSET_MM,
        path_offset_side=PATH_OFFSET_SIDE,
    )

    milling_process = milling_workpiece(workpiece_vertices, axial_cutting_depth=AXIAL_CUTTING_DEPTH_MM)
    milling_process.number_of_teeth = params["z_teeth"]
    milling_process.slice_height = params["axial_cutting_depth_mm"]

    N = len(trajectory_local_mm)
    t_hist = np.arange(N) * DT
    planned_feed_hist_mm_s = np.diff(pos_on_path_mm, prepend=pos_on_path_mm[0]) / DT
    if N > 1:
        planned_feed_hist_mm_s[0] = planned_feed_hist_mm_s[1]

    state = np.zeros((4, 1))
    u = np.zeros((2, 1))
    x_hist = np.zeros((N, 2))
    u_hist = np.zeros((N, 2))
    fmill_hist = np.zeros((N, 2))
    tool_center_nominal_hist_mm = trajectory_local_mm.copy()
    tool_center_actual_hist_mm = trajectory_local_mm.copy()
    phi_tool_hist = np.zeros(N)

    print("\n=== Trajectory Milling Simulation ===")
    print(f"Samples: {N}")
    print(f"dt: {DT:.1e} s")
    print(f"Samples per period: {params['samples_per_period']}")
    print(f"RPM: {params['rpm']:.1f}")
    print(f"Path feed speed: {params['feed_speed_mm_s']:.3f} mm/s")
    if final_s_mm is not None:
        print(f"Trajectory stop distance s_f: {final_s_mm:.3f} mm")
        print(f"Reached final logged s(t): {pos_on_path_mm[-1]:.3f} mm")
    if USE_PRECOMPUTED_FEED_PROFILE:
        print(f"Precomputed feed profile: {FEED_PROFILE_CSV}")
        print(
            f"Planned feed range: {planned_feed_hist_mm_s.min():.3f} .. "
            f"{planned_feed_hist_mm_s.max():.3f} mm/s"
        )
    if USE_ROBOT_DYNAMICS:
        print(f"Linearized model: {params['linearized_model_npz']}")
        print(f"M diag (kg): {np.diag(Mtot)}")
        print(f"C diag (N s/m): {np.diag(C2)}")
        print(f"K diag (N/m): {np.diag(K2)}")

    for i in range(N):
        t = t_hist[i]
        spindle_angle = t * params["omega_rad_s"]
        tool_center_nominal_mm = trajectory_local_mm[i]
        if USE_ROBOT_DYNAMICS:
            tool_center_mm = tool_center_nominal_mm + state[0:2, 0] * 1e3
        else:
            tool_center_mm = tool_center_nominal_mm

        milling_process.erase_step(
            tool_center_mm,
            spindle_angle,
            direction=params["spindle_spin"],
            t=t,
        )

        if i > 1:
            milling_forces = np.asarray(milling_process.total_milling_force[0:2], dtype=float).reshape(2, 1)
        else:
            milling_forces = np.zeros((2, 1))

        if USE_ROBOT_DYNAMICS:
            state = RungeKutta4(dynamics, t, state, DT, p_osc, u, milling_forces)

        fmill_hist[i, :] = milling_forces[:, 0]
        x_hist[i, :] = state[0:2, 0]
        u_hist[i, :] = u[:, 0]
        tool_center_actual_hist_mm[i, :] = tool_center_mm
        phi_tool_hist[i] = spindle_angle

        if i % PRINT_EVERY_N == 0:
            progress = 100.0 * i / max(1, N - 1)
            if USE_ROBOT_DYNAMICS:
                print(
                    f"Sim state: {progress:5.1f}% t={t:.3f} s, "
                    f"tool_center={tool_center_mm}, "
                    f"planned_feed={planned_feed_hist_mm_s[i]:.3f} mm/s, "
                    f"x={state[0:2, 0]}, "
                    f"milling_force={milling_forces[:, 0]}"
                )
            else:
                print(
                    f"Sim state: {progress:5.1f}% t={t:.3f} s, "
                    f"tool_center={tool_center_mm}, "
                    f"planned_feed={planned_feed_hist_mm_s[i]:.3f} mm/s, "
                    f"milling_force={milling_forces[:, 0]}"
                )

    if SAVE_EXPERIMENT_DATA:
        settings_hash, params_dir = save_params(params=params, base_dir=PARAMS_DIR)

        exp_dir, csv_path, run_id = save_experiment_csv(
            params={
                "dt": params["dt"],
                "rpm": params["rpm"],
                "feed_speed_mm_s": params["feed_speed_mm_s"],
            },
            settings_hash=settings_hash,
            output_dir=OUTPUT_DIR,
            t_hist=t_hist,
            x_hist=x_hist,
            u_hist=u_hist,
            fmill_hist=fmill_hist,
            fanalyt_hist=None,
            fzero_hist=None,
            tool_center_nominal_hist_mm=tool_center_nominal_hist_mm,
            tool_center_actual_hist_mm=tool_center_actual_hist_mm,
            tool_orientation_hist=phi_tool_hist,
        )

        path_trace_csv = Path(exp_dir) / "path_trace.csv"
        pd.DataFrame(
            {
                "time_s": t_hist,
                "position_along_path_mm": pos_on_path_mm,
                "planned_feed_mm_s": planned_feed_hist_mm_s,
                "tool_center_nominal_x_mm": tool_center_nominal_hist_mm[:, 0],
                "tool_center_nominal_y_mm": tool_center_nominal_hist_mm[:, 1],
                "tool_center_x_mm": tool_center_actual_hist_mm[:, 0],
                "tool_center_y_mm": tool_center_actual_hist_mm[:, 1],
                "fmill_x_N": fmill_hist[:, 0],
                "fmill_y_N": fmill_hist[:, 1],
            }
        ).to_csv(path_trace_csv, index=False)

        write_to_experimental_log(
            params=params,
            settings_hash=settings_hash,
            run_id=run_id,
            exp_dir=exp_dir,
            csv_path=csv_path,
            params_dir=params_dir,
            log_path="data/experimental_log.txt",
        )

        print("Saved experiment CSV to:", csv_path)
        print("Saved path trace CSV to:", path_trace_csv)
