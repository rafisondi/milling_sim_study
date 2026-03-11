import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
from pathlib import Path

from utils.save_utils import save_experiment_csv, save_params, write_to_experimental_log
from utils.dynamics_utils import load_macro_from_npz, MacroOscillatorParams, dynamics, RungeKutta4
from eraser_of_matter import milling_workpiece
from GeopmetryStandalone import Area2D



# Make extended_workpiece/ importable when running this file from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENDED_WORKPIECE_DIR = REPO_ROOT / "extended_workpiece"
if str(EXTENDED_WORKPIECE_DIR) not in sys.path:
    sys.path.append(str(EXTENDED_WORKPIECE_DIR))

from milling_path import MillingPath 


# -------------------------
# Global config
# -------------------------
DT = 1e-4
SAMPLES_PER_PERIOD = 360 
PRINT_EVERY_N = 2000
SAVE_EXPERIMENT_DATA = True

# Chafli geometry/path assets
MILLING_PATH_CSV = "data/paths/Workpiece_long_with_start_milling_path.csv"
WORKPIECE_PICKLE = "data/paths/Workpiece_long_with_start.pickle"

# Process settings
AXIAL_CUTTING_DEPTH_MM = 1.0
FEED_SPEED_MM_S = 40.0
SPINDLE_SPIN = -1
Z_TEETH = 4

PATH_OFFSET_MM = None
PATH_OFFSET_SIDE = "left"

# Save locations
OUTPUT_DIR = "data/experiments"
PARAMS_DIR = "data/experiments/settings"
LINEARIZED_MODEL_NPZ = "linearized_operational_space_xyz.npz"


# ----- derived parameters -----
rotation_period_s = DT * SAMPLES_PER_PERIOD
frequency_hz = 1.0 / max(rotation_period_s, 1e-12)
rpm = 60.0 * frequency_hz
rev_per_s = rpm / 60.0
omega = SPINDLE_SPIN * 2.0 * np.pi * rev_per_s
feed_per_tooth_mm = FEED_SPEED_MM_S / max(rev_per_s * Z_TEETH, 1e-12)


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

    # tool / cut
    "axial_cutting_depth_mm": AXIAL_CUTTING_DEPTH_MM,

    # geometry/path
    "milling_path_csv": MILLING_PATH_CSV,
    "workpiece_pickle": WORKPIECE_PICKLE,
    "path_offset_mm": PATH_OFFSET_MM,
    "path_offset_side": PATH_OFFSET_SIDE,

    # compliant-base model
    "linearized_model_npz": LINEARIZED_MODEL_NPZ,
    "micro_mass_diag_kg": 80.0,
}


def load_workpiece_vertices_from_pickle(pickle_path: str) -> np.ndarray:
    workpiece_area = Area2D.load_from_disk(pickle_path)
    boundary_points_vec = workpiece_area.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    return boundary_points.T


def generate_chafli_trajectory(path_csv: str) -> tuple[np.ndarray, np.ndarray]:
    milling_path_df = pd.read_csv(path_csv, index_col=0)

    milling_path = MillingPath(
        milling_path_df.to_numpy(),
        axial_cutting_dept=AXIAL_CUTTING_DEPTH_MM,
        feed_curve=FEED_SPEED_MM_S,
        offset=PATH_OFFSET_MM,
        offset_side=PATH_OFFSET_SIDE,
    )

    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []

    while pos_on_path <= milling_path.length():
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += FEED_SPEED_MM_S * DT

    return np.asarray(trajectory_local), np.asarray(pos_on_path_log)



if __name__ == "__main__":
    M2, C2, K2, _ = load_macro_from_npz(LINEARIZED_MODEL_NPZ, axes=(0, 1))
    M_micro = np.eye(2) * params["micro_mass_diag_kg"]
    Mtot = M2 + M_micro
    p_osc = MacroOscillatorParams(M=Mtot, C=C2, K=K2)

    workpiece_vertices = load_workpiece_vertices_from_pickle(WORKPIECE_PICKLE)
    trajectory_local_mm, pos_on_path_mm = generate_chafli_trajectory(MILLING_PATH_CSV)

    milling_process = milling_workpiece(workpiece_vertices, axial_cutting_depth=AXIAL_CUTTING_DEPTH_MM)
    milling_process.number_of_teeth = params["z_teeth"]
    milling_process.slice_height = params["axial_cutting_depth_mm"]
    milling_forces = np.zeros((2, 1))

    N = len(trajectory_local_mm)
    t_hist = np.arange(N) * DT

    state = np.zeros((4, 1))  # [x; xdot] in meters
    u = np.zeros((2, 1))      # external input force
    x_hist = np.zeros((N, 2))
    u_hist = np.zeros((N, 2))
    fmill_hist = np.zeros((N, 2))
    tool_center_nominal_hist_mm = trajectory_local_mm.copy()
    tool_center_actual_hist_mm = np.zeros((N, 2))
    phi_tool_hist = np.zeros(N)

    print("\n=== Trajectory Milling Simulation ===")
    print(f"Samples: {N}")
    print(f"dt: {DT:.1e} s")
    print(f"Samples per period: {params['samples_per_period']}")
    print(f"RPM: {params['rpm']:.1f}")
    print(f"Path feed speed: {params['feed_speed_mm_s']:.3f} mm/s")
    print(f"Linearized model: {params['linearized_model_npz']}")
    print(f"M diag (kg): {np.diag(Mtot)}")
    print(f"C diag (N s/m): {np.diag(C2)}")
    print(f"K diag (N/m): {np.diag(K2)}")

    for i in range(N):
        t = t_hist[i]
        spindle_angle = t * params["omega_rad_s"]
        tool_center_nominal_mm = trajectory_local_mm[i]
        tool_center_mm = tool_center_nominal_mm + state[0:2, 0] * 1e3

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

        state = RungeKutta4(dynamics, t, state, DT, p_osc, u, milling_forces)

        fmill_hist[i, :] = milling_forces[:, 0]
        x_hist[i, :] = state[0:2, 0]
        u_hist[i, :] = u[:, 0]
        tool_center_actual_hist_mm[i, :] = tool_center_mm
        phi_tool_hist[i] = spindle_angle

        if i % PRINT_EVERY_N == 0:
            progress = 100.0 * i / max(1, N - 1)
            print(
                f"Sim state: {progress:5.1f}% t={t:.3f} s, "
                f"tool_center={tool_center_mm}, "
                f"x={state[0:2, 0]}, "
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
                "tool_center_x_mm": trajectory_local_mm[:, 0],
                "tool_center_y_mm": trajectory_local_mm[:, 1],
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
