import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eraser_of_matter import milling_workpiece
from GeopmetryStandalone import Area2D
from utils.save_utils import save_experiment_csv, save_params, write_to_experimental_log


# Make extended_workpiece/ importable when running this file from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENDED_WORKPIECE_DIR = REPO_ROOT / "extended_workpiece"
if str(EXTENDED_WORKPIECE_DIR) not in sys.path:
    sys.path.append(str(EXTENDED_WORKPIECE_DIR))

from milling_path import MillingPath  # noqa: E402


# -------------------------
# Global config
# -------------------------
DT = 1e-4
PRINT_EVERY_N = 2000
SAVE_EXPERIMENT_DATA = True

# Chafli geometry/path assets
MILLING_PATH_CSV = "data/paths/Workpiece_long_with_start_milling_path.csv"
WORKPIECE_PICKLE = "data/paths/Workpiece_long_with_start.pickle"

# Process settings
AXIAL_CUTTING_DEPTH_MM = 5.0
FEED_SPEED_PATH_MM_S = 100.0
FEED_PER_TOOTH_MM = 0.09
FEED_SPEED_REV_COMP_MM_S = 40.0
SPINDLE_SPIN = -1
Z_TEETH = 1

# Optional path offset 
PATH_X_OFFSET_MM = 0.0
PATH_Y_OFFSET_MM = 0.0
PATH_OFFSET_MM = None
PATH_OFFSET_SIDE = "left"

# Save locations
OUTPUT_DIR = "data/experiments"
PARAMS_DIR = "data/experiments/settings"


# ----- derived parameters -----
rev_per_s = FEED_SPEED_REV_COMP_MM_S / max(FEED_PER_TOOTH_MM * Z_TEETH, 1e-12)
rpm = 60.0 * rev_per_s
omega = SPINDLE_SPIN * 2.0 * np.pi * rev_per_s


# ----- experiment dictionary -----
params = {
    # simulation
    "dt": DT,

    # spindle / process
    "rpm": rpm,
    "rev_per_s": rev_per_s,
    "omega_rad_s": omega,
    "spindle_spin": SPINDLE_SPIN,
    "z_teeth": Z_TEETH,
    "feed_per_tooth_mm": FEED_PER_TOOTH_MM,
    "feed_speed_rev_comp_mm_s": FEED_SPEED_REV_COMP_MM_S,

    # trajectory feed
    "feed_speed_path_mm_s": FEED_SPEED_PATH_MM_S,

    # tool / cut
    "axial_cutting_depth_mm": AXIAL_CUTTING_DEPTH_MM,

    # geometry/path
    "milling_path_csv": MILLING_PATH_CSV,
    "workpiece_pickle": WORKPIECE_PICKLE,
    "path_x_offset_mm": PATH_X_OFFSET_MM,
    "path_y_offset_mm": PATH_Y_OFFSET_MM,
    "path_offset_mm": PATH_OFFSET_MM,
    "path_offset_side": PATH_OFFSET_SIDE,
}


def load_workpiece_vertices_from_pickle(pickle_path: str) -> np.ndarray:
    workpiece_area = Area2D.load_from_disk(pickle_path)
    boundary_points_vec = workpiece_area.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    return boundary_points.T


def generate_chafli_trajectory(path_csv: str) -> tuple[np.ndarray, np.ndarray]:
    milling_path_df = pd.read_csv(path_csv, index_col=0)
    milling_path_df["x"] = milling_path_df["x"] + PATH_X_OFFSET_MM
    milling_path_df["y"] = milling_path_df["y"] + PATH_Y_OFFSET_MM

    milling_path = MillingPath(
        milling_path_df.to_numpy(),
        axial_cutting_dept=AXIAL_CUTTING_DEPTH_MM,
        feed_curve=FEED_SPEED_PATH_MM_S,
        offset=PATH_OFFSET_MM,
        offset_side=PATH_OFFSET_SIDE,
    )

    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []

    while pos_on_path <= milling_path.length():
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += FEED_SPEED_PATH_MM_S * DT

    return np.asarray(trajectory_local), np.asarray(pos_on_path_log)


if __name__ == "__main__":
    workpiece_vertices = load_workpiece_vertices_from_pickle(WORKPIECE_PICKLE)
    trajectory_local_mm, pos_on_path_mm = generate_chafli_trajectory(MILLING_PATH_CSV)

    milling_process = milling_workpiece(workpiece_vertices, axial_cutting_depth=AXIAL_CUTTING_DEPTH_MM)
    milling_process.number_of_teeth = params["z_teeth"]
    milling_process.slice_height = params["axial_cutting_depth_mm"]

    N = len(trajectory_local_mm)
    t_hist = np.arange(0, N * DT, DT)

    x_hist = np.zeros((N, 2))
    u_hist = np.zeros((N, 2))
    fmill_hist = np.zeros((N, 2))
    tool_center_nominal_hist_mm = trajectory_local_mm.copy()
    tool_center_actual_hist_mm = trajectory_local_mm.copy()
    phi_tool_hist = np.zeros(N)

    print("\n=== Trajectory Milling Simulation ===")
    print(f"Samples: {N}")
    print(f"dt: {DT:.1e} s")
    print(f"RPM: {params['rpm']:.1f}")
    print(f"Path feed speed: {FEED_SPEED_PATH_MM_S:.3f} mm/s")

    for i in range(N):
        t = t_hist[i]
        spindle_angle = t * params["omega_rad_s"]

        milling_process.erase_step(
            trajectory_local_mm[i],
            spindle_angle,
            direction=params["spindle_spin"],
            t=t,
        )

        if i > 1:
            milling_forces = milling_process.total_milling_force[0:2]
        else:
            milling_forces = np.zeros(2)

        fmill_hist[i, :] = milling_forces
        phi_tool_hist[i] = spindle_angle

        if i % PRINT_EVERY_N == 0:
            progress = 100.0 * i / max(1, N - 1)
            print(
                f"Sim state: {progress:5.1f}% t={t:.3f} s, "
                f"tool_center={trajectory_local_mm[i]}, "
                f"milling_force={milling_forces}"
            )

    if SAVE_EXPERIMENT_DATA:
        settings_hash, params_dir = save_params(params=params, base_dir=PARAMS_DIR)

        exp_dir, csv_path, run_id = save_experiment_csv(
            params={
                "dt": params["dt"],
                "rpm": params["rpm"],
                "feed_speed_mm_s": params["feed_speed_path_mm_s"],
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
