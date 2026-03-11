import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from eraser_of_matter import milling_workpiece
from milling_data import generate_chafli_trajectory, load_workpiece_vertices_from_pickle
from utils.save_utils import save_experiment_csv, save_params, write_to_experimental_log

DT = 1e-4
SAMPLES_PER_PERIOD = 720
PRINT_EVERY_N = 2000
SAVE_EXPERIMENT_DATA = True

MILLING_PATH_CSV = "data/paths/Workpiece_long_with_start_milling_path.csv"
WORKPIECE_PICKLE = "data/paths/Workpiece_long_with_start.pickle"

AXIAL_CUTTING_DEPTH_MM = 1.0
FEED_SPEED_MM_S = 40.0
SPINDLE_SPIN = -1
Z_TEETH = 4

TRAJECTORY_OFFSET_MM = 0.5
ENABLED_TRAJECTORY_RUNS = ("origin", "left", "right")

OUTPUT_DIR = "data/experiments"
PARAMS_DIR = "data/experiments/settings"

rotation_period_s = DT * SAMPLES_PER_PERIOD
frequency_hz = 1.0 / max(rotation_period_s, 1e-12)
rpm = 60.0 * frequency_hz
rev_per_s = rpm / 60.0
omega = SPINDLE_SPIN * 2.0 * np.pi * rev_per_s
feed_per_tooth_mm = FEED_SPEED_MM_S / max(rev_per_s * Z_TEETH, 1e-12)

params_base = {
    "dt": DT,
    "samples_per_period": SAMPLES_PER_PERIOD,
    "period": rotation_period_s,
    "frequency_hz": frequency_hz,
    "rpm": rpm,
    "rev_per_s": rev_per_s,
    "omega_rad_s": omega,
    "spindle_spin": SPINDLE_SPIN,
    "z_teeth": Z_TEETH,
    "feed_per_tooth_mm": feed_per_tooth_mm,
    "feed_speed_mm_s": FEED_SPEED_MM_S,
    "feed_speed_path_mm_s": FEED_SPEED_MM_S,
    "axial_cutting_depth_mm": AXIAL_CUTTING_DEPTH_MM,
    "milling_path_csv": MILLING_PATH_CSV,
    "workpiece_pickle": WORKPIECE_PICKLE,
    "trajectory_offset_mm": TRAJECTORY_OFFSET_MM,
}
def offset_by_normals(path_xy: np.ndarray, offset: float, side: str) -> np.ndarray:
    x, y = path_xy[:, 0], path_xy[:, 1]
    tx, ty = np.gradient(x), np.gradient(y)
    n = np.sqrt(tx**2 + ty**2)
    n = np.maximum(n, 1e-12)
    tx, ty = tx / n, ty / n

    nx, ny = -ty, tx
    s = 1.0 if side == "left" else -1.0

    return np.column_stack((x + s * offset * nx, y + s * offset * ny))


def run_trajectory(label, traj_mm, path_log, vertices):
    side = {
        "origin": {"path_offset_mm": 0.0, "path_offset_side": "none"},
        "left": {"path_offset_mm": TRAJECTORY_OFFSET_MM, "path_offset_side": "left"},
        "right": {"path_offset_mm": TRAJECTORY_OFFSET_MM, "path_offset_side": "right"},
    }[label]

    params = {**params_base, **side, "trajectory_variant": label}

    process = milling_workpiece(vertices, axial_cutting_depth=AXIAL_CUTTING_DEPTH_MM)
    process.number_of_teeth = params["z_teeth"]
    process.slice_height = params["axial_cutting_depth_mm"]

    N = len(traj_mm)
    t_hist = np.arange(N) * DT
    f_hist = np.zeros((N, 2))
    phi_hist = np.zeros(N)

    print(f"\n=== Trajectory: {label} ===")
    print(f"Samples: {N}, dt={DT:.1e}, RPM={params['rpm']:.1f}")

    for i in range(N):
        t = t_hist[i]
        phi = t * params["omega_rad_s"]

        process.erase_step(traj_mm[i], phi, direction=params["spindle_spin"], t=t)
        f_hist[i] = process.total_milling_force[:2] if i > 1 else 0
        phi_hist[i] = phi

        if i % PRINT_EVERY_N == 0:
            pct = 100 * i / max(N - 1, 1)
            print(f"[{label}] {pct:5.1f}% t={t:.3f}s force={f_hist[i]} pos={traj_mm[i]}")

    if not SAVE_EXPERIMENT_DATA:
        return

    settings_hash, params_dir = save_params(params=params, base_dir=PARAMS_DIR)

    exp_dir, csv_path, run_id = save_experiment_csv(
        params={"dt": params["dt"], "rpm": params["rpm"], "feed_speed_mm_s": params["feed_speed_mm_s"], "trajectory_variant": label},
        settings_hash=settings_hash,
        output_dir=OUTPUT_DIR,
        t_hist=t_hist,
        x_hist=np.zeros((N, 2)),
        u_hist=np.zeros((N, 2)),
        fmill_hist=f_hist,
        fanalyt_hist=None,
        fzero_hist=None,
        tool_center_nominal_hist_mm=traj_mm,
        tool_center_actual_hist_mm=traj_mm,
        tool_orientation_hist=phi_hist,
    )

    tagged_csv = Path(exp_dir) / f"experiment_data_{label}.csv"
    shutil.copy2(csv_path, tagged_csv)

    trace_path = Path(exp_dir) / f"path_trace_{label}.csv"
    pd.DataFrame({
        "time_s": t_hist,
        "position_along_path_mm": path_log,
        "tool_center_x_mm": traj_mm[:, 0],
        "tool_center_y_mm": traj_mm[:, 1],
        "fmill_x_N": f_hist[:, 0],
        "fmill_y_N": f_hist[:, 1],
    }).to_csv(trace_path, index=False)

    write_to_experimental_log(
        params=params,
        settings_hash=settings_hash,
        run_id=run_id,
        exp_dir=exp_dir,
        csv_path=tagged_csv,
        params_dir=params_dir,
        log_path="data/experimental_log.txt",
    )

    print("Saved experiment:", tagged_csv)


if __name__ == "__main__":
    vertices = load_workpiece_vertices_from_pickle(WORKPIECE_PICKLE)
    traj_origin, pos_log = generate_chafli_trajectory(
        MILLING_PATH_CSV,
        axial_cutting_depth_mm=AXIAL_CUTTING_DEPTH_MM,
        feed_speed_mm_s=FEED_SPEED_MM_S,
        dt=DT,
    )

    trajectories = {
        "origin": traj_origin,
        "left": offset_by_normals(traj_origin, TRAJECTORY_OFFSET_MM, "left"),
        "right": offset_by_normals(traj_origin, TRAJECTORY_OFFSET_MM, "right"),
    }

    invalid = [l for l in ENABLED_TRAJECTORY_RUNS if l not in trajectories]
    if invalid:
        raise ValueError(f"Unknown trajectory labels: {invalid}. Allowed: {tuple(trajectories.keys())}")

    for label in ENABLED_TRAJECTORY_RUNS:
        run_trajectory(label, trajectories[label], pos_log, vertices)
