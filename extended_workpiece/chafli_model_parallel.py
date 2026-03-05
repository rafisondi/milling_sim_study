import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp

from milling_path import MillingPath
from eraser_of_matter import milling_workpiece
from GeopmetryStandalone import Area2D  # T_inv unused here
from datetime import date

# =========================
# Finetuning / Config
# =========================

DT =  1e-5 # 1e-5 #1/4.8*1e-4 #
Samples_per_Period = 360
Period = DT * Samples_per_Period
frequency = 1 / Period
rpm = 60 * frequency
t_limit = 10

PRINT_EVERY_N = 100


FOLDER = os.path.join("data", "experiments")

OFFSET = None
OFFSET_SIDE = "left"

TOOL_DIAMETER_MM = 20.0
TOOL_RADIUS_MM = TOOL_DIAMETER_MM / 2.0
AXIAL_CUTTING_DEPTH_MM = 5.0

FEED_SPEED = 20.0     # [mm/s]
RPM_TARGET = rpm #3333.0 #/ 10 # 35
Z_TEETH = 1
SPINDLE_SPIN = -1


SIM_NAME = f"Model_{FEED_SPEED}mmps_{RPM_TARGET}rpm_{int(1/DT)}Hz_TRAJECTORY_{date.today():%Y%m%d%mm}"
print(SIM_NAME)
def spindle_from_rpm_and_feed(feed_mm_s: float, rpm: float, z_teeth: int, spin_dir: int):
    rev_per_s = rpm / 60.0
    omega = spin_dir * 2.0 * np.pi * rev_per_s          # [rad/s]
    fz_mm = feed_mm_s / (rev_per_s * z_teeth)           # [mm/tooth]
    return omega, fz_mm


def chafli_model_preparation(x_offset: float = 0.0, y_offset: float = 0.0, sim_name_suffix: str = "", log_prefix: str = ""):
    def log(msg: str):
        # Prefix + flush so output appears promptly from each process
        print(f"{log_prefix}{msg}", flush=True)

    # ---- Load path CSV
    milling_path_df = pd.read_csv(
        os.path.join("data", "paths", "Workpiece_long_with_start_milling_path copy.csv"),
        index_col=0
    )

    milling_path_df["x"] = milling_path_df["x"] + x_offset
    milling_path_df["y"] = milling_path_df["y"] + y_offset

    milling_path = MillingPath(
        milling_path_df.to_numpy(),
        feed_curve=FEED_SPEED,
        offset=OFFSET,
        offset_side=OFFSET_SIDE
    )

    # ---- Load initial workpiece polygon
    workpiece_Area2D = Area2D.load_from_disk(
        os.path.join("data", "paths", "Workpiece_long_with_start.pickle")
    )
    boundary_points_vec = workpiece_Area2D.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    workpiece_vertices = boundary_points.T  # shape (2, N)

    # ---- Create milling workpiece
    workpiece = milling_workpiece(workpiece_vertices, AXIAL_CUTTING_DEPTH_MM)

    workpiece.diameter_end_mill = TOOL_DIAMETER_MM
    workpiece.radius_tool = TOOL_RADIUS_MM
    workpiece.number_of_teeth = Z_TEETH
    workpiece.axial_cutting_depth = AXIAL_CUTTING_DEPTH_MM

    omega, fz_mm = spindle_from_rpm_and_feed(
        feed_mm_s=FEED_SPEED,
        rpm=RPM_TARGET,
        z_teeth=Z_TEETH,
        spin_dir=SPINDLE_SPIN
    )

    log(f"Feed speed = {FEED_SPEED:.3f} mm/s")
    log(f"RPM target = {RPM_TARGET:.2f} rev/min")
    log(f"Omega = {omega:.6f} rad/s")
    log(f"Derived fz = {fz_mm:.6f} mm/tooth")

    # ---- Path sampling (trajectory)
    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []

    step_dist = FEED_SPEED * DT
    path_len = milling_path.length()

    while pos_on_path <= path_len:
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += step_dist

    trajectory_local = np.asarray(trajectory_local)
    pos_on_path_log = np.asarray(pos_on_path_log)

    t_eval = np.arange(trajectory_local.shape[0], dtype=float) * DT

    force_log_xyz = np.zeros((3, trajectory_local.shape[0]), dtype=float)

    for i, t in enumerate(t_eval):
        spindle_orientation = t * omega
        workpiece.erase_step(
            trajectory_local[i],
            spindle_orientation,
            direction=SPINDLE_SPIN,
            t=t
        )

        milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1, 1, 1])
        force_log_xyz[:, i] = milling_wrench_WP[3:6]

        if i % PRINT_EVERY_N == 0:
            pct = (i / (len(t_eval) - 1) * 100.0) if len(t_eval) > 1 else 100.0
            log(f"{pct:.2f}%")

        # if t > t_limit:
        #     break

    # ---- Save results
    os.makedirs(FOLDER, exist_ok=True)

    def fmt(v: float) -> str:
        return f"{v:+.4f}".replace(".", "p")

    filename = f"{SIM_NAME}{sim_name_suffix}_dx{fmt(x_offset)}_dy{fmt(y_offset)}.csv"
    path = os.path.join(FOLDER, filename)

    df = pd.DataFrame({
        "Time": t_eval,
        "Position along path": pos_on_path_log,
        "x tool center WP": trajectory_local[:, 0],
        "y tool center WP": trajectory_local[:, 1],
        "x milling force WP": force_log_xyz[0, :],
        "y milling force WP": force_log_xyz[1, :],
        "z milling force WP": force_log_xyz[2, :],
    })
    df.to_csv(path, index=False)
    log(f"Saved: {path}")


def _worker_run_one_offset(args):
    dx, dy, idx = args
    pid = os.getpid()
    prefix = f"[run {idx} pid {pid} dx={dx:+.3f} dy={dy:+.3f}] "
    chafli_model_preparation(x_offset=dx, y_offset=dy, sim_name_suffix="", log_prefix=prefix)
    return (dx, dy)


def run_5_offsets_parallel(h: float, n_procs: int | None = None):
    offsets = [
        (0.0, 0.0),
        ( h,  0.0),
        (-h,  0.0),
        (0.0,  h),
        (0.0, -h),
    ]
    tasks = [(dx, dy, i) for i, (dx, dy) in enumerate(offsets)]

    # Use spawn for safety/portability (esp. when native libs are involved)
    ctx = mp.get_context("spawn")

    # If you want to cap CPU usage, set n_procs explicitly.
    # Default: min(#tasks, cpu_count)
    if n_procs is None:
        n_procs = min(len(tasks), os.cpu_count() or 1)

    # Make stdout unbuffered-ish (flush on prints already, but this helps in some environments)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    with ctx.Pool(processes=n_procs) as pool:
        # imap_unordered yields as jobs finish; worker prints appear live
        for _ in pool.imap_unordered(_worker_run_one_offset, tasks):
            pass


if __name__ == "__main__":
    h = 0.3 # 0.05
    run_5_offsets_parallel(h, n_procs=5)  # or omit n_procs to auto-pick