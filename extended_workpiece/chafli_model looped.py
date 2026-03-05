import os
import numpy as np
import pandas as pd

from milling_path import MillingPath
from eraser_of_matter import milling_workpiece
from GeopmetryStandalone import Area2D  # T_inv unused here
# from feed_optimizer2 import FeedOptimizer  # unused here

# =========================
# Finetuning / Config
# =========================

# --- Time resolution ---
DT = 1e-5              # [s] simulation time step
PRINT_EVERY_N = 100      # print progress every N steps

# --- Output ---
SIM_NAME = f"Chaflimodel_100mmps_3333rpm_{int(1/DT)}Hz_with_start_TRAJECTORY"
FOLDER = os.path.join("data", "experiments")

# --- Feed / path ---
# FEED_SPEED = 100.0       # [mm/s] tool center feed along path
OFFSET = None
OFFSET_SIDE = "left"     # 'left' or 'right'

# --- Tool geometry ---
TOOL_DIAMETER_MM = 20.0
TOOL_RADIUS_MM = TOOL_DIAMETER_MM / 2.0
AXIAL_CUTTING_DEPTH_MM = 5.0

# --- Spindle / cutting ---
FEED_SPEED = 10.0     # [mm/s]
RPM_TARGET = 3333.0 /5    # [rev/min]
Z_TEETH = 4
SPINDLE_SPIN = -1



def spindle_from_rpm_and_feed(feed_mm_s: float, rpm: float, z_teeth: int, spin_dir: int):
    rev_per_s = rpm / 60.0
    omega = spin_dir * 2.0 * np.pi * rev_per_s          # [rad/s]
    fz_mm = feed_mm_s / (rev_per_s * z_teeth)           # [mm/tooth]
    return omega, fz_mm



def chafli_model_preparation(x_offset: float = 0.0, y_offset: float = 0.0, sim_name_suffix: str = ""):
    # ---- Load path CSV
    milling_path_df = pd.read_csv(
        os.path.join("data", "paths", "Workpiece_long_with_start_milling_path.csv"),
        index_col=0
    )

    # Apply offsets (same units as CSV columns)
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

    # Keep these consistent with your tuning knobs
    workpiece.diameter_end_mill = TOOL_DIAMETER_MM
    workpiece.radius_tool = TOOL_RADIUS_MM
    workpiece.number_of_teeth = Z_TEETH
    workpiece.axial_cutting_depth = AXIAL_CUTTING_DEPTH_MM

    # ---- Spindle speed (compute ONCE)
    omega, fz_mm = spindle_from_rpm_and_feed(
        feed_mm_s=FEED_SPEED,
        rpm=RPM_TARGET,
        z_teeth=Z_TEETH,
        spin_dir=SPINDLE_SPIN
    )

    print(f"Feed speed = {FEED_SPEED:.3f} mm/s")
    print(f"RPM target = {RPM_TARGET:.2f} rev/min")
    print(f"Omega = {omega:.6f} rad/s")
    print(f"Derived fz = {fz_mm:.6f} mm/tooth")


    # ---- Path sampling (trajectory)
    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []

    step_dist = FEED_SPEED * DT  # mm per time step along path
    path_len = milling_path.length()

    while pos_on_path <= path_len:
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += step_dist

    trajectory_local = np.asarray(trajectory_local)
    pos_on_path_log = np.asarray(pos_on_path_log)

    # ---- Time vector matches trajectory length exactly
    t_eval = np.arange(trajectory_local.shape[0], dtype=float) * DT

    # ---- Force simulation loop (WP-frame)
    force_log_xyz = np.zeros((3, trajectory_local.shape[0]), dtype=float)

    for i, t in enumerate(t_eval):
        spindle_orientation = t * omega  # [rad]
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
            print(f"{pct:.2f}%")

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
    print("Saved:", path)


def run_5_offsets(h: float):
    offsets = [
        (0.0, 0.0),
        ( h,  0.0),
        (-h,  0.0),
        (0.0,  h),
        (0.0, -h),
    ]
    for dx, dy in offsets:
        chafli_model_preparation(x_offset=dx, y_offset=dy, sim_name_suffix="")


if __name__ == "__main__":
    h = 0.05
    run_5_offsets(h)

# 
#  
#  import os
# import numpy as np
# import pandas as pd
# from milling_path import MillingPath
# from eraser_of_matter import milling_workpiece
# from GeopmetryStandalone import Area2D, T_inv
# from feed_optimizer2 import FeedOptimizer

# dt = 2.5e-5
# sim_name = f"Chaflimodel_100mmps_3333rpm_{int(1/dt)}Hz_with_start_TRAJECTORY"
# folder = os.path.join("data", "experiments")
# feed_speed = 100         # [mm/s]
# axial_cutting_depth = 5. # [mm]

# def compute_spindle(rev_comp_mm_s: float, fz_mm: float, z_teeth: int, spin_dir: int):
#     """
#     Returns:
#       rpm [rev/min]
#       omega [rad/s] (signed with spin_dir)
#     """
#     rpm = (60.0 / (fz_mm * z_teeth)) * rev_comp_mm_s
#     omega = spin_dir * rpm * 2.0 * np.pi / 60.0
#     return rpm, omega

# # --- Time resolution ---
# DT = 2.5e-5                     # [s] simulation time step
# PRINT_EVERY_N = 100             # print progress every N steps

# # --- Feed / path ---
# FEED_SPEED = 100.0              # [mm/s] tool center feed along path
# OFFSET = None
# OFFSET_SIDE = "left"

# # --- Tool geometry ---
# TOOL_DIAMETER_MM = 20.0         # [mm] <-- set this (radius derived below)
# TOOL_RADIUS_MM = TOOL_DIAMETER_MM / 2.0  # [mm]

# AXIAL_CUTTING_DEPTH_MM = 5.0    # [mm]

# # --- Spindle / cutting ---
# SPINDLE_SPIN = -1               # +1 / -1 direction
# FEED_PER_TOOTH_MM = 0.09        # [mm/tooth]
# Z_TEETH = 4                     # number of teeth (should match milling_workpiece.number_of_teeth)

# # Option A (your current approach): choose "equivalent feed speed" for RPM calc
# FEED_SPEED_REV_COMP_MM_S = 40.0 # [mm/s] used to compute RPM

# T_base_workpiece = np.array(
#     [[ 0.,  1.,  0., -7.         ],
#      [-1.,  0.,  0., -1667.40003 ],
#      [ 0.,  0.,  1.,  0.48569812 ],
#      [ 0.,  0.,  0.,  1.         ]]
# )

# spindle_spin = -1
# feed_per_tooth = 0.09       # [mm]
# feed_speed_rev_comp = 40    # [mm/s]

# offset = None
# offset_side = 'left'        # 'left' or 'right'


# def chafli_model_preparation(x_offset=0.0, y_offset=0.0, sim_name_suffix=""):
#     milling_path_df = pd.read_csv(
#         os.path.join("data", "paths", "Workpiece_long_with_start_milling_path.csv"),
#         index_col=0
#     )

#     # Apply offsets (in same units as the CSV path file columns)
#     milling_path_df["x"] = milling_path_df["x"] + x_offset
#     milling_path_df["y"] = milling_path_df["y"] + y_offset

#     milling_path = MillingPath(
#         milling_path_df.to_numpy(),
#         feed_curve=feed_speed,
#         offset=offset,
#         offset_side=offset_side
#     )

#     workpiece_Area2D = Area2D.load_from_disk(os.path.join("data", "paths", "Workpiece_long_with_start.pickle"))
#     boundary_points_vec = workpiece_Area2D.get_boundary_points()
#     boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
#     workpiece_vertices = np.transpose(boundary_points)
#     # workpiece = milling_workpiece(workpiece_vertices, milling_path.axial_cutting_depth)

#     workpiece = milling_workpiece(workpiece_vertices, milling_path.axial_cutting_depth)
#     workpiece.diameter_end_mill = TOOL_DIAMETER_MM
#     workpiece.radius_tool = TOOL_RADIUS_MM
#     workpiece.number_of_teeth = Z_TEETH
#     workpiece.axial_cutting_depth = AXIAL_CUTTING_DEPTH_MM

#     rpm, omega = compute_spindle(
#         rev_comp_mm_s=FEED_SPEED_REV_COMP_MM_S,
#         fz_mm=FEED_PER_TOOTH_MM,
#         z_teeth=workpiece.number_of_teeth,
#         spin_dir=SPINDLE_SPIN
#     )

#     print(f"dt = {DT} s  (Fs={1/DT:.1f} Hz)")
#     print(f"tool radius = {workpiece.radius_tool:.3f} mm  (diameter={2*workpiece.radius_tool:.3f} mm)")
#     print(f"rpm = {rpm:.2f} rev/min")
#     print(f"omega = {omega:.4f} rad/s")

#     # Path generation
#     pos_on_path = 0.0
#     trajectory_local = []
#     pos_on_path_log = []
#     while pos_on_path <= milling_path.length():
#         trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
#         pos_on_path_log.append(pos_on_path)
#         pos_on_path += feed_speed * dt
#     trajectory_local = np.array(trajectory_local)
#     pos_on_path_log = np.array(pos_on_path_log)

#     rev_per_min = 60 / (feed_per_tooth * workpiece.number_of_teeth) * feed_speed_rev_comp
#     omega = spindle_spin * rev_per_min * 2 * np.pi / 60
#     print("omega:", omega)

#     # Force simulation loop (in WP-frame)
#     force_log_xyz = np.zeros((3, len(trajectory_local)))
#     t_eval = np.arange(0, dt * len(trajectory_local), dt)
#     for i, t in enumerate(t_eval):
#         spindle_orientation = t * omega
#         workpiece.erase_step(trajectory_local[i], spindle_orientation, direction=spindle_spin, t=t)
#         milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1 , 1 , 1])
#         force_log_xyz[:, i] = milling_wrench_WP[3:6]

#         if i % 100 == 0:
#             print(f"{t/t_eval[-1] * 100:.2f}%")

#     os.makedirs(folder, exist_ok=True)

#     # Build filename that encodes offsets (avoid '.' in filenames by formatting)
#     def fmt(v):
#         return f"{v:+.4f}".replace(".", "p")  # +0p0100 etc

#     filename = f"{sim_name}{sim_name_suffix}_dx{fmt(x_offset)}_dy{fmt(y_offset)}.csv"
#     path = os.path.join(folder, filename)

#     df = pd.DataFrame({
#         "Time": t_eval,
#         "Position along path": pos_on_path_log,
#         "x tool center WP": trajectory_local[:, 0],
#         "y tool center WP": trajectory_local[:, 1],
#         "x milling force WP": force_log_xyz[0, :],
#         "y milling force WP": force_log_xyz[1, :],
#         "z milling force WP": force_log_xyz[2, :],
#     })
#     df.to_csv(path, index=False)
#     print("Saved:", path)


# def run_5_offsets(h):
#     # h_vec = [(0,0), (+h,0), (-h,0), (0,+h), (0,-h)]
#     offsets = [
#         (0.0, 0.0),
#         ( h,  0.0),
#         (-h,  0.0),
#         (0.0,  h),
#         (0.0, -h),
#     ]
#     for dx, dy in offsets:
#         chafli_model_preparation(x_offset=dx, y_offset=dy, sim_name_suffix="")


# if __name__ == "__main__":
#     h = 0.01
#     run_5_offsets(h)
